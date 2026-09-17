"""Каркас Telegram-бота: доступ и состояние проверяются без сети (Б1, Б3)."""

from __future__ import annotations

import asyncio

import pytest

from cryptomcp.bot import (
    Bot,
    BotConfig,
    BotStore,
    access_denied_text,
    encode_callback,
    forbidden_words,
    health,
    is_allowed,
    main_menu,
    render_manual_message,
    render_order_book_message,
    render_watch_summary,
    render_watchlist_message,
)
from cryptomcp.orderbook import build_order_book


class FakeTelegram:
    """Записывает ответы бота, не открывая соединение с Telegram."""

    def __init__(self):
        self.answers: list[tuple[str, str | None]] = []
        self.messages: list[dict] = []

    async def answer_callback(self, callback_id, text=None):
        self.answers.append((callback_id, text))

    async def send_message(self, chat_id, text, *, reply_markup=None, force_reply=False):
        message = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
            "force_reply": force_reply,
        }
        self.messages.append(message)
        return message


def message(user_id: int, text: str, *, chat_id: int = 777) -> dict:
    return {"message": {"from": {"id": user_id}, "chat": {"id": chat_id}, "text": text}}


def callback(user_id: int, action: str, *, chat_id: int = 777) -> dict:
    return {
        "callback_query": {
            "id": "callback-1",
            "from": {"id": user_id},
            "data": action,
            "message": {"chat": {"id": chat_id}},
        }
    }


@pytest.fixture
def bot(tmp_path):
    telegram = FakeTelegram()
    config = BotConfig(
        token="not-a-real-token",
        allowed_user_ids=(42,),
        database_path=str(tmp_path / "bot.sqlite"),
    )
    return Bot(config, telegram=telegram), telegram


@pytest.mark.asyncio
async def test_b3_id_works_for_everyone_but_other_command_is_rejected(bot):
    """Б3: /id не открывает доступ, зато владелец видит id без чтения логов."""
    worker, telegram = bot

    await worker.handle_update(message(7, "/id"))
    await worker.handle_update(message(7, "/start"))

    assert telegram.messages[0]["text"] == "Ваш id: 7"
    assert telegram.messages[1]["text"] == access_denied_text(7)


@pytest.mark.asyncio
async def test_b3_checks_callback_user_not_group_chat_and_answers_spinner_first(bot):
    """Кнопка в группе видна всем: решает from.id, а callback закрыт сразу."""
    worker, telegram = bot

    await worker.handle_update(callback(7, "watchlist", chat_id=-100123))

    assert telegram.answers == [("callback-1", None)]
    assert telegram.messages[-1]["text"] == access_denied_text(7)


@pytest.mark.asyncio
async def test_start_remembers_private_chat_of_allowed_owner_and_shows_exact_menu(bot):
    """Утренний обзор сможет писать только тому, кто уже сам начал чат."""
    worker, telegram = bot

    await worker.handle_update(message(42, "/start", chat_id=4242))

    assert worker.store.chat_for_user(42) == 4242
    assert telegram.messages[-1]["reply_markup"] == main_menu()
    assert telegram.messages[-1]["reply_markup"][0][0]["text"] == "📋 Список наблюдения"


def test_empty_allowlist_is_safe_default():
    """Пустая настройка не превращает нового бота в публичный анализатор."""
    assert not is_allowed(42, ())
    assert is_allowed(42, (42, 7))
    assert not is_allowed(7, (42,))


def test_callback_counts_utf8_bytes_for_longest_archive_symbol():
    """Б9: китайский тикер выглядит коротким, но Telegram лимитирует байты."""
    value = encode_callback("a", "币安人生USDT", "1d")

    assert len(value.encode("utf-8")) <= 64
    with pytest.raises(ValueError):
        encode_callback("a", "币" * 22, "1d")


def test_health_measures_successful_get_updates_not_process_existence(tmp_path, monkeypatch):
    """Б1: PID жив без удачного polling не является здоровым ботом."""
    path = str(tmp_path / "bot.sqlite")
    assert health(path)[0] is False
    store = BotStore(path)
    store.mark_poll_success(1_000_000)
    monkeypatch.setattr("cryptomcp.bot.now_ms", lambda: 1_030_000)

    assert health(path) == (True, "последний getUpdates 30 с назад")


def test_b4_watchlist_template_is_short_plain_language_and_legends_are_conditional():
    """Б4: бот строит список из полей базы, не из длинного текста MCP-инструмента."""
    rows = [
        {
            "symbol": "IDUSDT",
            "tf": "1d",
            "last_rank": 1,
            "rank_at_entry": 5,
            "price_at_entry": 10,
            "accumulation_score": 1,
            "entered_by": "scanner",
        },
        {
            "symbol": "EULUSDT",
            "tf": "1d",
            "last_rank": 2,
            "rank_at_entry": 2,
            "price_at_entry": 10,
            "accumulation_score": 0,
            "entered_by": "scanner",
        },
    ]
    scans = {
        ("IDUSDT", "1d"): {"narrow_bars": 21, "price": 11.9},
        ("EULUSDT", "1d"): {"narrow_bars": 9, "price": 9.7},
    }

    text = render_watchlist_message(rows, scans, now=1_789_870_000_000)

    assert "📋 Список наблюдения" in text
    assert "ID      затишье 21 день · с входа +19.0%   ⬆ ⚠️ 🟢" in text
    assert "⬆⬇ сдвинулась" in text
    assert "⚠️ цена ушла" in text
    assert "🟢 при входе" in text
    assert "\n 2. EUL" in text
    assert "\n 2. EUL     затишье 9 дней · с входа -3.0%   ⬇" not in text
    assert len(text) < 1500
    assert not forbidden_words(text)


def test_b4_order_book_template_uses_orderbook_data_and_hides_thin_market_bias():
    """Б4: суммы и покрытие получают из build_order_book, не регулярным выражением."""
    book = build_order_book(
        {
            "bids": [["100", "500"], ["99", "1000"], ["98", "1000"]],
            "asks": [["101", "200"], ["102", "1000"], ["103", "1000"]],
        },
        timestamp_ms=1_789_870_000_000,
        last_price=100.5,
        limit=100,
        depth_pcts=(1.0, 2.0),
        turnover_24h_usdt=1_500_000,
    )

    text = render_order_book_message(book, "IDUSDT", precision=2)

    assert "📊 Стакан ID · фьючерсы" in text
    assert "В пределах 1%:" in text
    assert "монета слишком тонкая, перевес ничего не значит" in text
    assert "Крупные заявки: пока не оцениваем" in text
    assert len(text) < 1500
    assert not forbidden_words(text)


def test_manual_template_keeps_note_without_market_tool_text():
    """Ручная заметка — часть находки, не строка, которую можно потерять в таблице."""
    text = render_manual_message(
        [
            {
                "symbol": "SKYAIUSDT",
                "tf": "4h",
                "price_at_entry": 0.123456,
                "note": "длинный коридор",
            }
        ],
        now=1_789_870_000_000,
    )

    assert "⭐ Мои находки" in text
    assert "SKYAI · 4h" in text
    assert "длинный коридор" in text
    assert not forbidden_words(text)


@pytest.mark.asyncio
async def test_b8_completed_bot_watch_sends_summary_to_starting_chat(monkeypatch, tmp_path):
    """Б8: тикер бота завершился — итог приходит сам ровно в чат запуска."""
    from cryptomcp import server

    class Templates:
        async def watch_summary(self, watch_id):
            assert watch_id == "watch_1234abcd"
            return "👁 Наблюдение ARB завершено · 30 мин · 12 снимков"

    telegram = FakeTelegram()
    worker = Bot(
        BotConfig(token="x", allowed_user_ids=(42,), database_path=str(tmp_path / "bot.sqlite")),
        telegram=telegram,
        templates=Templates(),
    )

    async def start(symbol, *, duration_min):
        """Бот зовёт функцию сессии, а не разбирает текст инструмента (Б4)."""
        assert (symbol, duration_min) == ("ARBUSDT", 30)
        return {"watch_id": "watch_1234abcd", "symbol": "ARBUSDT"}

    async def finished():
        return None

    monkeypatch.setattr(server, "begin_order_book_watch", start)
    server._watch_tasks["watch_1234abcd"] = asyncio.create_task(finished())
    await worker.handle_update(callback(42, "observe-30:ARBUSDT", chat_id=-10099))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert any(item["text"].startswith("👁 Наблюдаю") for item in telegram.messages)
    assert telegram.messages[-1]["chat_id"] == -10099
    assert telegram.messages[-1]["text"].startswith("👁 Наблюдение ARB завершено")
    assert worker.store.unfinished_watches() == []


@pytest.mark.asyncio
async def test_b8_restart_tells_the_original_chat_that_watch_must_restart(tmp_path):
    """Б8: завершать строку молча нельзя — у сессии после рестарта нет тикера."""
    telegram = FakeTelegram()
    worker = Bot(
        BotConfig(token="x", allowed_user_ids=(42,), database_path=str(tmp_path / "bot.sqlite")),
        telegram=telegram,
    )
    worker.store.remember_watch("watch_restart", -10042, "ARBUSDT", 1)

    await worker._recover_interrupted_watches()

    assert telegram.messages[0]["chat_id"] == -10042
    assert "прервано перезапуском" in telegram.messages[0]["text"]
    assert worker.store.unfinished_watches() == []


@pytest.mark.asyncio
async def test_manual_add_uses_ticker_timeframe_and_note_without_mcp_text(tmp_path):
    """Б4: ручная находка создаётся структурированным путём, включая заметку."""
    class Templates:
        async def symbol_suggestions(self):
            return ["IDUSDT"]

        async def add_manual(self, symbol, timeframe, note):
            assert (symbol, timeframe, note) == ("IDUSDT", "4h", "длинный коридор")
            return "⭐ Мои находки · ID"

    telegram = FakeTelegram()
    worker = Bot(
        BotConfig(token="x", allowed_user_ids=(42,), database_path=str(tmp_path / "bot.sqlite")),
        telegram=telegram,
        templates=Templates(),
    )

    await worker.handle_update(callback(42, "manual-add"))
    await worker.handle_update(callback(42, "symbol-manual-add:IDUSDT"))
    await worker.handle_update(callback(42, "manual-tf-4h:IDUSDT"))
    await worker.handle_update(message(42, "длинный коридор"))

    assert telegram.messages[-1]["text"] == "⭐ Мои находки · ID"
    assert telegram.messages[-1]["reply_markup"] is not None


def _liquid_book():
    """Стакан ликвидной монеты: сто уровней покрывают доли процента."""
    bids = [[f"{100 - i * 0.001:.3f}", "1000"] for i in range(100)]
    asks = [[f"{100.01 + i * 0.001:.3f}", "1000"] for i in range(100)]
    return build_order_book(
        {"bids": bids, "asks": asks},
        timestamp_ms=1_789_870_000_000,
        last_price=100.005,
        limit=1000,
        depth_pcts=(1.0, 2.0),
        turnover_24h_usdt=50_000_000,
    )


def test_incomplete_coverage_still_shows_the_sums():
    """Приёмка 17.09: у ARBUSDT шаблон печатал только «стакан не виден»."""
    text = render_order_book_message(_liquid_book(), "ARBUSDT", precision=3)

    assert "В пределах 1%:" in text
    assert "покупают на" in text and "продают на" in text
    assert "видно только до" in text
    assert not forbidden_words(text)


def test_spread_is_counted_in_price_steps_not_by_dividing_itself():
    """Приёмка 17.09: прежняя формула всегда печатала «1 шаг цены»."""
    book = build_order_book(
        {"bids": [["77.01", "10"], ["76.90", "10"]], "asks": [["77.15", "10"], ["77.30", "10"]]},
        timestamp_ms=1_789_870_000_000,
        last_price=77.05,
        limit=100,
        depth_pcts=(1.0,),
        turnover_24h_usdt=5_000_000,
    )

    text = render_order_book_message(book, "KODEX200USDT", precision=2, tick_size=0.01)

    assert "14 шагов цены" in text


def test_spread_without_tick_size_falls_back_to_percent():
    book = build_order_book(
        {"bids": [["100", "10"], ["99", "10"]], "asks": [["101", "10"], ["102", "10"]]},
        timestamp_ms=1_789_870_000_000,
        last_price=100.5,
        limit=100,
        depth_pcts=(1.0,),
        turnover_24h_usdt=5_000_000,
    )

    text = render_order_book_message(book, "IDUSDT", precision=2)

    assert "% цены" in text
    assert "шаг цены" not in text


def test_watch_summary_rounds_duration_up_and_declines_snapshots(monkeypatch):
    """Приёмка 17.09: двухминутная сессия из 24 снимков печаталась как «1 мин»."""
    from cryptomcp import bot as module

    start = 1_789_870_000_000
    snapshots = [{"ts": start + i * 5_000} for i in range(24)]

    class FakeWatch:
        WatchAdmissionError = RuntimeError

        @staticmethod
        def read_only():
            return None

        @staticmethod
        def snapshots(con, watch_id):
            return snapshots

        @staticmethod
        def trades(con, watch_id, *, from_ts, to_ts):
            return []

        @staticmethod
        def trade_gaps(con, watch_id, *, from_ts, to_ts):
            return []

        @staticmethod
        def classify_liquidity(snaps, trades, gaps):
            return {"outcomes": {
                "устоял": [], "исполнен": [], "снят у цены": [],
                "снят заранее": [], "похоже на айсберг": [],
            }}

    monkeypatch.setattr(module, "watch", FakeWatch)

    text = render_watch_summary(
        {"watch_id": "watch_1", "symbol": "ARBUSDT", "started_at": start}, now=start,
    )

    assert "2 мин" in text
    assert "24 снимка" in text


def test_manual_list_shows_move_since_it_was_added():
    """Ход цены — то, ради чего ручная запись и заводится (приёмка 17.09)."""
    text = render_manual_message(
        [{"symbol": "PHAUSDT", "tf": "4h", "price_at_entry": 0.027, "note": "пробой"}],
        now=1_789_870_000_000,
        prices={"PHAUSDT": 0.0297},
    )

    assert "с добавления +10.0%" in text
    assert "пробой" in text


def test_manual_list_without_a_live_price_says_so():
    text = render_manual_message(
        [{"symbol": "PHAUSDT", "tf": "4h", "price_at_entry": 0.027, "note": "пробой"}],
        now=1_789_870_000_000,
    )

    assert "нет свежей цены" in text


def test_watchlist_tail_shows_move_and_stale_mark_for_manual_rows():
    now = 1_789_870_000_000
    rows = [
        {"symbol": "IDUSDT", "tf": "1d", "entered_by": "scanner", "last_rank": 1,
         "rank_at_entry": 1, "price_at_entry": 0.03},
        {"symbol": "PHAUSDT", "tf": "4h", "entered_by": "manual", "last_rank": None,
         "rank_at_entry": None, "price_at_entry": 0.027},
    ]
    scans = {
        ("IDUSDT", "1d"): {"narrow_bars": 21, "price": 0.031,
                           "closed_through_ms": now - 3_600_000},
        ("PHAUSDT", "4h"): {"narrow_bars": 4, "price": 0.0297,
                            "closed_through_ms": now - 3 * 86_400_000},
    }

    text = render_watchlist_message(rows, scans, now=now)

    assert "Ваши: PHA +10.0%" in text
    assert "(данные от" in text
