"""Тесты доставки уведомлений (PLAN §4.24).

Два свойства важнее остальных и проверяются здесь в первую очередь:
доставка не поднимает исключений ни при каком ответе сети — иначе сбой
Telegram стоил бы часа открытого интереса, — и молчание, когда состав списка
не изменился, иначе ежечасное «изменений нет» превратит канал в шум.

В сеть тесты не ходят: httpx подменяется целиком.
"""

from __future__ import annotations

import pytest

from cryptomcp import notify
from cryptomcp.notify import (
    Telegram,
    notify_watchlist,
    render_watchlist_delta,
    tradingview_url,
    watchlist_delta_keyboard,
)

EMPTY: dict[str, list] = {"entered": [], "exited": [], "promoted": []}


def entry(symbol, tf, **fields) -> dict:
    """Запись дельты. Словарь, а не кортеж: полей пять."""
    return {"symbol": symbol, "tf": tf, "market": None,
            "rank": None, "narrow_bars": None, **fields}


class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeClient:
    """Подменяет httpx.AsyncClient: записывает вызовы, отвечает по сценарию."""

    calls: list[dict] = []
    script: list = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def post(self, url, json):
        type(self).calls.append({"url": url, "json": json})
        answer = self.script.pop(0) if self.script else FakeResponse(200)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def transport(monkeypatch):
    FakeClient.calls = []
    FakeClient.script = []
    monkeypatch.setattr(notify.httpx, "AsyncClient", FakeClient)
    # Пауза между попытками не должна растягивать прогон тестов.
    monkeypatch.setattr(notify, "RETRY_DELAY_S", 0)
    return FakeClient


class TestRender:
    def test_nothing_changed_means_no_message(self):
        """Молчание — не пустая строка, а отсутствие сообщения."""
        assert render_watchlist_delta(EMPTY) is None
        assert render_watchlist_delta({}) is None

    def test_all_three_sections(self):
        text = render_watchlist_delta({
            "entered": [entry("ONDOUSDT", "4h", rank=2, market="futures",
                              narrow_bars=19)],
            "promoted": [entry("HBARUSDT", "1d", rank=7, market="futures",
                               narrow_bars=7)],
            "exited": [entry("WLDUSDT", "4h", reason="пробой", market="spot")],
        }, now_ms=1_756_800_000_000)

        assert "Вошли (1)" in text
        assert "#ONDOUSDT 4h перп · ранг 2 · узк 19 (3.2 сут)" in text
        assert "Подтверждены (1)" in text
        assert "#HBARUSDT 1d перп · ранг 7 · узк 7 (7.0 сут)" in text
        assert "Вышли (1)" in text
        assert "#WLDUSDT 4h спот · пробой" in text

    def test_ticker_is_a_hashtag_not_a_link(self):
        """Нажатие на тикер собирает по каналу всю историю этой монеты.

        Обёрнутый в тег <a> хэштег перестаёт быть хэштегом, поэтому ссылка на
        график уехала в конец строки отдельным словом, а не пропала.
        """
        text = render_watchlist_delta(
            {"entered": [entry("ONDOUSDT", "4h", rank=2, market="futures")]}
        )

        assert "#ONDOUSDT" in text
        assert ">ONDOUSDT</a>" not in text
        assert ">график</a>" in text

    def test_symbol_is_a_link_to_the_same_timeframe(self):
        """Смысл ссылки — открыть тот же контракт и тот же таймфрейм."""
        text = render_watchlist_delta(
            {"entered": [entry("ONDOUSDT", "4h", rank=2, market="futures")]}
        )

        assert '<a href="https://www.tradingview.com/chart/?' in text
        assert "symbol=BINANCE%3AONDOUSDT.P" in text
        assert "interval=240" in text

    def test_link_follows_the_market_of_the_episode(self):
        """Отобрали по споту — вести на спот: ряды расходятся вдвое."""
        text = render_watchlist_delta(
            {"entered": [entry("HOMEUSDT", "4h", rank=2, market="spot")]}
        )

        assert "symbol=BINANCE%3AHOMEUSDT&" in text
        assert ".P" not in text
        assert "4h спот" in text

    def test_duration_is_printed_in_days_too(self):
        """«узк 19» на 4h выглядит внушительнее «узк 7» на 1d — а это 3.2 против 7."""
        text = render_watchlist_delta({
            "entered": [entry("A", "4h", rank=1, market="spot", narrow_bars=19),
                        entry("B", "1d", rank=2, market="spot", narrow_bars=7)],
        })

        assert "узк 19 (3.2 сут)" in text
        assert "узк 7 (7.0 сут)" in text

    def test_outside_text_is_escaped(self):
        """Причина выхода приходит извне: угловая скобка не должна стать тегом."""
        text = render_watchlist_delta(
            {"exited": [entry("WLDUSDT", "4h", reason="цена < уровня",
                              market="spot")]}
        )

        assert "цена &lt; уровня" in text

    def test_only_filled_sections_appear(self):
        text = render_watchlist_delta(
            {"exited": [entry("WLDUSDT", "4h", reason="истёк срок",
                              market="spot")]}
        )

        assert "Вышли" in text
        assert "Вошли" not in text
        assert "Подтверждены" not in text


class TestTradingViewUrl:
    """Промах в рынке тихо открывает соседний график — это стоит теста."""

    def test_perpetual_gets_the_suffix(self):
        url = tradingview_url("ONDOUSDT", market="futures")

        assert "symbol=BINANCE%3AONDOUSDT.P" in url

    def test_spot_has_no_suffix(self):
        assert "symbol=BINANCE%3AONDOUSDT&" in tradingview_url(
            "ONDOUSDT", "4h", "spot"
        )

    def test_unknown_market_goes_to_spot(self):
        """У старых эпизодов рынка нет; спотовый тикер есть у любой пары."""
        assert ".P" not in tradingview_url("ONDOUSDT", "4h", None)

    def test_known_timeframes(self):
        assert "interval=240" in tradingview_url("ONDOUSDT", "4h")
        assert "interval=D" in tradingview_url("ONDOUSDT", "1d")

    def test_unknown_timeframe_is_omitted(self):
        """Лучше график на умолчании биржи, чем ссылка, которую отвергнут."""
        assert "interval" not in tradingview_url("ONDOUSDT", "17m")
        assert "interval" not in tradingview_url("ONDOUSDT")


class TestFromEnv:
    """Отсутствие секретов — штатный режим, а не ошибка."""

    def test_both_set(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-5436206705")

        assert isinstance(Telegram.from_env(), Telegram)

    @pytest.mark.parametrize("token,chat", [("", "-1"), ("123:abc", ""), ("", "")])
    def test_partial_config_is_not_enough(self, monkeypatch, token, chat):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", chat)

        assert Telegram.from_env() is None

    def test_unset_variables(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        assert Telegram.from_env() is None


class TestSend:
    @pytest.mark.asyncio
    async def test_success(self, transport):
        assert await Telegram("123:abc", "-1").send("привет") is True
        assert len(transport.calls) == 1
        assert transport.calls[0]["url"].endswith("/bot123:abc/sendMessage")
        assert transport.calls[0]["json"]["chat_id"] == "-1"
        assert transport.calls[0]["json"]["text"] == "привет"

    @pytest.mark.asyncio
    async def test_client_error_is_not_retried(self, transport):
        """403 — бот не в группе. Повтор ничего не изменит, а задержит прогон."""
        transport.script = [FakeResponse(403, "bot was kicked")]

        assert await Telegram("123:abc", "-1").send("привет") is False
        assert len(transport.calls) == 1

    @pytest.mark.asyncio
    async def test_server_error_is_retried(self, transport):
        transport.script = [FakeResponse(502), FakeResponse(200)]

        assert await Telegram("123:abc", "-1").send("привет") is True
        assert len(transport.calls) == 2

    @pytest.mark.asyncio
    async def test_network_failure_never_raises(self, transport):
        """Главное свойство: сбой сети не роняет прогон сборщика."""
        transport.script = [OSError("сеть недоступна"), OSError("сеть недоступна")]

        assert await Telegram("123:abc", "-1").send("привет") is False
        assert len(transport.calls) == 2

    @pytest.mark.asyncio
    async def test_message_goes_as_html(self, transport):
        """Без parse_mode ссылка пришла бы сырым тегом."""
        await Telegram("123:abc", "-1").send("<a href=\"u\">X</a>")

        assert transport.calls[0]["json"]["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_long_message_is_trimmed(self, transport):
        await Telegram("123:abc", "-1").send("я" * 9000)

        sent = transport.calls[0]["json"]["text"]
        assert len(sent) <= 4096
        assert sent.endswith("список обрезан")

    @pytest.mark.asyncio
    async def test_trim_never_cuts_a_tag_in_half(self, transport):
        """Разрезанный тег — это 400 и потеря всего сообщения, а не хвоста."""
        line = '  + <a href="https://www.tradingview.com/chart/?symbol=X">SYM</a> 4h'
        await Telegram("123:abc", "-1").send("\n".join([line] * 200))

        sent = transport.calls[0]["json"]["text"]
        assert len(sent) <= 4096
        assert sent.count("<a ") == sent.count("</a>")

    @pytest.mark.asyncio
    async def test_empty_text_is_not_sent(self, transport):
        assert await Telegram("123:abc", "-1").send("") is False
        assert transport.calls == []


class TestNotifyWatchlist:
    @pytest.mark.asyncio
    async def test_silent_when_nothing_changed(self, transport, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1")

        assert await notify_watchlist(EMPTY) is False
        assert transport.calls == []

    @pytest.mark.asyncio
    async def test_sends_when_something_changed(self, transport, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1")

        assert await notify_watchlist(
            {"entered": [entry("CAKEUSDT", "4h", rank=3, market="spot")]}
        ) is True
        text = transport.calls[0]["json"]["text"]
        assert "CAKEUSDT" in text
        assert "tradingview.com" in text
        keyboard = transport.calls[0]["json"]["reply_markup"]["inline_keyboard"]
        assert keyboard[0][0]["text"] == "🔍 Разобрать CAKE 🧠"
        assert keyboard[0][1]["callback_data"] == "a:CAKEUSDT:4h:b"

    @pytest.mark.asyncio
    async def test_without_secrets_it_is_a_no_op(self, transport, monkeypatch):
        """Без бота сборщик работает молча — это не ошибка."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        assert await notify_watchlist(
            {"entered": [entry("CAKEUSDT", "4h", rank=3, market="spot")]}
        ) is False
        assert transport.calls == []


class TestAccumulationDelta:
    """Дельта второго списка уходит своим сообщением (PLAN §4.43)."""

    NOW = 1_788_400_000_000

    def test_silence_when_nothing_changed(self):
        from cryptomcp.notify import render_accumulation_delta

        assert render_accumulation_delta({"entered": [], "exited": []}) is None

    def test_entered_and_exited_are_named(self):
        from cryptomcp.notify import render_accumulation_delta

        text = render_accumulation_delta(
            {
                "entered": [{"symbol": "IDUSDT", "tf": "1d", "market": "spot",
                             "rank": 2, "narrow_bars": 21}],
                "exited": [{"symbol": "UBUSDT", "tf": "1d", "market": "futures",
                            "reason": "кластер пропал", "narrow_bars": 3}],
            },
            now_ms=self.NOW,
        )

        assert text.startswith("Накопление ·")
        assert "IDUSDT" in text and "ранг 2" in text
        assert "UBUSDT" in text and "кластер пропал" in text

    def test_it_is_not_mixed_into_the_watchlist_message(self):
        """Два списка — два сообщения: иначе не сравнить, какой полезнее."""
        from cryptomcp.notify import render_accumulation_delta, render_watchlist_delta

        changes = {"entered": [{"symbol": "IDUSDT", "tf": "1d", "market": "spot",
                                "rank": 1, "narrow_bars": 21}], "exited": [],
                   "promoted": []}

        assert "Накопление" in render_accumulation_delta(changes, now_ms=self.NOW)
        assert "Накопление" not in render_watchlist_delta(changes, now_ms=self.NOW)


def test_b9_callbacks_fit_telegram_byte_limit_for_chinese_symbol():
    """Б9: Telegram измеряет callback_data UTF-8-байтами, не буквами на экране."""
    keyboard = watchlist_delta_keyboard(
        {"entered": [entry("币安人生USDT", "1d", rank=1)], "promoted": []}
    )

    assert keyboard is not None
    assert all(
        len(button["callback_data"].encode("utf-8")) <= 64
        for row in keyboard for button in row
    )
