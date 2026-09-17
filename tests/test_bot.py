"""Каркас Telegram-бота: доступ и состояние проверяются без сети (Б1, Б3)."""

from __future__ import annotations

import pytest

from cryptomcp.bot import (
    Bot,
    BotConfig,
    BotStore,
    access_denied_text,
    encode_callback,
    health,
    is_allowed,
    main_menu,
)


class FakeTelegram:
    """Записывает ответы бота, не открывая соединение с Telegram."""

    def __init__(self):
        self.answers: list[tuple[str, str | None]] = []
        self.messages: list[dict] = []

    async def answer_callback(self, callback_id, text=None):
        self.answers.append((callback_id, text))

    async def send_message(self, chat_id, text, *, reply_markup=None, force_reply=False):
        message = {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
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
