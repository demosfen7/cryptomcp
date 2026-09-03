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
from cryptomcp.notify import Telegram, notify_watchlist, render_watchlist_delta

EMPTY: dict[str, list] = {"entered": [], "exited": [], "promoted": []}


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
            "entered": [("ONDOUSDT", "4h", 2)],
            "promoted": [("HBARUSDT", "1d", 7)],
            "exited": [("WLDUSDT", "4h", "пробой")],
        }, now_ms=1_756_800_000_000)

        assert "Вошли (1)" in text
        assert "+ ONDOUSDT 4h · ранг 2" in text
        assert "Подтверждены (1)" in text
        assert "↑ HBARUSDT 1d · ранг 7" in text
        assert "Вышли (1)" in text
        assert "− WLDUSDT 4h · пробой" in text

    def test_only_filled_sections_appear(self):
        text = render_watchlist_delta({"exited": [("WLDUSDT", "4h", "истёк срок")]})

        assert "Вышли" in text
        assert "Вошли" not in text
        assert "Подтверждены" not in text


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
    async def test_long_message_is_trimmed(self, transport):
        await Telegram("123:abc", "-1").send("я" * 9000)

        sent = transport.calls[0]["json"]["text"]
        assert len(sent) <= 4096
        assert sent.endswith("список обрезан")

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

        assert await notify_watchlist({"entered": [("CAKEUSDT", "4h", 3)]}) is True
        assert "CAKEUSDT" in transport.calls[0]["json"]["text"]

    @pytest.mark.asyncio
    async def test_without_secrets_it_is_a_no_op(self, transport, monkeypatch):
        """Без бота сборщик работает молча — это не ошибка."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        assert await notify_watchlist({"entered": [("CAKEUSDT", "4h", 3)]}) is False
        assert transport.calls == []
