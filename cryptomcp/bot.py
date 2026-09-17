"""Telegram-бот со сценариями поверх cryptomcp (PLAN §4.39).

Это отдельный процесс и отдельная SQLite-база.  Бот намеренно не пишет в
``market.sqlite``: ошибка обработки кнопки не должна затронуть часовой прогон
невосстановимых данных деривативов.  Сетевой транспорт здесь тонкий; правила
доступа, состояние пользователя и последующие сценарии держатся в обычных,
тестируемых функциях.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from . import storage

log = logging.getLogger("cryptomcp.bot")

TELEGRAM_API = "https://api.telegram.org"
POLL_TIMEOUT_S = 30
CALLBACK_LIMIT_BYTES = 64
DEFAULT_DAILY_BUDGET_USD = 1.0
DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_MORNING_TIME = "08:00"

# База лежит рядом с архивом в том же именованном томе.  Это не колонка в
# market.sqlite: тот файл буквально принадлежит только collector (Б1, Б6).
DEFAULT_PATH = os.environ.get("CRYPTOMCP_BOT_DB") or os.path.join(
    os.path.dirname(storage.DEFAULT_PATH) or ".", "bot.sqlite"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_users (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_expenses (
    id INTEGER PRIMARY KEY,
    created_at INTEGER NOT NULL,
    scenario TEXT NOT NULL,
    symbol TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS bot_expenses_created_at ON bot_expenses (created_at);

CREATE TABLE IF NOT EXISTS bot_watches (
    watch_id TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER
);
"""


def now_ms() -> int:
    """Unix-время в миллисекундах; вынесено для точных тестов дня и лимита."""
    return int(time.time() * 1000)


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Открыть собственную БД бота, не касаясь рыночного архива."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    con.commit()
    return con


@dataclass(frozen=True)
class BotConfig:
    """Настройка, которая не содержит и не логирует сами секреты."""

    token: str
    allowed_user_ids: tuple[int, ...]
    daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD
    timezone: str = DEFAULT_TIMEZONE
    morning_time: str = DEFAULT_MORNING_TIME
    database_path: str = DEFAULT_PATH

    @classmethod
    def from_env(cls) -> BotConfig:
        raw_ids = os.environ.get("BOT_ALLOWED_USER_IDS", "")
        ids: list[int] = []
        for part in raw_ids.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError as error:
                raise ValueError("BOT_ALLOWED_USER_IDS содержит нечисловой id") from error
        budget = float(os.environ.get("BOT_DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD))
        if budget < 0:
            raise ValueError("BOT_DAILY_BUDGET_USD не может быть отрицательным")
        timezone = os.environ.get("BOT_TIMEZONE", DEFAULT_TIMEZONE)
        ZoneInfo(timezone)  # Проверяем сейчас, а не в утреннем фоне.
        morning = os.environ.get("BOT_MORNING_TIME", DEFAULT_MORNING_TIME).strip()
        if morning:
            try:
                dt.time.fromisoformat(morning)
            except ValueError as error:
                raise ValueError("BOT_MORNING_TIME ожидается в формате HH:MM") from error
        return cls(
            token=(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip(),
            allowed_user_ids=tuple(ids),
            daily_budget_usd=budget,
            timezone=timezone,
            morning_time=morning,
            database_path=os.environ.get("CRYPTOMCP_BOT_DB", DEFAULT_PATH),
        )


class BotStore:
    """Маленькое хранилище состояния бота; единственный его писатель — бот."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path

    def _con(self) -> sqlite3.Connection:
        return connect(self.path)

    def set_meta(self, key: str, value: str) -> None:
        con = self._con()
        try:
            con.execute(
                "INSERT INTO bot_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            con.commit()
        finally:
            con.close()

    def get_meta(self, key: str) -> str | None:
        con = self._con()
        try:
            row = con.execute("SELECT value FROM bot_meta WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else None
        finally:
            con.close()

    def mark_poll_success(self, stamp_ms: int) -> None:
        self.set_meta("last_get_updates_at", str(stamp_ms))

    def last_poll_success(self) -> int | None:
        value = self.get_meta("last_get_updates_at")
        return int(value) if value is not None else None

    def remember_user(self, user_id: int, chat_id: int, stamp_ms: int) -> None:
        con = self._con()
        try:
            con.execute(
                "INSERT INTO bot_users(user_id, chat_id, started_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET chat_id = excluded.chat_id, "
                "updated_at = excluded.updated_at",
                (user_id, chat_id, stamp_ms, stamp_ms),
            )
            con.commit()
        finally:
            con.close()

    def chat_for_user(self, user_id: int) -> int | None:
        con = self._con()
        try:
            row = con.execute(
                "SELECT chat_id FROM bot_users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return int(row["chat_id"]) if row else None
        finally:
            con.close()


def is_allowed(user_id: int, allowed_user_ids: tuple[int, ...]) -> bool:
    """Одна точка решения Б3: пустой список означает никого не пускать."""
    return bool(allowed_user_ids) and user_id in allowed_user_ids


def access_denied_text(user_id: int) -> str:
    return f"Нет доступа, ваш id {user_id}."


def encode_callback(action: str, symbol: str | None = None, tf: str | None = None) -> str:
    """Компактная callback_data; Telegram считает UTF-8-байты, не символы."""
    values = [action]
    if symbol is not None:
        values.append(symbol.upper())
    if tf is not None:
        values.append(tf)
    result = ":".join(values)
    if len(result.encode("utf-8")) > CALLBACK_LIMIT_BYTES:
        raise ValueError("callback_data длиннее 64 байтов Telegram")
    return result


def main_menu() -> list[list[dict[str, str]]]:
    """Макет главного меню из MOCKUPS.md §2, без скрытых сценариев."""
    rows = (
        (("📋 Список наблюдения", "watchlist"), ("🌅 Утренний обзор 🧠", "morning")),
        (("🔍 Разбор монеты 🧠", "analyse"), ("✅ Перед сделкой 🧠", "before")),
        (("📊 Стакан", "book"), ("👁 Наблюдение за стаканом", "observe")),
        (("🕵 Тихий набор 🧠", "quiet"), ("⏪ Задним числом 🧠", "history")),
        (("📈 Итоги недели 🧠", "weekly"), ("⭐ Мои находки", "manual")),
    )
    return [
        [
            {"text": text, "callback_data": encode_callback(action)}
            for text, action in row
        ]
        for row in rows
    ]


class TelegramAPI:
    """Минимальный клиент Bot API; не повторяет логику сценариев."""

    def __init__(self, token: str, *, timeout: float = POLL_TIMEOUT_S + 10) -> None:
        self._token = token
        self._timeout = timeout

    async def call(self, method: str, payload: dict[str, Any]) -> Any:
        url = f"{TELEGRAM_API}/bot{self._token}/{method}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
        body = response.json()
        if response.status_code != 200 or not body.get("ok"):
            raise RuntimeError(
                f"Telegram {method}: {body.get('description', response.status_code)}"
            )
        return body.get("result")

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": POLL_TIMEOUT_S,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = await self.call("getUpdates", payload)
        return list(result or [])

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: list[list[dict[str, str]]] | None = None,
        force_reply: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_markup is not None:
            payload["reply_markup"] = {"inline_keyboard": reply_markup}
        elif force_reply:
            payload["reply_markup"] = {"force_reply": True, "selective": True}
        return dict(await self.call("sendMessage", payload))

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        payload: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
        await self.call("answerCallbackQuery", payload)


class Bot:
    """Long-polling процесс и общий диспетчер команд.

    Сценарии добавляются отдельными обработчиками, но доступ проверяется до
    них всегда. Это особенно важно для будущих вызовов Claude: чужое нажатие
    не должно тратить ни вес Binance, ни дневной бюджет.
    """

    def __init__(
        self,
        config: BotConfig,
        *,
        telegram: TelegramAPI | Any | None = None,
        store: BotStore | None = None,
    ) -> None:
        self.config = config
        self.telegram = telegram or TelegramAPI(config.token)
        self.store = store or BotStore(config.database_path)
        self._offset: int | None = None

    async def run(self) -> None:
        if not self.config.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
        while True:
            try:
                updates = await self.telegram.get_updates(self._offset)
                self.store.mark_poll_success(now_ms())
                for update in updates:
                    self._offset = int(update["update_id"]) + 1
                    await self.handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - сеть не должна завершить polling
                log.warning("getUpdates не удался: %s", error)
                await asyncio.sleep(2)

    async def handle_update(self, update: dict[str, Any]) -> None:
        if "callback_query" in update:
            await self._handle_callback(dict(update["callback_query"]))
        elif "message" in update:
            await self._handle_message(dict(update["message"]))

    async def _handle_message(self, message: dict[str, Any]) -> None:
        sender = message.get("from") or {}
        user_id = int(sender.get("id", 0))
        chat_id = int((message.get("chat") or {}).get("id", 0))
        text = str(message.get("text") or "").strip()
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command == "/id":
            await self.telegram.send_message(chat_id, f"Ваш id: {user_id}")
            return
        if not is_allowed(user_id, self.config.allowed_user_ids):
            await self.telegram.send_message(chat_id, access_denied_text(user_id))
            return
        if command in {"/start", "/menu"}:
            self.store.remember_user(user_id, chat_id, now_ms())
            await self.telegram.send_message(
                chat_id, "Выберите сценарий.", reply_markup=main_menu()
            )
            return
        await self.telegram.send_message(chat_id, "Нажмите /menu, чтобы выбрать сценарий.")

    async def _handle_callback(self, callback: dict[str, Any]) -> None:
        # Ответить надо ДО любой работы: Telegram иначе оставляет спиннер на
        # кнопке, а будущий Claude может отвечать десятки секунд (Б5).
        callback_id = str(callback.get("id", ""))
        await self.telegram.answer_callback(callback_id)
        sender = callback.get("from") or {}
        user_id = int(sender.get("id", 0))
        message = callback.get("message") or {}
        chat_id = int((message.get("chat") or {}).get("id", 0))
        if not is_allowed(user_id, self.config.allowed_user_ids):
            await self.telegram.send_message(chat_id, access_denied_text(user_id))
            return
        action = str(callback.get("data") or "")
        if action == "watchlist":
            await self.telegram.send_message(chat_id, "Список наблюдения готовится.")
            return
        if action == "manual":
            await self.telegram.send_message(chat_id, "Мои находки готовы.")
            return
        await self.telegram.send_message(chat_id, "Сценарий будет доступен после настройки данных.")


def health(
    path: str = DEFAULT_PATH, *, max_age_s: int = 2 * POLL_TIMEOUT_S + 30
) -> tuple[bool, str]:
    """Осмысленный healthcheck Б1: был ли успешный getUpdates, а не жив ли PID."""
    if not Path(path).exists():
        return False, "бот ещё не получил getUpdates"
    stamp = BotStore(path).last_poll_success()
    if stamp is None:
        return False, "бот ещё не получил getUpdates"
    age_s = (now_ms() - stamp) / 1000
    if age_s > max_age_s:
        return False, f"последний getUpdates {age_s:.0f} с назад"
    return True, f"последний getUpdates {age_s:.0f} с назад"


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("CRYPTOMCP_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _render_command(_scenario: str, _symbol: str | None) -> str:
    """Точка входа Б11; шаблоны добавляются на этапе 2 без Telegram-токена."""
    return "Проверочная команда сценариев будет доступна после добавления шаблонов."


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m cryptomcp.bot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("loop", help="слушать Telegram getUpdates")
    health_parser = sub.add_parser("health", help="проверить свежесть getUpdates")
    health_parser.add_argument("--db", default=os.environ.get("CRYPTOMCP_BOT_DB", DEFAULT_PATH))
    render_parser = sub.add_parser("render", help="напечатать шаблон без Telegram")
    render_parser.add_argument("scenario")
    render_parser.add_argument("symbol", nargs="?")
    args = parser.parse_args()
    _configure_logging()
    if args.command == "health":
        ok, text = health(args.db)
        print(text)
        raise SystemExit(0 if ok else 1)
    if args.command == "render":
        print(asyncio.run(_render_command(args.scenario, args.symbol)))
        return
    asyncio.run(Bot(BotConfig.from_env()).run())


if __name__ == "__main__":
    main()
