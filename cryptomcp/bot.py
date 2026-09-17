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
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from . import manual, storage
from . import orderbook_watch as watch
from .orderbook import OrderBook, build_order_book
from .symbols import format_price

log = logging.getLogger("cryptomcp.bot")

TELEGRAM_API = "https://api.telegram.org"
POLL_TIMEOUT_S = 30
CALLBACK_LIMIT_BYTES = 64
DEFAULT_DAILY_BUDGET_USD = 1.0
DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_MORNING_TIME = "08:00"

# MOCKUPS.md §1.3: эти технические термины не должны попадать в сообщения
# человеку. Проверка живёт рядом с рендерами, чтобы ею пользовались и тесты,
# и проверочная команда Б11.
FORBIDDEN_WORDS = (
    "перцентиль", "pct", "atr", "bbw", "ema", "rsi", "poc", "value area",
    "имбаланс", "imbalance", "takerb", "oi", "squeeze_index", "дельта",
    "квадрант", "n/a",
)

# Пороги только показа, не торговой аналитики (MOCKUPS.md §3.1, §3.3).
RANK_MOVE_BADGE = 3
PRICE_MOVE_WARNING_PCT = 10.0
IMBALANCE_SMALL = 0.10
IMBALANCE_LARGE = 0.30

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

    def remember_watch(self, watch_id: str, chat_id: int, symbol: str, stamp_ms: int) -> None:
        con = self._con()
        try:
            con.execute(
                "INSERT INTO bot_watches(watch_id, chat_id, symbol, started_at) "
                "VALUES (?, ?, ?, ?)",
                (watch_id, chat_id, symbol, stamp_ms),
            )
            con.commit()
        finally:
            con.close()

    def unfinished_watches(self) -> list[dict[str, Any]]:
        con = self._con()
        try:
            return [
                dict(row)
                for row in con.execute(
                    "SELECT * FROM bot_watches WHERE finished_at IS NULL ORDER BY started_at"
                )
            ]
        finally:
            con.close()

    def finish_watch(self, watch_id: str, stamp_ms: int) -> None:
        con = self._con()
        try:
            con.execute(
                "UPDATE bot_watches SET finished_at = ? WHERE watch_id = ?",
                (stamp_ms, watch_id),
            )
            con.commit()
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


def short_symbol(symbol: str) -> str:
    """Показать тикер без USDT, не испортив китайское имя монеты."""
    return symbol.removesuffix("USDT")


def utc_stamp(stamp_ms: int, *, comma: bool = True) -> str:
    separator = ", " if comma else " "
    return dt.datetime.fromtimestamp(stamp_ms / 1000, dt.UTC).strftime(f"%d.%m{separator}%H:%M UTC")


def amount_usd(value: float) -> str:
    """Короткая денежная сумма для одного экрана телефона."""
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f} млн $"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f} тыс. $"
    return f"{value:.0f} $"


def forbidden_words(text: str) -> set[str]:
    lowered = text.lower()
    return {word for word in FORBIDDEN_WORDS if word in lowered}


def _move(row: dict[str, Any], scan: dict[str, Any]) -> float | None:
    entry = row.get("price_at_entry")
    current = scan.get("price")
    if not entry or not current:
        return None
    return (float(current) / float(entry) - 1) * 100


def render_watchlist_message(
    rows: list[dict[str, Any]], scans: dict[tuple[str, str], dict[str, Any]], *, now: int
) -> str:
    """Человеческий список из строк хранилища, а не из get_watchlist-текста."""
    if not rows:
        return (
            f"📋 Список наблюдения · {utc_stamp(now)}\n"
            "Сейчас сканер не держит монет в списке."
        )
    ordered = sorted(
        rows,
        key=lambda row: (row.get("last_rank") is None, row.get("last_rank") or 10_000),
    )[:12]
    lines = [
        f"📋 Список наблюдения · {utc_stamp(now)}",
        "Монеты с самым долгим и глубоким затишьем на дневном",
        "графике. Выше — сильнее.",
        "",
    ]
    has_up = has_down = has_warning = has_accumulation = False
    for position, row in enumerate(ordered, start=1):
        scan = scans.get((str(row["symbol"]), str(row["tf"]))) or {}
        bars = scan.get("narrow_bars")
        if bars:
            days = int(round(int(bars) * _timeframe_days(str(row["tf"]))))
            quiet = f"затишье {days} {_day_word(days)}"
        else:
            quiet = "затишье прервалось"
        movement = _move(row, scan)
        movement_text = f"{movement:+.1f}%" if movement is not None else "нет свежей цены"
        badges: list[str] = []
        before, current = row.get("rank_at_entry"), row.get("last_rank")
        if before is not None and current is not None:
            moved = int(before) - int(current)
            if moved >= RANK_MOVE_BADGE:
                badges.append("⬆")
                has_up = True
            elif moved <= -RANK_MOVE_BADGE:
                badges.append("⬇")
                has_down = True
        if movement is not None and abs(movement) > PRICE_MOVE_WARNING_PCT:
            badges.append("⚠️")
            has_warning = True
        if row.get("accumulation_score") == 1:
            badges.append("🟢")
            has_accumulation = True
        line = (
            f"{position:>2}. {short_symbol(str(row['symbol'])):<7} "
            f"{quiet} · с входа {movement_text}"
        )
        if badges:
            line += "   " + " ".join(badges)
        lines.append(line)
    manual_rows = [row for row in rows if str(row.get("entered_by", "")).startswith("manual")]
    if manual_rows:
        manual_text = " · ".join(short_symbol(str(row["symbol"])) for row in manual_rows[:3])
        lines += ["", f"Ваши: {manual_text}"]
    legend: list[str] = []
    if has_up or has_down:
        legend.append("⬆⬇ сдвинулась в списке на 3+ места с момента входа")
    if has_warning:
        legend.append("⚠️ цена ушла от входа больше чем на 10%")
    if has_accumulation:
        legend.append("🟢 при входе был признак тихого набора")
    if legend:
        lines += [""] + legend
    return "\n".join(lines)


def _timeframe_days(timeframe: str) -> float:
    return {"1w": 7, "1d": 1, "4h": 4 / 24, "1h": 1 / 24}.get(timeframe, 1)


def _day_word(value: int) -> str:
    last_two = value % 100
    if 11 <= last_two <= 14:
        return "дней"
    if value % 10 == 1:
        return "день"
    if 2 <= value % 10 <= 4:
        return "дня"
    return "дней"


def _balance_words(value: float | None, turnover: float) -> str:
    if turnover < 2_000_000:
        return "монета слишком тонкая, перевес ничего не значит"
    if value is None or abs(value) <= IMBALANCE_SMALL:
        return "примерно поровну"
    who = "покупателей" if value > 0 else "продавцов"
    strength = "немного больше" if abs(value) <= IMBALANCE_LARGE else "заметно больше"
    return f"{who} {strength}"


def render_order_book_message(book: OrderBook, symbol: str, *, precision: int = 6) -> str:
    """Шаблон стакана из OrderBook, посчитанного ядром, без разбора текста MCP."""
    price = format_price(book.last_price, precision)
    spread_steps = max(1, round(book.spread / max(book.best_ask - book.best_bid, 1e-12)))
    lines = [
        f"📊 Стакан {short_symbol(symbol)} · фьючерсы · "
        f"{utc_stamp(book.timestamp_ms, comma=False)}",
        f"Цена {price} · между покупкой и продажей {spread_steps} шаг цены",
        "",
    ]
    visible_ranges = [item for item in book.ranges if item.depth_pct in {1.0, 2.0}]
    for item in visible_ranges:
        if not item.fully_covered:
            coverage = min(item.bid_coverage_pct, item.ask_coverage_pct)
            lines.append(f"Дальше {coverage:.1f}% стакан не виден.")
            continue
        percent = int(item.depth_pct) if item.depth_pct.is_integer() else item.depth_pct
        lines += [
            f"В пределах {percent}%:",
            f"  покупают на {amount_usd(item.bid_notional_usdt)} · "
            f"продают на {amount_usd(item.ask_notional_usdt)}",
            f"  → {_balance_words(item.imbalance, book.turnover_24h_usdt)}",
            "",
        ]
    one_pct = next((item for item in book.ranges if item.depth_pct == 1.0), None)
    if one_pct and one_pct.fully_covered and book.turnover_24h_usdt > 0:
        share = (
            (one_pct.bid_notional_usdt + one_pct.ask_notional_usdt)
            / book.turnover_24h_usdt
            * 100
        )
        lines += [
            f"Весь стакан в пределах 1% — это {share:.0f}% суточного оборота",
            f"монеты ({amount_usd(book.turnover_24h_usdt)}).",
            "",
        ]
    lines += [
        "Крупные заявки: пока не оцениваем — нет истории",
        "для сравнения.",
    ]
    return "\n".join(lines)


def render_watch_summary(watch_row: dict[str, Any], *, now: int) -> str:
    """Итог сессии по classify_liquidity: только факты, без вывода о намерении."""
    con = watch.read_only()
    try:
        snapshots = watch.snapshots(con, str(watch_row["watch_id"]))
        end = int(snapshots[-1]["ts"]) if snapshots else int(watch_row["started_at"])
        trades = watch.trades(
            con, str(watch_row["watch_id"]), from_ts=int(watch_row["started_at"]), to_ts=end
        )
        gaps = watch.trade_gaps(
            con, str(watch_row["watch_id"]), from_ts=int(watch_row["started_at"]), to_ts=end
        )
    finally:
        if con is not None:
            con.close()
    outcomes = watch.classify_liquidity(snapshots, trades, gaps)["outcomes"]
    duration = max(0, end - int(watch_row["started_at"])) // 60_000
    lines = [
        f"👁 Наблюдение {short_symbol(str(watch_row['symbol']))} завершено · "
        f"{duration} мин · {len(snapshots)} снимков",
        "",
        "Что происходило с заявками рядом с ценой:",
    ]
    labels = (
        ("✅ устояли под ударами", "устоял"),
        ("✅ исполнены сделками", "исполнен"),
        ("↩️ убраны, когда цена подошла", "снят у цены"),
        ("🚩 убраны заранее, до подхода", "снят заранее"),
        ("🧊 похоже на скрытый объём", "похоже на айсберг"),
    )
    for label, key in labels:
        entries = outcomes[key]
        notional = sum(float(item.get("notional_usdt", 0)) for item in entries)
        suffix = f" · {amount_usd(notional)}" if key != "похоже на айсберг" else ""
        lines.append(f"  {label:<31} {len(entries):>3}{suffix}")
    if gaps:
        lines += ["", f"Пропусков в данных: {len(gaps)}, их интервалы не разобраны."]
    else:
        lines += ["", "Пропусков в данных: нет."]
    return "\n".join(lines)


def render_manual_message(rows: list[dict[str, Any]], *, now: int) -> str:
    lines = [f"⭐ Мои находки · {utc_stamp(now)}"]
    if not rows:
        return "\n".join(lines + ["Здесь пока нет ручных находок."])
    for row in rows:
        price = row.get("price_at_entry")
        note = row.get("note") or "без заметки"
        entry = f"по цене {format_price(float(price), 6)}" if price else "цена не сохранена"
        lines.append(f"{short_symbol(str(row['symbol']))} · {row['tf']} · {entry} · {note}")
    return "\n".join(lines)


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


class TemplateService:
    """Получает данные шаблонов из тех же структур, что и MCP-инструменты.

    Здесь специально нет вызова get_watchlist/get_order_book и разбора их
    текста. Первый изменяет слова для модели, второй — длинную выдачу; оба не
    являются контрактом между ботом и ядром (Б4).
    """

    async def watchlist(self) -> str:
        path = storage.DEFAULT_PATH
        if not os.path.exists(path):
            return "📋 Список наблюдения\nАрхив ещё не создан сборщиком."
        con = storage.connect(path, read_only=True)
        manual_con = manual.read_only()
        stamp = now_ms()
        try:
            manual_rows = manual.entries(manual_con, now_ms=stamp)
            rows = storage.episodes(con, limit=100)
            rows = _merge_manual_rows(rows, manual_rows)
            scans = {
                (item["symbol"], timeframe): item
                for timeframe in {str(row["tf"]) for row in rows}
                for item in storage.latest_scan(con, timeframe)
            }
            return render_watchlist_message(rows, scans, now=stamp)
        finally:
            con.close()
            if manual_con is not None:
                manual_con.close()

    async def book(self, symbol: str) -> str:
        # Импорт внутри метода не создаёт MCP-транспорт при проверочной
        # команде списка; это именно те же клиент, registry и build_order_book,
        # которыми пользуется сервер, но текст в чат никогда не переиспользуем.
        from . import server

        client, _, registry, _, market = await server._ctx("futures")
        info = await registry.get(symbol)
        snapshot = await client.order_book(info.symbol, limit=100)
        price, turnover, stamp = await asyncio.gather(
            client.ticker_price(info.symbol), client.ticker_24hr(info.symbol), client.now_ms()
        )
        book = build_order_book(
            snapshot,
            timestamp_ms=stamp,
            last_price=float(price["price"]),
            limit=100,
            depth_pcts=(1.0, 2.0),
            turnover_24h_usdt=float(turnover["quoteVolume"]),
        )
        # Сейчас кнопка определена для фьючерсов; рынок не скрываем в словах
        # рендера и не выдаём спотовый стакан за перпетуал.
        assert market.name == "futures"
        return render_order_book_message(book, info.symbol, precision=info.price_precision)

    async def manual(self) -> str:
        con = manual.read_only()
        try:
            return render_manual_message(manual.entries(con, now_ms=now_ms()), now=now_ms())
        finally:
            if con is not None:
                con.close()

    async def watch_summary(self, watch_id: str) -> str:
        con = watch.read_only()
        try:
            row = watch.get_watch(con, watch_id)
        finally:
            if con is not None:
                con.close()
        if row is None:
            return "👁 Данные наблюдения уже удалены после завершения."
        return render_watch_summary(row, now=now_ms())

    async def symbol_suggestions(self) -> list[str]:
        """Первые восемь монет списка для выбора, без обращения к тексту MCP."""
        if not os.path.exists(storage.DEFAULT_PATH):
            return []
        con = storage.connect(storage.DEFAULT_PATH, read_only=True)
        try:
            return [
                str(row["symbol"])
                for row in storage.episodes(con, limit=8)
                if row.get("last_rank") is not None
            ][:8]
        finally:
            con.close()

    async def add_manual(self, symbol: str, timeframe: str, note: str) -> str:
        """Ручная запись использует ту же отдельную БД, но не текст инструмента."""
        from . import server

        client, _, registry, _, _ = await server._ctx("futures")
        info = await registry.get(symbol)
        ticker, stamp = await asyncio.gather(client.ticker_24hr(info.symbol), client.now_ms())
        con = manual.connect()
        try:
            manual.add(
                con,
                info.symbol,
                timeframe,
                entered_at=stamp,
                note=note or None,
                price=float(ticker["lastPrice"]),
            )
            rows = manual.entries(con, now_ms=stamp)
        finally:
            con.close()
        return render_manual_message(rows, now=stamp)

    async def remove_manual(self, symbol: str, reason: str) -> str:
        from . import server

        client, _, registry, _, _ = await server._ctx("futures")
        info = await registry.get(symbol)
        stamp = await client.now_ms()
        con = manual.connect()
        try:
            manual.remove(con, info.symbol, removed_at=stamp, reason=reason or None)
            rows = manual.entries(con, now_ms=stamp)
        finally:
            con.close()
        return render_manual_message(rows, now=stamp)


def _merge_manual_rows(
    scanner_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Локальная, минимальная часть collector.merge_manual для шаблона.

    Не тянем весь collector (и его плановые зависимости) в обработчик
    нажатия. Смысл совпадает с существующим объединением: одна монета на ТФ —
    одна строка, сканер даёт ранг, ручная запись — заметку.
    """
    by_key = {(row["symbol"], row["tf"]): row for row in scanner_rows}
    merged = [dict(row) for row in scanner_rows]
    for entry in manual_rows:
        key = (entry["symbol"], entry["tf"])
        twin = by_key.get(key)
        if twin is not None:
            twin["note"] = entry.get("note")
            twin["entered_by"] = "scanner+manual"
            continue
        merged.append(
            {
                "symbol": entry["symbol"],
                "tf": entry["tf"],
                "entered_at": entry["entered_at"],
                "entered_by": "manual",
                "price_at_entry": entry.get("price_at_entry"),
                "last_rank": None,
                "rank_at_entry": None,
                "accumulation_score": None,
                "note": entry.get("note"),
            }
        )
    return merged


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
        templates: TemplateService | Any | None = None,
    ) -> None:
        self.config = config
        self.telegram = telegram or TelegramAPI(config.token)
        self.store = store or BotStore(config.database_path)
        self.templates = templates or TemplateService()
        self._offset: int | None = None
        self._pending_symbol: dict[int, str] = {}
        self._pending_note: dict[int, tuple[str, str, str]] = {}

    async def run(self) -> None:
        if not self.config.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
        await self._recover_interrupted_watches()
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
        note = self._pending_note.pop(user_id, None)
        if note is not None:
            symbol, timeframe, operation = note
            if operation == "add":
                reply = await self.templates.add_manual(symbol, timeframe, text)
            else:
                reply = await self.templates.remove_manual(symbol, text)
            await self.telegram.send_message(chat_id, reply, reply_markup=_manual_keyboard())
            return
        pending = self._pending_symbol.pop(user_id, None)
        if pending and text:
            await self._use_symbol(chat_id, user_id, pending, _normalise_symbol(text))
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
        notification_match = re.fullmatch(r"a:([^:]+):([^:]+):([abm])", action)
        if notification_match:
            symbol, _timeframe, target = notification_match.groups()
            if target == "b":
                await self._use_symbol(chat_id, user_id, "book", symbol)
            elif target == "m":
                await self._use_symbol(chat_id, user_id, "manual-add", symbol)
            else:
                await self._use_symbol(chat_id, user_id, "analyse", symbol)
            return
        timeframe_match = re.fullmatch(r"manual-tf-(1d|4h|1h):(.+)", action)
        if timeframe_match:
            timeframe, symbol = timeframe_match.groups()
            self._pending_note[user_id] = (symbol, timeframe, "add")
            await self.telegram.send_message(
                chat_id,
                f"Заметка к {short_symbol(symbol)} — зачем добавили?",
                force_reply=True,
            )
            return
        observe_match = re.fullmatch(r"observe-(15|30|60):(.+)", action)
        if observe_match:
            await self._start_watch(chat_id, observe_match.group(2), int(observe_match.group(1)))
            return
        if action.startswith("symbol-"):
            action_name, _, symbol = action.partition(":")
            await self._use_symbol(
                chat_id, user_id, action_name.removeprefix("symbol-"), symbol
            )
            return
        if action.startswith("type-"):
            self._pending_symbol[user_id] = action.removeprefix("type-")
            await self.telegram.send_message(
                chat_id, "Напишите тикер: можно ID или IDUSDT.", force_reply=True
            )
            return
        if action == "watchlist":
            text = await self.templates.watchlist()
            await self.telegram.send_message(chat_id, text, reply_markup=_watchlist_keyboard())
            return
        if action == "manual":
            text = await self.templates.manual()
            await self.telegram.send_message(chat_id, text, reply_markup=_manual_keyboard())
            return
        if action == "manual-add":
            await self._ask_symbol(chat_id, "manual-add")
            return
        if action == "manual-remove":
            await self._ask_symbol(chat_id, "manual-remove")
            return
        if action in {"analyse", "before", "book", "observe", "history"}:
            await self._ask_symbol(chat_id, action)
            return
        await self.telegram.send_message(chat_id, "Сценарий будет доступен после настройки данных.")

    async def _ask_symbol(self, chat_id: int, action: str) -> None:
        # Первые восемь берём из текущего списка только на этапе UI; когда
        # архива нет, ForceReply остаётся рабочим в личном чате и группе.
        suggestions = await self.templates.symbol_suggestions()
        keys = [
            {
                "text": f"🔍 {short_symbol(symbol)}",
                "callback_data": encode_callback(f"symbol-{action}", symbol),
            }
            for symbol in suggestions
        ]
        rows = [keys[index:index + 4] for index in range(0, len(keys), 4)]
        rows.append([
            {"text": "⌨️ Ввести тикер", "callback_data": encode_callback(f"type-{action}")}
        ])
        await self.telegram.send_message(
            chat_id,
            "Напишите тикер: можно ID или IDUSDT.",
            reply_markup=rows,
        )

    async def _use_symbol(self, chat_id: int, user_id: int, action: str, symbol: str) -> None:
        if action == "book":
            text = await self.templates.book(symbol)
            await self.telegram.send_message(chat_id, text, reply_markup=_book_keyboard(symbol))
            return
        if action == "observe":
            await self.telegram.send_message(
                chat_id,
                f"👁 Выберите длительность наблюдения за {short_symbol(symbol)}.",
                reply_markup=_observe_keyboard(symbol),
            )
            return
        if action == "manual-add":
            await self.telegram.send_message(
                chat_id,
                f"Добавить {short_symbol(symbol)}: выберите масштаб.",
                reply_markup=_manual_timeframe_keyboard(symbol),
            )
            return
        if action == "manual-remove":
            self._pending_note[user_id] = (symbol, "", "remove")
            await self.telegram.send_message(
                chat_id,
                f"Почему убрать {short_symbol(symbol)}? Можно коротко.",
                force_reply=True,
            )
            return
        await self.telegram.send_message(
            chat_id, "Сценарий с Claude будет добавлен на следующем этапе."
        )

    async def _start_watch(self, chat_id: int, symbol: str, duration_min: int) -> None:
        """Б8: запускает тот же тикер, но задача принадлежит процессу бота."""
        from . import server

        result = await server.start_order_book_watch(symbol, duration_min=duration_min)
        match = re.match(r"Сессия (watch_[0-9a-f]+) запущена", result)
        if match is None:
            await self.telegram.send_message(chat_id, result)
            return
        watch_id = match.group(1)
        self.store.remember_watch(watch_id, chat_id, symbol, now_ms())
        await self.telegram.send_message(
            chat_id, f"👁 Наблюдаю за {short_symbol(symbol)} {duration_min} мин. Итог пришлю сам."
        )
        asyncio.create_task(self._send_watch_summary(watch_id, chat_id, symbol))

    async def _send_watch_summary(self, watch_id: str, chat_id: int, symbol: str) -> None:
        from . import server

        task = server._watch_tasks.get(watch_id)
        try:
            if task is not None:
                await task
            text = await self.templates.watch_summary(watch_id)
            await self.telegram.send_message(
                chat_id, text, reply_markup=_watch_summary_keyboard(symbol)
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - итог не должен завершить polling
            log.warning("итог наблюдения %s не отправлен: %s", watch_id, error)
        finally:
            self.store.finish_watch(watch_id, now_ms())

    async def _recover_interrupted_watches(self) -> None:
        """После рестарта невозможно продолжить память процесса; не притворяемся."""
        for row in self.store.unfinished_watches():
            symbol = str(row["symbol"])
            await self.telegram.send_message(
                int(row["chat_id"]),
                f"Наблюдение за {short_symbol(symbol)} прервано перезапуском, запустить заново?",
                reply_markup=_observe_keyboard(symbol),
            )
            self.store.finish_watch(str(row["watch_id"]), now_ms())


def _normalise_symbol(text: str) -> str:
    symbol = text.strip().upper()
    if not symbol:
        raise ValueError("тикер пуст")
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def _watchlist_keyboard() -> list[list[dict[str, str]]]:
    return [
        [
            {"text": "Все монеты", "callback_data": encode_callback("watchlist")},
            {"text": "🌅 Утренний обзор 🧠", "callback_data": encode_callback("morning")},
        ]
    ]


def _manual_keyboard() -> list[list[dict[str, str]]]:
    return [[
        {"text": "➕ Добавить", "callback_data": encode_callback("manual-add")},
        {"text": "➖ Убрать", "callback_data": encode_callback("manual-remove")},
    ]]


def _manual_timeframe_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return [[
        {"text": "1d", "callback_data": encode_callback("manual-tf-1d", symbol)},
        {"text": "4h", "callback_data": encode_callback("manual-tf-4h", symbol)},
        {"text": "1h", "callback_data": encode_callback("manual-tf-1h", symbol)},
    ]]


def _book_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "text": "👁 Понаблюдать 15 мин",
                "callback_data": encode_callback("observe-15", symbol),
            },
            {"text": "30 мин", "callback_data": encode_callback("observe-30", symbol)},
            {"text": "60 мин", "callback_data": encode_callback("observe-60", symbol)},
        ],
        [{"text": "🧠 Объяснить", "callback_data": encode_callback("explain-book", symbol)}],
    ]


def _observe_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return [[
        {"text": "15 мин", "callback_data": encode_callback("observe-15", symbol)},
        {"text": "30 мин", "callback_data": encode_callback("observe-30", symbol)},
        {"text": "60 мин", "callback_data": encode_callback("observe-60", symbol)},
    ]]


def _watch_summary_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return [[
        {"text": "🧠 Объяснить", "callback_data": encode_callback("explain-watch", symbol)},
        {"text": "👁 Ещё 30 мин", "callback_data": encode_callback("observe-30", symbol)},
    ]]


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


async def _render_command(scenario: str, symbol: str | None) -> str:
    """Б11: тот же рендер, что в Telegram, но без токена и Bot API."""
    templates = TemplateService()
    aliases = {
        "watchlist": "watchlist",
        "list": "watchlist",
        "список": "watchlist",
        "book": "book",
        "стакан": "book",
        "manual": "manual",
        "мои": "manual",
    }
    chosen = aliases.get(scenario.lower())
    if chosen == "watchlist":
        return await templates.watchlist()
    if chosen == "manual":
        return await templates.manual()
    if chosen == "book":
        if not symbol:
            raise ValueError("для стакана нужен тикер: render book ARBUSDT")
        ticker = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
        return await templates.book(ticker)
    raise ValueError("сценарий: watchlist, book или manual")


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
