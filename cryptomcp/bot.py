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
import json
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

from . import assistant as claude
from . import manual, storage
from . import orderbook_watch as watch
from .errors import ToolError
from .notify import tradingview_url
from .orderbook import OrderBook, build_order_book
from .symbols import format_price

log = logging.getLogger("cryptomcp.bot")

TELEGRAM_API = "https://api.telegram.org"
POLL_TIMEOUT_S = 30
CALLBACK_LIMIT_BYTES = 64
DEFAULT_DAILY_BUDGET_USD = 1.0
DEFAULT_TIMEZONE = "Europe/Berlin"
DEFAULT_MORNING_TIME = "08:00"

#: Сколько часов после назначенного времени обзор ещё имеет смысл. Вечером он
#: уже не утренний, а пропущенное утро не досылается (Б7).
MORNING_WINDOW_H = 2

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


#: Глубина стакана для кнопки. Сто уровней у ликвидной монеты покрывают лишь
#: ±0.6% (замер ARBUSDT 17.09.2026), и шаблон оставался без единой суммы.
#: Тысяча уровней стоит 20 единиц веса вместо 5 — для разового нажатия это
#: приемлемо, а сессия наблюдения по-прежнему ходит сотней.
BOOK_LIMIT = 1000


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

    def add_expense(
        self,
        *,
        scenario: str,
        symbol: str | None,
        usage: Any,
        cost_usd: float,
        stamp_ms: int,
    ) -> None:
        """Записать расход по фактическому usage ответа, а не по оценке."""
        con = self._con()
        try:
            con.execute(
                "INSERT INTO bot_expenses(created_at, scenario, symbol, input_tokens, "
                "cache_creation_tokens, cache_read_tokens, output_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stamp_ms, scenario, symbol,
                    int(getattr(usage, "input_tokens", 0)),
                    int(getattr(usage, "cache_creation_tokens", 0)),
                    int(getattr(usage, "cache_read_tokens", 0)),
                    int(getattr(usage, "output_tokens", 0)),
                    float(cost_usd),
                ),
            )
            con.commit()
        finally:
            con.close()

    def spent_since(self, stamp_ms: int) -> float:
        """Сколько потрачено начиная с момента — границу дня считает вызывающий."""
        con = self._con()
        try:
            row = con.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM bot_expenses "
                "WHERE created_at >= ?",
                (stamp_ms,),
            ).fetchone()
            return float(row["total"] or 0.0)
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


def day_start_ms(timezone: str, stamp_ms: int) -> int:
    """Начало календарных суток в часовом поясе бота.

    День считается по месту жительства владельца, а не по UTC: «сегодня
    потрачено» должно совпадать с его сегодня.
    """
    zone = ZoneInfo(timezone)
    local = dt.datetime.fromtimestamp(stamp_ms / 1000, zone)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def budget_refusal_text(spent: float, budget: float, timezone: str) -> str:
    """Отказ по лимиту (MOCKUPS.md §4.5): что потрачено и когда обновится."""
    return (
        f"💸 Лимит на сегодня исчерпан: потрачено {claude.cost_words(spent)} "
        f"из ${budget:.0f}.\n"
        f"Кнопки без 🧠 работают. Лимит обновится в 00:00 ({timezone})."
    )


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


def chart_name(symbol: str, tf: str | None = None, market: str | None = None,
               width: int = 0) -> str:
    """Тикер, открывающий график по нажатию.

    В списке двенадцать монет, и двенадцать адресов целиком растянули бы
    сообщение на три экрана. Поэтому здесь адрес прячется в сам тикер, а
    видимым текстом он стоит там, где монета одна: под стаканом, под разбором
    и в уведомлении о дельте (замечание владельца 17.09.2026).

    Ширина добивается пробелами СНАРУЖИ ссылки: внутри они попали бы под
    подчёркивание и столбец поехал бы.
    """
    short = short_symbol(symbol)
    url = tradingview_url(symbol, tf, market)
    padding = " " * max(0, width - len(short))
    return f'<a href="{url}">{short}</a>{padding}'


def _stale_days(scan: dict[str, Any], now: int) -> str | None:
    """Насколько устарела запись скана, если она перестала обновляться."""
    closed = scan.get("closed_through_ms")
    if closed is None:
        return None
    if (now - int(closed)) / 86_400_000 < 1.5:
        return None
    moment = dt.datetime.fromtimestamp(int(closed) / 1000, dt.UTC)
    return f"от {moment.strftime('%d.%m')}"


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
            f"{position:>2}. "
            f"{chart_name(str(row['symbol']), str(row['tf']), row.get('market'), 7)} "
            f"{quiet} · с входа {movement_text}"
        )
        if badges:
            line += "   " + " ".join(badges)
        lines.append(line)
    manual_rows = [row for row in rows if str(row.get("entered_by", "")).startswith("manual")]
    if manual_rows:
        # Ход цены и возраст данных — то же, что у сканерных строк: без них
        # своя находка выглядит записью без судьбы (приёмка 17.09.2026).
        parts = []
        for row in manual_rows[:3]:
            scan = scans.get((str(row["symbol"]), str(row["tf"]))) or {}
            movement = _move(row, scan)
            stale = _stale_days(scan, now)
            text = chart_name(str(row["symbol"]), str(row["tf"]), row.get("market"))
            text += f" {movement:+.1f}%" if movement is not None else " нет свежей цены"
            if stale is not None:
                text += f" (данные {stale})"
            parts.append(text)
        lines += ["", "Ваши: " + " · ".join(parts)]
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


def _spread_words(book: OrderBook, tick_size: float) -> str:
    """Разрыв между покупкой и продажей — в шагах цены монеты.

    Прежняя формула делила разрыв сам на себя и всегда печатала «1 шаг»
    (приёмка 17.09.2026). Шаг берётся у биржи; без него честнее показать
    проценты, чем выдуманное число шагов.
    """
    if tick_size and tick_size > 0:
        steps = max(1, round(book.spread / tick_size))
        return f"{steps} {_step_word(steps)} цены"
    return f"{book.spread_pct:.2f}% цены"


def _step_word(value: int) -> str:
    last_two = value % 100
    if 11 <= last_two <= 14:
        return "шагов"
    if value % 10 == 1:
        return "шаг"
    if 2 <= value % 10 <= 4:
        return "шага"
    return "шагов"


def render_order_book_message(
    book: OrderBook, symbol: str, *, precision: int = 6, tick_size: float = 0.0
) -> str:
    """Шаблон стакана из OrderBook, посчитанного ядром, без разбора текста MCP.

    Неполное покрытие не выбрасывает блок, а дописывает строку «дальше стакан
    не виден»: у ликвидной монеты сто уровней покрывают ±0.6%, и прежний
    шаблон на ARBUSDT печатал две строки про невидимый стакан и ни одной
    суммы (приёмка 17.09.2026).
    """
    price = format_price(book.last_price, precision)
    lines = [
        f"📊 Стакан {short_symbol(symbol)} · фьючерсы · "
        f"{utc_stamp(book.timestamp_ms, comma=False)}",
        f"Цена {price} · между покупкой и продажей {_spread_words(book, tick_size)}",
        "",
    ]
    visible_ranges = [item for item in book.ranges if item.depth_pct in {1.0, 2.0}]
    for item in visible_ranges:
        percent = int(item.depth_pct) if item.depth_pct.is_integer() else item.depth_pct
        lines += [
            f"В пределах {percent}%:",
            f"  покупают на {amount_usd(item.bid_notional_usdt)} · "
            f"продают на {amount_usd(item.ask_notional_usdt)}",
            f"  → {_balance_words(item.imbalance, book.turnover_24h_usdt)}",
        ]
        if not item.fully_covered:
            coverage = min(item.bid_coverage_pct, item.ask_coverage_pct)
            lines.append(
                f"  видно только до {coverage:.1f}% — дальше заявок в ответе биржи нет"
            )
        lines.append("")
    one_pct = next((item for item in book.ranges if item.depth_pct == 1.0), None)
    if one_pct and book.turnover_24h_usdt > 0:
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
        "",
        tradingview_url(symbol, market="futures"),
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
    # Округление вверх: двухминутная сессия из 24 снимков печаталась как
    # «1 мин» (приёмка 17.09.2026).
    duration = -(-max(0, end - int(watch_row["started_at"])) // 60_000)
    # Снимок текущего состояния зовут по той же функции, что и итог: разница
    # только в слове. Обещать «завершено» идущей сессии нельзя.
    running = str(watch_row.get("status") or "") == "active"
    state = "идёт" if running else "завершено"
    lines = [
        f"👁 Наблюдение {short_symbol(str(watch_row['symbol']))} {state} · "
        f"{duration} мин · {len(snapshots)} {_snapshot_word(len(snapshots))}",
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


def _snapshot_word(value: int) -> str:
    last_two = value % 100
    if 11 <= last_two <= 14:
        return "снимков"
    if value % 10 == 1:
        return "снимок"
    if 2 <= value % 10 <= 4:
        return "снимка"
    return "снимков"


def render_manual_message(
    rows: list[dict[str, Any]],
    *,
    now: int,
    prices: dict[str, float] | None = None,
) -> str:
    """Свои находки с ходом цены от добавления.

    Ход — главное в этом списке: ради сравнения своего выбора со сканерным
    запись и заводится, а цена входа без текущей на вопрос не отвечает
    (приёмка 17.09.2026).
    """
    lines = [f"⭐ Мои находки · {utc_stamp(now)}"]
    if not rows:
        return "\n".join(lines + ["Здесь пока нет ручных находок."])
    prices = prices or {}
    for row in rows:
        price = row.get("price_at_entry")
        note = row.get("note") or "без заметки"
        if price:
            entry = f"по цене {format_price(float(price), 6)}"
            current = prices.get(str(row["symbol"]))
            move = (
                f"{(float(current) / float(price) - 1) * 100:+.1f}%"
                if current else "нет свежей цены"
            )
        else:
            entry, move = "цена не сохранена", "нет свежей цены"
        lines.append(
            f"{chart_name(str(row['symbol']), str(row['tf']))} · {row['tf']} · "
            f"{entry} · с добавления {move} · {note}"
        )
    return "\n".join(lines)


#: Главное меню одним списком: из него собираются и кнопки под полем ввода,
#: и inline-меню, и разбор нажатой подписи. Два списка разъехались бы на
#: первой же новой кнопке, и одна из них перестала бы работать молча.
MENU_ROWS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("📋 Список наблюдения", "watchlist"), ("🌅 Утренний обзор 🧠", "morning")),
    (("🔍 Разбор монеты 🧠", "analyse"), ("✅ Перед сделкой 🧠", "before")),
    (("📊 Стакан", "book"), ("👁 Наблюдение", "observe")),
    (("🕵 Тихий набор 🧠", "quiet"), ("⏪ Задним числом 🧠", "history")),
    (("📈 Итоги недели 🧠", "weekly"), ("⭐ Мои находки", "manual")),
)

#: Подпись кнопки под полем ввода приходит обычным текстом сообщения — по ней
#: и узнаём сценарий.
MENU_ACTIONS: dict[str, str] = {
    text: action for row in MENU_ROWS for text, action in row
}

BACK_LABEL = "⬅️ Главное меню"
HIDE_LABEL = "⌨️ Убрать кнопки"

#: Команды в кнопке «Меню» рядом с полем ввода. Нужны, чтобы меню было под
#: рукой и в группе, где кнопок под полем не бывает.
BOT_COMMANDS = (
    ("menu", "главное меню"),
    ("list", "список наблюдения"),
    ("book", "стакан монеты"),
    ("morning", "утренний обзор"),
    ("id", "показать мой id"),
)

COMMAND_ACTIONS = {"/list": "watchlist", "/book": "book", "/morning": "morning"}

#: Команды, принимающие тикер аргументом: `/book ICP`. В группе у ботов по
#: умолчанию включён режим приватности, и обычный текст до них не доходит —
#: только команды и ответы на их сообщения. Замечание владельца 17.09.2026:
#: тикер, написанный в группе обычным сообщением, пропал молча.
SYMBOL_COMMANDS = {
    "/book": "book",
    "/analyse": "analyse",
    "/before": "before",
    "/watch": "observe",
    "/add": "manual-add",
}


def main_menu() -> list[list[dict[str, str]]]:
    """Inline-меню: то же, что под полем ввода, но внутри сообщения."""
    return [
        [
            {"text": text, "callback_data": encode_callback(action)}
            for text, action in row
        ]
        for row in MENU_ROWS
    ]


def reply_menu() -> dict[str, Any]:
    """Кнопки под полем ввода: меню видно всегда, вызывать его не нужно.

    `is_persistent` оставляет клавиатуру на экране, `resize_keyboard` делает
    кнопки по высоте текста. Последней строкой — «убрать кнопки»: клавиатура
    занимает пол-экрана телефона, и у человека должен быть способ её свернуть
    прямо там, где она мешает.
    """
    rows = [[{"text": text} for text, _ in row] for row in MENU_ROWS]
    rows.append([{"text": HIDE_LABEL}])
    return {
        "keyboard": rows,
        "resize_keyboard": True,
        "is_persistent": True,
        "input_field_placeholder": "Выберите кнопку или напишите тикер",
    }


def hide_keyboard() -> dict[str, Any]:
    """Свернуть клавиатуру под полем ввода."""
    return {"remove_keyboard": True}


def with_back(
    rows: list[list[dict[str, str]]] | None = None,
) -> list[list[dict[str, str]]]:
    """Дописать возврат в главное меню последней строкой любого экрана."""
    result = [list(row) for row in (rows or [])]
    result.append([{"text": BACK_LABEL, "callback_data": encode_callback("menu")}])
    return result


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
        reply_markup: list[list[dict[str, str]]] | dict[str, Any] | None = None,
        force_reply: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if isinstance(reply_markup, dict):
            # Клавиатура под полем ввода приходит готовой структурой.
            payload["reply_markup"] = reply_markup
        elif reply_markup is not None:
            payload["reply_markup"] = {"inline_keyboard": reply_markup}
        elif force_reply:
            payload["reply_markup"] = {"force_reply": True, "selective": True}
        return dict(await self.call("sendMessage", payload))

    async def edit_message(self, chat_id: int, message_id: int, text: str,
                           *, reply_markup: list[list[dict[str, str]]] | None = None) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id, "message_id": message_id,
            "text": text, "parse_mode": "HTML",
        }
        if reply_markup is not None:
            payload["reply_markup"] = {"inline_keyboard": reply_markup}
        await self.call("editMessageText", payload)

    async def set_my_commands(self, commands: tuple[tuple[str, str], ...]) -> None:
        """Список команд в кнопке «Меню» — ставится один раз при старте."""
        await self.call("setMyCommands", {
            "commands": [
                {"command": name, "description": about} for name, about in commands
            ]
        })

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
        snapshot = await client.order_book(info.symbol, limit=BOOK_LIMIT)
        price, turnover, stamp = await asyncio.gather(
            client.ticker_price(info.symbol), client.ticker_24hr(info.symbol), client.now_ms()
        )
        book = build_order_book(
            snapshot,
            timestamp_ms=stamp,
            last_price=float(price["price"]),
            limit=BOOK_LIMIT,
            depth_pcts=(1.0, 2.0),
            turnover_24h_usdt=float(turnover["quoteVolume"]),
        )
        # Сейчас кнопка определена для фьючерсов; рынок не скрываем в словах
        # рендера и не выдаём спотовый стакан за перпетуал.
        assert market.name == "futures"
        return render_order_book_message(
            book, info.symbol, precision=info.price_precision,
            tick_size=float(getattr(info, "tick_size", 0.0) or 0.0),
        )

    async def manual(self) -> str:
        con = manual.read_only()
        try:
            rows = manual.entries(con, now_ms=now_ms())
        finally:
            if con is not None:
                con.close()
        return render_manual_message(
            rows, now=now_ms(), prices=await self._live_prices(rows)
        )

    @staticmethod
    async def _live_prices(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Живые цены по ручным записям: их монет может не быть в архиве вовсе.

        Ровно поэтому их и заводят руками (PLAN §4.28). Сбой цены не должен
        отменить весь список — тогда пропадёт и заметка, ради которой запись
        сделана.
        """
        if not rows:
            return {}
        from . import server

        try:
            client, _, _, _, _ = await server._ctx("futures")
            prices = {}
            for symbol in {str(row["symbol"]) for row in rows}:
                try:
                    ticker = await client.ticker_price(symbol)
                    prices[symbol] = float(ticker["price"])
                except Exception:  # noqa: BLE001 - цена не обязана быть
                    continue
            return prices
        except Exception:  # noqa: BLE001
            return {}

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


class _PendingStore:
    """Ожидаемый ответ пользователя, переживающий перезапуск бота."""

    def __init__(self, store: BotStore, kind: str) -> None:
        self._store = store
        self._kind = kind

    def _key(self, user_id: int) -> str:
        return f"pending:{self._kind}:{user_id}"

    def __setitem__(self, user_id: int, value: Any) -> None:
        self._store.set_meta(self._key(user_id), json.dumps(value, ensure_ascii=False))

    def pop(self, user_id: int, default: Any = None) -> Any:
        raw = self._store.get_meta(self._key(user_id))
        if raw is None:
            return default
        self._store.set_meta(self._key(user_id), "")
        try:
            value = json.loads(raw)
        except ValueError:
            return default
        if value is None:
            return default
        return tuple(value) if isinstance(value, list) else value


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
        assistant: Any | None = None,
    ) -> None:
        self.config = config
        self.telegram = telegram or TelegramAPI(config.token)
        self.store = store or BotStore(config.database_path)
        self.templates = templates or TemplateService()
        # Без ключа ассистента нет, и это штатный режим: кнопки без 🧠
        # работают, а кнопки с 🧠 честно говорят, что Claude не настроен.
        self.assistant = assistant if assistant is not None else claude.Assistant.from_env()
        self._offset: int | None = None
        # Шаг диалога хранится в базе, а не в памяти процесса: выкат между
        # вопросом и ответом стирал бы его, и сообщение пропадало бы молча.
        self._pending_symbol = _PendingStore(self.store, "symbol")
        self._pending_note = _PendingStore(self.store, "note")
        self._pending_minutes = _PendingStore(self.store, "minutes")
        self._busy: set[int] = set()
        self._last_texts: dict[int, tuple[str, str]] = {}

    async def run(self) -> None:
        if not self.config.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")
        await self._publish_commands()
        await self._recover_interrupted_watches()
        while True:
            try:
                updates = await self.telegram.get_updates(self._offset)
                self.store.mark_poll_success(now_ms())
                for update in updates:
                    self._offset = int(update["update_id"]) + 1
                    await self.handle_update(update)
                if await self._morning_due(now_ms()):
                    await self._send_morning()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - сеть не должна завершить polling
                log.warning("getUpdates не удался: %s", error)
                await asyncio.sleep(2)

    async def _publish_commands(self) -> None:
        """Список команд в кнопке «Меню». Сбой здесь не стоит опроса."""
        publish = getattr(self.telegram, "set_my_commands", None)
        if publish is None:
            return
        try:
            await publish(BOT_COMMANDS)
        except Exception as error:  # noqa: BLE001 - меню не важнее работы бота
            log.warning("не удалось опубликовать команды: %s", error)

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
        # Чат владельца запоминается при любом его обращении, а не только на
        # /start: иначе утренний обзор некуда слать после того, как человек
        # один раз нажал кнопку вместо команды.
        self.store.remember_user(user_id, chat_id, now_ms())
        if command in {"/start", "/menu"}:
            await self._show_menu(chat_id, greeting=command == "/start")
            return
        argument = text.split(maxsplit=1)[1].strip() if " " in text else ""
        if command in SYMBOL_COMMANDS and argument:
            await self._use_symbol(
                chat_id, user_id, SYMBOL_COMMANDS[command], _normalise_symbol(argument),
            )
            return
        if command in COMMAND_ACTIONS:
            await self._dispatch_action(chat_id, user_id, COMMAND_ACTIONS[command])
            return
        action = MENU_ACTIONS.get(text)
        if action is not None:
            await self._dispatch_action(chat_id, user_id, action)
            return
        if text == BACK_LABEL:
            await self._show_menu(chat_id)
            return
        if text == HIDE_LABEL:
            await self._hide_keyboard(chat_id)
            return
        minutes_for = self._pending_minutes.pop(user_id, None)
        if minutes_for is not None:
            digits = text.strip().replace("мин", "").strip()
            if not digits.isdigit() or not 1 <= int(digits) <= watch.MAX_DURATION_MIN:
                await self.telegram.send_message(
                    chat_id,
                    f"Нужно число от 1 до {watch.MAX_DURATION_MIN}. "
                    "Попробуйте ещё раз кнопкой «⌨️ Своё время».",
                    reply_markup=with_back(),
                )
                return
            await self._start_watch(chat_id, minutes_for, int(digits))
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
        # Ответ на просьбу «напишите тикер» узнаётся и без памяти о шаге:
        # в группе бот видит ответы на свои сообщения всегда, а состояние
        # могло не пережить выкат.
        replied = str(
            (message.get("reply_to_message") or {}).get("text") or ""
        )
        if ASK_SYMBOL_TEXT in replied and text:
            await self._use_symbol(chat_id, user_id, "analyse", _normalise_symbol(text))
            return
        await self._show_menu(chat_id)

    def _keyboard_hidden(self, chat_id: int) -> bool:
        return self.store.get_meta(f"keyboard_hidden:{chat_id}") == "1"

    def _remember_keyboard(self, chat_id: int, *, hidden: bool) -> None:
        self.store.set_meta(f"keyboard_hidden:{chat_id}", "1" if hidden else "0")

    async def _show_menu(
        self, chat_id: int, *, greeting: bool = False, keyboard: bool | None = None
    ) -> None:
        """Показать меню одним сообщением.

        Клавиатура под полем ввода присылается только когда её просят: на
        `/start` и по кнопке «показать кнопки». Иначе каждое возвращение в
        меню давало два сообщения подряд и то же самое меню дважды — в
        сообщении и под полем (замечание владельца 17.09.2026).
        """
        show_keyboard = greeting if keyboard is None else keyboard
        if show_keyboard:
            await self.telegram.send_message(
                chat_id,
                "Кнопки под полем ввода. Убрать их можно кнопкой "
                f"«{HIDE_LABEL}», вернуть — командой /menu.",
                reply_markup=reply_menu(),
            )
            self._remember_keyboard(chat_id, hidden=False)
        await self.telegram.send_message(
            chat_id, "Выберите сценарий.",
            reply_markup=main_menu() + [[self._keyboard_button(chat_id)]],
        )

    def _keyboard_button(self, chat_id: int) -> dict[str, str]:
        """Одна кнопка-переключатель: что сейчас можно сделать с клавиатурой."""
        if self._keyboard_hidden(chat_id):
            return {
                "text": "⌨️ Показать кнопки",
                "callback_data": encode_callback("kb-show"),
            }
        return {"text": HIDE_LABEL, "callback_data": encode_callback("kb-hide")}

    async def _hide_keyboard(self, chat_id: int) -> None:
        self._remember_keyboard(chat_id, hidden=True)
        await self.telegram.send_message(
            chat_id,
            "Кнопки убраны. Вернуть — командой /menu или кнопкой ниже.",
            reply_markup=hide_keyboard(),
        )
        await self.telegram.send_message(
            chat_id, "Меню:", reply_markup=main_menu() + [[self._keyboard_button(chat_id)]],
        )

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
        self.store.remember_user(user_id, chat_id, now_ms())
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
        observe_match = re.fullmatch(r"observe-(\d+):(.+)", action)
        if observe_match:
            await self._start_watch(chat_id, observe_match.group(2), int(observe_match.group(1)))
            return
        ask_match = re.fullmatch(r"observe-ask:(.+)", action)
        if ask_match:
            symbol = ask_match.group(1)
            self._pending_minutes[user_id] = symbol
            await self.telegram.send_message(
                chat_id,
                f"Сколько минут наблюдать за {short_symbol(symbol)}? "
                f"Число от 1 до {watch.MAX_DURATION_MIN}.",
                force_reply=True,
            )
            return
        watch_match = re.fullmatch(r"watch-(now|stop):(.+)", action)
        if watch_match:
            what, watch_id = watch_match.groups()
            if what == "now":
                await self._watch_snapshot(chat_id, watch_id)
            else:
                await self._stop_watch(chat_id, watch_id)
            return
        if action == "watches":
            await self._show_watches(chat_id)
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
                chat_id, _ask_symbol_text(chat_id), force_reply=True
            )
            return
        if action == "kb-hide":
            await self._hide_keyboard(chat_id)
            return
        if action == "kb-show":
            await self._show_menu(chat_id, keyboard=True)
            return
        if action in MENU_ACTIONS.values() or action in {
            "menu", "manual-add", "manual-remove",
        }:
            await self._dispatch_action(chat_id, user_id, action)
            return
        explain_match = re.fullmatch(r"explain-(book|watch):(.+)", action)
        if explain_match:
            kind, symbol = explain_match.groups()
            shown = self._last_texts.get(chat_id)
            if shown is None or shown[0] != kind:
                await self.telegram.send_message(
                    chat_id, "Нечего объяснять: покажите стакан или итог наблюдения заново.",
                )
                return
            await self._run_scenario(
                chat_id, user_id, "explain", symbol=symbol, payload=shown[1],
            )
            return
        history_match = re.fullmatch(r"hist-(1|7):(.+)", action)
        if history_match:
            days, symbol = int(history_match.group(1)), history_match.group(2)
            await self._run_scenario(
                chat_id, user_id, "history", symbol=symbol,
                as_of_ms=now_ms() - days * 86_400_000,
            )
            return
        await self.telegram.send_message(
            chat_id, "Не понял кнопку.", reply_markup=with_back(),
        )

    async def _dispatch_action(self, chat_id: int, user_id: int, action: str) -> None:
        """Один разбор действия для кнопок меню, подписей под полем и команд."""
        if action == "menu":
            await self._show_menu(chat_id)
            return
        if action == "watchlist":
            text = await self.templates.watchlist()
            await self.telegram.send_message(
                chat_id, text, reply_markup=_watchlist_keyboard(),
            )
            return
        if action == "manual":
            text = await self.templates.manual()
            await self.telegram.send_message(
                chat_id, text, reply_markup=_manual_keyboard(),
            )
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
        if action in {"morning", "quiet", "weekly"}:
            await self._run_scenario(chat_id, user_id, action)
            return
        await self.telegram.send_message(
            chat_id, "Такого сценария нет.", reply_markup=with_back(),
        )

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
            chat_id, _ask_symbol_text(chat_id), reply_markup=with_back(rows),
        )

    async def _use_symbol(self, chat_id: int, user_id: int, action: str, symbol: str) -> None:
        if action == "book":
            text = await self.templates.book(symbol)
            # Запоминаем показанное: кнопка «Объяснить» отправляет Claude
            # ровно те числа, которые человек видит, и ни одного лишнего.
            self._last_texts[chat_id] = ("book", text)
            await self.telegram.send_message(chat_id, text, reply_markup=_book_keyboard(symbol))
            return
        if action in {"analyse", "before"}:
            await self._run_scenario(chat_id, user_id, action, symbol=symbol)
            return
        if action == "history":
            await self.telegram.send_message(
                chat_id,
                f"⏪ Каким был {short_symbol(symbol)} до движения?",
                reply_markup=_history_keyboard(symbol),
            )
            return
        if action == "observe":
            running = [
                row for row in await self._running_watches()
                if str(row["symbol"]) == symbol
            ]
            if running:
                # Ровно тот случай, что поймал владелец: два наблюдения по
                # одной монете запустились подряд, а снять их было нечем.
                watch_id = str(running[0]["watch_id"])
                left = max(0, int(running[0]["ends_at"]) - now_ms()) // 60_000
                await self.telegram.send_message(
                    chat_id,
                    f"👁 За {short_symbol(symbol)} уже наблюдаю, осталось {left} мин.",
                    reply_markup=_running_watch_keyboard(watch_id, symbol),
                )
                return
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

    async def _run_scenario(
        self,
        chat_id: int,
        user_id: int,
        name: str,
        *,
        symbol: str | None = None,
        payload: str | None = None,
        as_of_ms: int | None = None,
    ) -> None:
        """Провести сценарий Claude: лимит, работа, подпись стоимости.

        Порядок проверок важен. Лимит — до запроса: фактическую цену узнаём
        только после ответа, и пускать сценарий, зная, что денег нет, значит
        тратить их сверх решения владельца. Занятость — на пользователя: два
        разбора подряд стоят вдвое и приходят вперемешку.
        """
        scenario = claude.SCENARIOS.get(name)
        if scenario is None:
            await self.telegram.send_message(chat_id, "Такого сценария нет.")
            return
        if self.assistant is None:
            await self.telegram.send_message(
                chat_id,
                "🧠 Claude не настроен: нет ключа. Кнопки без 🧠 работают.",
            )
            return
        stamp = now_ms()
        spent = self.store.spent_since(day_start_ms(self.config.timezone, stamp))
        if spent + claude.SCENARIO_ESTIMATE_USD > self.config.daily_budget_usd:
            await self.telegram.send_message(
                chat_id,
                budget_refusal_text(spent, self.config.daily_budget_usd, self.config.timezone),
            )
            return
        if user_id in self._busy:
            await self.telegram.send_message(chat_id, "Уже разбираю, подождите.")
            return

        self._busy.add(user_id)
        title = f"{short_symbol(symbol)} " if symbol else ""
        progress = await self.telegram.send_message(
            chat_id, f"⏳ {scenario.title.capitalize()} {title}— считаю…".replace("  ", " ")
        )
        try:
            params = {
                "symbol": symbol or "",
                "short": short_symbol(symbol) if symbol else "",
                "stamp": utc_stamp(stamp),
                "payload": claude.scenario_payload(payload or ""),
                "as_of_ms": as_of_ms or 0,
                "as_of_stamp": utc_stamp(as_of_ms) if as_of_ms else "",
            }
            answer = await self.assistant.run(scenario, **params)
        except Exception as error:  # noqa: BLE001 - кнопка не должна ронять polling
            log.warning("сценарий %s не отработал: %s", name, error)
            await self.telegram.send_message(
                chat_id, f"⚠️ Claude не ответил: {error}. Деньги за это не списаны.",
            )
            return
        finally:
            self._busy.discard(user_id)

        self.store.add_expense(
            scenario=name, symbol=symbol, usage=answer.usage,
            cost_usd=answer.cost_usd, stamp_ms=stamp,
        )
        today = self.store.spent_since(day_start_ms(self.config.timezone, stamp))
        chart = f"{tradingview_url(symbol, market='futures')}" if symbol else ""
        footer = (
            (f"\n\n{chart}" if chart else "")
            + f"\n\n─ {scenario.title} {claude.cost_words(answer.cost_usd)} · "
            f"сегодня {claude.cost_words(today)} из ${self.config.daily_budget_usd:.0f}"
        )
        text = (answer.text or "Claude вернул пустой ответ.") + footer
        keyboard = _scenario_keyboard(name, symbol)
        message_id = (progress or {}).get("message_id")
        edit = getattr(self.telegram, "edit_message", None)
        if edit is not None and message_id:
            try:
                await edit(chat_id, int(message_id), text, reply_markup=keyboard)
                return
            except Exception as error:  # noqa: BLE001 - правка не обязана удаться
                log.debug("правка сообщения не удалась: %s", error)
        await self.telegram.send_message(chat_id, text, reply_markup=keyboard)

    async def _morning_due(self, stamp_ms: int) -> bool:
        """Пора ли слать утренний обзор — раз в сутки и без досылки задним числом."""
        if not self.config.morning_time:
            return False
        zone = ZoneInfo(self.config.timezone)
        local = dt.datetime.fromtimestamp(stamp_ms / 1000, zone)
        planned = dt.datetime.combine(
            local.date(), dt.time.fromisoformat(self.config.morning_time), zone
        )
        # Окно, а не «после времени»: иначе бот, поднятый вечером, шлёт
        # «утренний обзор» в десять вечера — поймано на первом выкате
        # 17.09.2026, в 22:04 по месту.
        if not planned <= local < planned + dt.timedelta(hours=MORNING_WINDOW_H):
            return False
        today = local.date().isoformat()
        if self.store.get_meta("last_morning_date") == today:
            return False
        # Отмечаем ДО отправки: неудачная попытка не должна превращаться в
        # цикл повторов, который съест дневной лимит за одно утро.
        self.store.set_meta("last_morning_date", today)
        return True

    async def _send_morning(self) -> None:
        """Обзор уходит в личный чат владельца — первому, кто написал /start."""
        for user_id in self.config.allowed_user_ids:
            chat_id = self.store.chat_for_user(user_id)
            if chat_id is None:
                continue
            await self._run_scenario(chat_id, user_id, "morning")
            return
        log.info("утренний обзор некому слать: владелец не писал боту /start")

    async def _start_watch(self, chat_id: int, symbol: str, duration_min: int) -> None:
        """Б8: запускает тот же тикер, но задача принадлежит процессу бота.

        Сессия заводится теми же функциями хранилища и тикером, что и у
        MCP-инструмента, а не разбором его текста регуляркой: текст пишется
        для модели и меняется свободно (приёмка 17.09.2026, Б4).
        """
        from . import server

        try:
            session = await server.begin_order_book_watch(
                symbol, duration_min=duration_min
            )
        except (ToolError, watch.WatchAdmissionError) as error:
            reason = getattr(error, "message", None) or str(error)
            await self.telegram.send_message(chat_id, f"⚠️ {reason}")
            return
        watch_id = str(session["watch_id"])
        self.store.remember_watch(watch_id, chat_id, symbol, now_ms())
        await self.telegram.send_message(
            chat_id,
            f"👁 Наблюдаю за {short_symbol(symbol)} {duration_min} мин. Итог пришлю сам.",
            reply_markup=_running_watch_keyboard(watch_id, symbol),
        )
        asyncio.create_task(self._send_watch_summary(watch_id, chat_id, symbol))

    async def _running_watches(self) -> list[dict[str, Any]]:
        """Идущие наблюдения, запущенные этим ботом."""
        mine = {
            str(row["watch_id"]): row for row in self.store.unfinished_watches()
        }
        if not mine:
            return []
        con = watch.read_only()
        try:
            rows = [
                row for row in watch.list_watches(con)
                if str(row["watch_id"]) in mine and str(row["status"]) == "active"
            ]
        finally:
            if con is not None:
                con.close()
        return rows

    async def _show_watches(self, chat_id: int) -> None:
        rows = await self._running_watches()
        if not rows:
            await self.telegram.send_message(
                chat_id,
                "Сейчас ничего не наблюдаю. Запустить — кнопка «👁 Наблюдение».",
                reply_markup=with_back(),
            )
            return
        lines = ["👁 Идут наблюдения:"]
        keys: list[list[dict[str, str]]] = []
        for row in rows:
            left = max(0, int(row["ends_at"]) - now_ms()) // 60_000
            lines.append(
                f"{short_symbol(str(row['symbol']))} · осталось {left} мин · "
                f"снимков {row['snapshot_count']}"
            )
            watch_id = str(row["watch_id"])
            keys.append([
                {
                    "text": f"📸 {short_symbol(str(row['symbol']))}",
                    "callback_data": encode_callback("watch-now", watch_id),
                },
                {
                    "text": "⏹ Снять",
                    "callback_data": encode_callback("watch-stop", watch_id),
                },
            ])
        await self.telegram.send_message(
            chat_id, "\n".join(lines), reply_markup=with_back(keys),
        )

    async def _watch_snapshot(self, chat_id: int, watch_id: str) -> None:
        """Показать, что уже видно, не дожидаясь конца наблюдения."""
        text = await self.templates.watch_summary(watch_id)
        symbol = self._watch_symbol(watch_id)
        await self.telegram.send_message(
            chat_id, text,
            reply_markup=(
                _running_watch_keyboard(watch_id, symbol) if symbol else with_back()
            ),
        )

    async def _stop_watch(self, chat_id: int, watch_id: str) -> None:
        """Снять наблюдение и сразу показать, что успело набраться."""
        from . import server

        stopped = server.end_order_book_watch(watch_id)
        text = await self.templates.watch_summary(watch_id)
        self.store.finish_watch(watch_id, now_ms())
        head = "⏹ Наблюдение снято." if stopped else "Это наблюдение уже не идёт."
        await self.telegram.send_message(
            chat_id, f"{head}\n\n{text}", reply_markup=with_back(),
        )

    def _watch_symbol(self, watch_id: str) -> str | None:
        for row in self.store.unfinished_watches():
            if str(row["watch_id"]) == watch_id:
                return str(row["symbol"])
        return None

    async def _send_watch_summary(self, watch_id: str, chat_id: int, symbol: str) -> None:
        from . import server

        task = server._watch_tasks.get(watch_id)
        try:
            if task is not None:
                await task
            text = await self.templates.watch_summary(watch_id)
            self._last_texts[chat_id] = ("watch", text)
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


ASK_SYMBOL_TEXT = "Напишите тикер"


def _ask_symbol_text(chat_id: int) -> str:
    """В группе просьба сопровождается подсказкой про ответ на сообщение.

    У ботов в группах по умолчанию включён режим приватности: обычный текст до
    них не доходит вовсе, и человек видит тишину вместо ответа (замечание
    владельца 17.09.2026). Отрицательный chat_id — это группа или канал.
    """
    base = f"{ASK_SYMBOL_TEXT}: можно ID или IDUSDT."
    if chat_id < 0:
        return (
            f"{base}\nВ группе ответьте на это сообщение (свайп влево) или "
            "напишите командой: /book ICP, /analyse ICP, /watch ICP."
        )
    return base


def _normalise_symbol(text: str) -> str:
    symbol = text.strip().upper()
    if not symbol:
        raise ValueError("тикер пуст")
    return symbol if symbol.endswith("USDT") else f"{symbol}USDT"


def _watchlist_keyboard() -> list[list[dict[str, str]]]:
    return with_back([
        [
            {"text": "Все монеты", "callback_data": encode_callback("watchlist")},
            {"text": "🌅 Утренний обзор 🧠", "callback_data": encode_callback("morning")},
        ]
    ])


def _manual_keyboard() -> list[list[dict[str, str]]]:
    return with_back([[
        {"text": "➕ Добавить", "callback_data": encode_callback("manual-add")},
        {"text": "➖ Убрать", "callback_data": encode_callback("manual-remove")},
    ]])


def _manual_timeframe_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return with_back([[
        {"text": "1d", "callback_data": encode_callback("manual-tf-1d", symbol)},
        {"text": "4h", "callback_data": encode_callback("manual-tf-4h", symbol)},
        {"text": "1h", "callback_data": encode_callback("manual-tf-1h", symbol)},
    ]])


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
        [{"text": BACK_LABEL, "callback_data": encode_callback("menu")}],
    ]


def _history_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return with_back([[
        {"text": "Сутки назад", "callback_data": encode_callback("hist-1", symbol)},
        {"text": "Неделю назад", "callback_data": encode_callback("hist-7", symbol)},
    ]])


def _scenario_keyboard(name: str, symbol: str | None) -> list[list[dict[str, str]]] | None:
    """Следующий шаг под ответом: разбор ведёт к проверке и стакану."""
    if symbol is None:
        return with_back()
    book = {"text": "📊 Стакан", "callback_data": encode_callback("symbol-book", symbol)}
    if name == "analyse":
        before = encode_callback("symbol-before", symbol)
        return with_back([[{"text": "✅ Перед сделкой 🧠", "callback_data": before}, book]])
    if name in {"before", "history"}:
        analyse = encode_callback("symbol-analyse", symbol)
        return with_back([[{"text": "🔍 Разбор 🧠", "callback_data": analyse}, book]])
    return with_back()


def _observe_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return with_back([
        [
            {"text": "15 мин", "callback_data": encode_callback("observe-15", symbol)},
            {"text": "30 мин", "callback_data": encode_callback("observe-30", symbol)},
            {"text": "60 мин", "callback_data": encode_callback("observe-60", symbol)},
        ],
        [
            {
                "text": "⌨️ Своё время",
                "callback_data": encode_callback("observe-ask", symbol),
            },
            {
                "text": "📊 Снимок сейчас",
                "callback_data": encode_callback("symbol-book", symbol),
            },
        ],
        [{"text": "👁 Мои наблюдения", "callback_data": encode_callback("watches")}],
    ])


def _running_watch_keyboard(watch_id: str, symbol: str) -> list[list[dict[str, str]]]:
    """Что можно сделать с идущим наблюдением: посмотреть сейчас или снять."""
    return with_back([[
        {"text": "📸 Что уже видно", "callback_data": encode_callback("watch-now", watch_id)},
        {"text": "⏹ Снять", "callback_data": encode_callback("watch-stop", watch_id)},
    ]])


def _watch_summary_keyboard(symbol: str) -> list[list[dict[str, str]]]:
    return with_back([[
        {"text": "🧠 Объяснить", "callback_data": encode_callback("explain-watch", symbol)},
        {"text": "👁 Ещё 30 мин", "callback_data": encode_callback("observe-30", symbol)},
    ]])


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
        "watch": "watch",
        "наблюдение": "watch",
        "итог": "watch",
    }
    chosen = aliases.get(scenario.lower())
    if chosen == "watchlist":
        return await templates.watchlist()
    if chosen == "manual":
        return await templates.manual()
    if chosen == "watch":
        if not symbol:
            raise ValueError(
                "для итога наблюдения нужен watch_id: render watch watch_1a2b3c4d"
            )
        return await templates.watch_summary(symbol)
    if chosen == "book":
        if not symbol:
            raise ValueError("для стакана нужен тикер: render book ARBUSDT")
        ticker = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
        return await templates.book(ticker)
    raise ValueError("сценарий: watchlist, book, manual или watch")


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
