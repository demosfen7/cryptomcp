"""Временное хранилище наблюдений за стаканом (SPEC-order-book-v2 B.4--B.6).

Сервер и часовик пишут сюда, но НЕ в ``market.sqlite``. Ошибка в сессии не
должна даже теоретически лишить сборщик невосстановимого часа деривативов;
отдельный файл повторяет границу ручного watchlist из PLAN §4.28 (решение Р1).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from . import storage
from .orderbook import BookLevel, OrderBook

#: Сессия запрашивает limit=100: меньшая глубина перестаёт быть полезной для
#: diff, а на споте это не экономит вес; на futures 100 дороже 50, но эта
#: глубина остаётся принятым компромиссом (README задания, решение Р2).
WATCH_LIMIT = 100
DEFAULT_INTERVAL_SEC = 5
DEFAULT_DURATION_MIN = 60
MAX_DURATION_MIN = 240
RETENTION_AFTER_END_MIN = 120
MAX_ACTIVE_WATCHES = 10
SESSION_WEIGHT_SHARE = 0.5

#: B.7: стартовые, не откалиброваны; значения предписаны примером B.9.
LONG_LIVED_FRACTION = 0.5
VANISHED_BEFORE_TOUCH_SNAPSHOTS = 3
NEAR_PRICE_PCT = 0.3
FILL_SHARE = 0.8
SUMMARY_TOP_LEVELS = 10
#: asyncio может начать следующий плановый тик на несколько миллисекунд раньше.
#: Такой джиттер не должен откладывать polling trades ещё на весь интервал.
TRADE_POLL_JITTER_MS = 50

DEFAULT_PATH = os.environ.get("CRYPTOMCP_ORDER_BOOK_WATCH_DB") or os.path.join(
    os.path.dirname(storage.DEFAULT_PATH) or ".", "order_book_watch.sqlite"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS order_book_watches (
    watch_id                TEXT PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    market                  TEXT NOT NULL,
    interval_sec            INTEGER NOT NULL,
    duration_min            INTEGER NOT NULL,
    depth_pcts              TEXT NOT NULL,
    started_at              INTEGER NOT NULL,
    ends_at                 INTEGER NOT NULL,
    status                  TEXT NOT NULL CHECK (status IN ('active', 'stopped', 'expired')),
    stopped_at              INTEGER,
    retention_after_end_min INTEGER NOT NULL,
    snapshot_count          INTEGER NOT NULL DEFAULT 0,
    error_count             INTEGER NOT NULL DEFAULT 0,
    last_snapshot_at        INTEGER,
    weight_per_min          REAL NOT NULL,
    turnover_24h            REAL,
    turnover_taken_at       INTEGER,
    trade_last_id           INTEGER,
    last_trade_fetch_at     INTEGER,
    trade_request_count     INTEGER NOT NULL DEFAULT 0,
    trade_extra_page_count  INTEGER NOT NULL DEFAULT 0,
    trade_gap_count         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS order_book_watches_status_end
    ON order_book_watches (status, ends_at);

CREATE TABLE IF NOT EXISTS watch_snapshots (
    watch_id     TEXT NOT NULL REFERENCES order_book_watches(watch_id) ON DELETE CASCADE,
    ts           INTEGER NOT NULL,
    mid_price    REAL NOT NULL,
    best_bid     REAL NOT NULL,
    best_ask     REAL NOT NULL,
    spread       REAL NOT NULL,
    spread_pct   REAL NOT NULL,
    turnover_24h REAL NOT NULL,
    bids_json    TEXT NOT NULL,
    asks_json    TEXT NOT NULL,
    ranges_json  TEXT NOT NULL,
    PRIMARY KEY (watch_id, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS watch_diffs (
    watch_id   TEXT NOT NULL REFERENCES order_book_watches(watch_id) ON DELETE CASCADE,
    ts         INTEGER NOT NULL,
    side       TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
    price      REAL NOT NULL,
    qty_before REAL NOT NULL,
    qty_after  REAL NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('appeared', 'disappeared', 'grew', 'shrank')),
    PRIMARY KEY (watch_id, ts, side, price, event_type)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS watch_diffs_window ON watch_diffs (watch_id, ts);

CREATE TABLE IF NOT EXISTS watch_trades (
    watch_id       TEXT NOT NULL REFERENCES order_book_watches(watch_id) ON DELETE CASCADE,
    agg_id         INTEGER NOT NULL,
    ts             INTEGER NOT NULL,
    price          REAL NOT NULL,
    qty            REAL NOT NULL,
    buyer_is_maker INTEGER NOT NULL CHECK (buyer_is_maker IN (0, 1)),
    PRIMARY KEY (watch_id, agg_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS watch_trades_window ON watch_trades (watch_id, ts);

CREATE TABLE IF NOT EXISTS watch_trade_gaps (
    watch_id      TEXT NOT NULL REFERENCES order_book_watches(watch_id) ON DELETE CASCADE,
    before_agg_id INTEGER NOT NULL,
    after_agg_id  INTEGER NOT NULL,
    started_at    INTEGER NOT NULL,
    ended_at      INTEGER NOT NULL,
    PRIMARY KEY (watch_id, before_agg_id, after_agg_id)
) WITHOUT ROWID;
"""


class WatchAdmissionError(ValueError):
    """Новый наблюдатель не помещается в выделенную половину веса IP."""


def now_ms() -> int:
    return int(time.time() * 1000)


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Открыть отдельную базу на запись, создав схему при первом обращении."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0)
    con.row_factory = sqlite3.Row
    # Здесь два процесса-писателя (сервер и collector), в отличие от
    # manual.py. Без busy_timeout редкая одновременная уборка роняла бы тик.
    con.execute("PRAGMA busy_timeout = 5000")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.executescript(SCHEMA)
    _migrate_session_columns(con)
    con.commit()
    return con


def _migrate_session_columns(con: sqlite3.Connection) -> None:
    """Добавить поля С1--С3 в уже созданный временный файл без потери сессий."""
    existing = {
        str(row["name"])
        for row in con.execute("PRAGMA table_info(order_book_watches)")
    }
    additions = {
        "turnover_24h": "REAL",
        "turnover_taken_at": "INTEGER",
        "trade_last_id": "INTEGER",
        "last_trade_fetch_at": "INTEGER",
        "trade_request_count": "INTEGER NOT NULL DEFAULT 0",
        "trade_extra_page_count": "INTEGER NOT NULL DEFAULT 0",
        "trade_gap_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for name, definition in additions.items():
        if name not in existing:
            con.execute(f"ALTER TABLE order_book_watches ADD COLUMN {name} {definition}")


def read_only(path: str = DEFAULT_PATH) -> sqlite3.Connection | None:
    """Открыть существующий файл без создания: отсутствие сессий штатно."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def weight_per_minute(interval_sec: int, depth_weight: int, trade_weight: int = 0) -> float:
    """Вес стакана и aggTrades из формулы С2, на IP-минуту."""
    return (
        60.0 / interval_sec * depth_weight
        + 60.0 / max(5, interval_sec) * trade_weight
    )


def start(
    con: sqlite3.Connection,
    *,
    symbol: str,
    market: str,
    interval_sec: int,
    duration_min: int,
    depth_pcts: tuple[float, ...],
    depth_weight: int,
    market_weight_limit: int,
    trade_weight: int = 0,
    started_at: int | None = None,
    turnover_24h: float | None = None,
    turnover_taken_at: int | None = None,
) -> dict[str, Any]:
    """Атомарно допустить сессию либо отказать ДО первого снимка (B.4)."""
    if interval_sec < 1:
        raise WatchAdmissionError("interval_sec должен быть не меньше 1")
    if not 1 <= duration_min <= MAX_DURATION_MIN:
        raise WatchAdmissionError(
            f"duration_min должен быть в диапазоне 1..{MAX_DURATION_MIN}"
        )
    started_at = now_ms() if started_at is None else started_at
    requested = weight_per_minute(interval_sec, depth_weight, trade_weight)
    ceiling = market_weight_limit * SESSION_WEIGHT_SHARE

    # Проверка и INSERT одной транзакцией: два одновременных start не должны
    # оба увидеть свободный бюджет, а потом вместе выйти за половину лимита.
    con.execute("BEGIN IMMEDIATE")
    try:
        active = [dict(row) for row in con.execute(
            "SELECT market, weight_per_min FROM order_book_watches "
            "WHERE status = 'active' AND ends_at > ?",
            (started_at,),
        )]
        active_count = len(active)
        used = sum(row["weight_per_min"] for row in active if row["market"] == market)
        if active_count >= MAX_ACTIVE_WATCHES:
            raise WatchAdmissionError(
                f"достигнут потолок {MAX_ACTIVE_WATCHES} одновременных сессий; "
                "новый снимок не запущен"
            )
        if used + requested > ceiling + 1e-9:
            raise WatchAdmissionError(
                f"сессия просит {requested:g} ед/мин, уже занято {used:g} ед/мин, "
                f"потолок рынка {market} — {ceiling:g} ед/мин; снимок не запущен"
            )
        watch_id = f"watch_{uuid.uuid4().hex[:8]}"
        ends_at = started_at + duration_min * 60_000
        row = {
            "watch_id": watch_id,
            "symbol": symbol.upper(),
            "market": market,
            "interval_sec": interval_sec,
            "duration_min": duration_min,
            "depth_pcts": list(depth_pcts),
            "started_at": started_at,
            "ends_at": ends_at,
            "status": "active",
            "stopped_at": None,
            "retention_after_end_min": RETENTION_AFTER_END_MIN,
            "snapshot_count": 0,
            "error_count": 0,
            "last_snapshot_at": None,
            "weight_per_min": requested,
            "turnover_24h": turnover_24h,
            "turnover_taken_at": turnover_taken_at,
            "trade_last_id": None,
            "last_trade_fetch_at": None,
            "trade_request_count": 0,
            "trade_extra_page_count": 0,
            "trade_gap_count": 0,
        }
        con.execute(
            "INSERT INTO order_book_watches "
            "(watch_id, symbol, market, interval_sec, duration_min, depth_pcts, "
            "started_at, ends_at, status, retention_after_end_min, weight_per_min, "
            "turnover_24h, turnover_taken_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                watch_id, row["symbol"], market, interval_sec, duration_min,
                json.dumps(row["depth_pcts"]), started_at, ends_at, "active",
                RETENTION_AFTER_END_MIN, requested, turnover_24h, turnover_taken_at,
            ),
        )
        con.commit()
        return row
    except Exception:
        con.rollback()
        raise


def get_watch(con: sqlite3.Connection | None, watch_id: str) -> dict[str, Any] | None:
    if con is None:
        return None
    row = con.execute(
        "SELECT * FROM order_book_watches WHERE watch_id = ?", (watch_id,)
    ).fetchone()
    return _watch_row(row) if row else None


def list_watches(con: sqlite3.Connection | None) -> list[dict[str, Any]]:
    if con is None:
        return []
    watches = [_watch_row(row) for row in con.execute(
        "SELECT * FROM order_book_watches ORDER BY started_at DESC"
    )]
    for watch in watches:
        watch["storage_bytes"] = storage_bytes(con, watch["watch_id"])
    return watches


def stop(con: sqlite3.Connection, watch_id: str, *, stopped_at: int | None = None) -> bool:
    stopped_at = now_ms() if stopped_at is None else stopped_at
    cursor = con.execute(
        "UPDATE order_book_watches SET status = 'stopped', stopped_at = ? "
        "WHERE watch_id = ? AND status = 'active'",
        (stopped_at, watch_id),
    )
    con.commit()
    return bool(cursor.rowcount)


def delete(con: sqlite3.Connection, watch_id: str) -> bool:
    cursor = con.execute("DELETE FROM order_book_watches WHERE watch_id = ?", (watch_id,))
    con.commit()
    return bool(cursor.rowcount)


def record_snapshot(
    con: sqlite3.Connection, watch_id: str, book: OrderBook
) -> list[dict[str, Any]]:
    """Записать сырой снимок и diff к предыдущему снимку ЭТОЙ сессии (B.5)."""
    watch = con.execute(
        "SELECT status, ends_at FROM order_book_watches WHERE watch_id = ?", (watch_id,)
    ).fetchone()
    if watch is None:
        raise ValueError(f"сессия {watch_id} не найдена")
    if watch["status"] != "active" or book.timestamp_ms >= watch["ends_at"]:
        raise ValueError(f"сессия {watch_id} больше не принимает снимки")
    previous = con.execute(
        "SELECT bids_json, asks_json FROM watch_snapshots WHERE watch_id = ? "
        "ORDER BY ts DESC LIMIT 1",
        (watch_id,),
    ).fetchone()
    bids_json = _levels_json(book.bids)
    asks_json = _levels_json(book.asks)
    ranges_json = json.dumps([asdict(item) for item in book.ranges], separators=(",", ":"))
    con.execute(
        "INSERT INTO watch_snapshots "
        "(watch_id, ts, mid_price, best_bid, best_ask, spread, spread_pct, turnover_24h, "
        "bids_json, asks_json, ranges_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            watch_id, book.timestamp_ms, book.mid_price, book.best_bid, book.best_ask,
            book.spread, book.spread_pct, book.turnover_24h_usdt,
            bids_json, asks_json, ranges_json,
        ),
    )
    # Первый снимок — начальная точка, не шестьдесят ложных «появлений».
    # Событие имеет смысл только как изменение между двумя снимками B.5.
    diffs = _diff(
        json.loads(previous["bids_json"]),
        json.loads(previous["asks_json"]),
        json.loads(bids_json),
        json.loads(asks_json),
    ) if previous else []
    con.executemany(
        "INSERT INTO watch_diffs "
        "(watch_id, ts, side, price, qty_before, qty_after, event_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                watch_id, book.timestamp_ms, item["side"], item["price"],
                item["qty_before"], item["qty_after"], item["event_type"],
            )
            for item in diffs
        ],
    )
    con.execute(
        "UPDATE order_book_watches SET snapshot_count = snapshot_count + 1, "
        "last_snapshot_at = ? WHERE watch_id = ?",
        (book.timestamp_ms, watch_id),
    )
    con.commit()
    return diffs


def record_error(con: sqlite3.Connection, watch_id: str) -> None:
    con.execute(
        "UPDATE order_book_watches SET error_count = error_count + 1 WHERE watch_id = ?",
        (watch_id,),
    )
    con.commit()


def record_trades(
    con: sqlite3.Connection,
    watch_id: str,
    rows: Iterable[dict[str, Any]],
    *,
    request_count: int,
    extra_pages: int,
) -> dict[str, int]:
    """Сохранить сырые aggTrades и явные разрывы их сквозного номера (С1--С3)."""
    session = con.execute(
        "SELECT trade_last_id FROM order_book_watches WHERE watch_id = ?", (watch_id,)
    ).fetchone()
    if session is None:
        raise ValueError(f"сессия {watch_id} не найдена")
    last_id = session["trade_last_id"]
    last_ts: int | None = None
    if last_id is not None:
        previous = con.execute(
            "SELECT ts FROM watch_trades WHERE watch_id = ? AND agg_id = ?",
            (watch_id, last_id),
        ).fetchone()
        last_ts = int(previous["ts"]) if previous else None

    trades = sorted((_trade(row) for row in rows), key=lambda item: item["agg_id"])
    stored = gaps = 0
    for trade in trades:
        if last_id is not None and trade["agg_id"] > last_id + 1:
            con.execute(
                "INSERT OR IGNORE INTO watch_trade_gaps "
                "(watch_id, before_agg_id, after_agg_id, started_at, ended_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    watch_id, last_id, trade["agg_id"],
                    last_ts if last_ts is not None else trade["ts"], trade["ts"],
                ),
            )
            gaps += 1
        cursor = con.execute(
            "INSERT OR IGNORE INTO watch_trades "
            "(watch_id, agg_id, ts, price, qty, buyer_is_maker) VALUES (?, ?, ?, ?, ?, ?)",
            (
                watch_id, trade["agg_id"], trade["ts"], trade["price"], trade["qty"],
                int(trade["buyer_is_maker"]),
            ),
        )
        stored += int(cursor.rowcount)
        if last_id is None or trade["agg_id"] > last_id:
            last_id, last_ts = trade["agg_id"], trade["ts"]

    con.execute(
        "UPDATE order_book_watches SET trade_last_id = ?, "
        "trade_request_count = trade_request_count + ?, "
        "trade_extra_page_count = trade_extra_page_count + ?, "
        "trade_gap_count = trade_gap_count + ? WHERE watch_id = ?",
        (last_id, request_count, extra_pages, gaps, watch_id),
    )
    con.commit()
    return {"stored": stored, "gaps": gaps}


def mark_trade_fetch(con: sqlite3.Connection, watch_id: str, fetched_at: int) -> None:
    con.execute(
        "UPDATE order_book_watches SET last_trade_fetch_at = ? WHERE watch_id = ?",
        (fetched_at, watch_id),
    )
    con.commit()


def expire(con: sqlite3.Connection, watch_id: str, *, expired_at: int | None = None) -> bool:
    """Пометить закончившуюся сессию сразу, не оставляя active до часа уборки."""
    expired_at = now_ms() if expired_at is None else expired_at
    cursor = con.execute(
        "UPDATE order_book_watches SET status = 'expired', stopped_at = ? "
        "WHERE watch_id = ? AND status = 'active' AND ends_at <= ?",
        (expired_at, watch_id, expired_at),
    )
    con.commit()
    return bool(cursor.rowcount)


def trades(
    con: sqlite3.Connection | None, watch_id: str, *, from_ts: int, to_ts: int
) -> list[dict[str, Any]]:
    if con is None:
        return []
    rows = con.execute(
        "SELECT * FROM watch_trades WHERE watch_id = ? AND ts > ? AND ts <= ? ORDER BY agg_id",
        (watch_id, from_ts, to_ts),
    )
    return [dict(row) for row in rows]


def trade_gaps(
    con: sqlite3.Connection | None, watch_id: str, *, from_ts: int, to_ts: int
) -> list[dict[str, Any]]:
    if con is None:
        return []
    rows = con.execute(
        "SELECT * FROM watch_trade_gaps WHERE watch_id = ? "
        "AND ended_at > ? AND started_at <= ? ORDER BY ended_at",
        (watch_id, from_ts, to_ts),
    )
    return [dict(row) for row in rows]


def classify_liquidity(
    snapshots: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Разобрать судьбу видимых уровней по снимкам и aggTrades (С4).

    Это не оценка намерений участника. Результат отвечает только, удержала ли
    заявка поток, была ли исполнена, ушла у цены или ушла перед подходом.
    """
    outcomes: dict[str, list[dict[str, Any]]] = {
        "устоял": [],
        "исполнен": [],
        "снят у цены": [],
        "снят заранее": [],
        "похоже на айсберг": [],
        "n/a": [],
    }
    ignored = 0
    trades_by_interval = sorted(trade_rows, key=lambda item: int(item["ts"]))
    for index, (before, after) in enumerate(zip(snapshots, snapshots[1:], strict=False)):
        start, end = int(before["ts"]), int(after["ts"])
        interval_trades = [
            item for item in trades_by_interval if start < int(item["ts"]) <= end
        ]
        has_gap = any(
            int(gap["ended_at"]) > start and int(gap["started_at"]) <= end for gap in gaps
        )
        for side, old_levels, new_levels in (
            ("bid", before["bids"], after["bids"]),
            ("ask", before["asks"], after["asks"]),
        ):
            old = {float(item["price"]): item for item in old_levels}
            new = {float(item["price"]): item for item in new_levels}
            if not old or not new:
                continue
            lower, upper = max(min(old), min(new)), min(max(old), max(new))
            if lower > upper:
                continue
            for price, level in old.items():
                if not lower <= price <= upper:
                    continue
                item = _level_outcome(side, price, level, before, after, interval_trades)
                if has_gap:
                    outcomes["n/a"].append({
                        **item,
                        "reason": "пропуск aggTrades в интервале снимков",
                    })
                    continue
                if price in new:
                    if item["traded_qty"] > 0:
                        outcomes["устоял"].append(item)
                        if item["traded_qty"] > item["qty_before"]:
                            outcomes["похоже на айсберг"].append(item)
                    continue
                if item["traded_qty"] >= FILL_SHARE * item["qty_before"]:
                    outcomes["исполнен"].append(item)
                elif _was_touched(side, price, interval_trades):
                    outcomes["снят у цены"].append(item)
                elif _vanished_before_touch(price, before, after, snapshots[index + 2:]):
                    outcomes["снят заранее"].append(item)
                else:
                    ignored += 1
    return {"outcomes": outcomes, "ignored": ignored}


def long_lived_levels(
    watch: dict[str, Any], snapshots: list[dict[str, Any]], *, now: int
) -> list[dict[str, Any]]:
    """Уровни, реально присутствовавшие большую долю уже прошедшей сессии (п.5)."""
    if not snapshots:
        return []
    interval_ms = int(watch["interval_sec"]) * 1000
    elapsed = max(interval_ms, min(now, int(watch["ends_at"])) - int(watch["started_at"]))
    seen: dict[tuple[str, float], tuple[int, dict[str, Any]]] = {}
    for snapshot in snapshots:
        for side, levels in (("bid", snapshot["bids"]), ("ask", snapshot["asks"])):
            for level in levels:
                key = (side, float(level["price"]))
                count, _ = seen.get(key, (0, level))
                seen[key] = (count + 1, level)
    result = []
    for (side, price), (count, level) in seen.items():
        present_ms = count * interval_ms
        if present_ms / elapsed < LONG_LIVED_FRACTION:
            continue
        result.append({
            "side": side,
            "price": price,
            "notional_usdt": float(level["notional_usdt"]),
            "minutes": present_ms / 60_000,
            "snapshots": count,
        })
    return sorted(result, key=lambda item: (-item["notional_usdt"], item["side"], item["price"]))


def storage_bytes(con: sqlite3.Connection | None, watch_id: str) -> int:
    """Сумма реально сохранённых колонок этой сессии, без выдуманных КБ на тик."""
    if con is None:
        return 0
    snapshot_bytes = con.execute(
        "SELECT COALESCE(SUM(length(bids_json) + length(asks_json) + length(ranges_json)), 0) "
        "FROM watch_snapshots WHERE watch_id = ?",
        (watch_id,),
    ).fetchone()[0]
    diff_bytes = con.execute(
        "SELECT COALESCE(SUM(length(side) + length(CAST(price AS TEXT)) + "
        "length(CAST(qty_before AS TEXT)) + length(CAST(qty_after AS TEXT)) + "
        "length(event_type)), 0) FROM watch_diffs WHERE watch_id = ?",
        (watch_id,),
    ).fetchone()[0]
    trade_bytes = con.execute(
        "SELECT COALESCE(SUM(length(CAST(agg_id AS TEXT)) + length(CAST(ts AS TEXT)) + "
        "length(CAST(price AS TEXT)) + length(CAST(qty AS TEXT)) + "
        "length(CAST(buyer_is_maker AS TEXT))), 0) FROM watch_trades WHERE watch_id = ?",
        (watch_id,),
    ).fetchone()[0]
    gap_bytes = con.execute(
        "SELECT COALESCE(SUM(length(CAST(before_agg_id AS TEXT)) + "
        "length(CAST(after_agg_id AS TEXT)) + length(CAST(started_at AS TEXT)) + "
        "length(CAST(ended_at AS TEXT))), 0) FROM watch_trade_gaps WHERE watch_id = ?",
        (watch_id,),
    ).fetchone()[0]
    return int(snapshot_bytes) + int(diff_bytes) + int(trade_bytes) + int(gap_bytes)


def snapshots(
    con: sqlite3.Connection | None,
    watch_id: str,
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> list[dict[str, Any]]:
    if con is None:
        return []
    where, params = _window_where(watch_id, from_ts, to_ts)
    rows = con.execute(
        "SELECT * FROM watch_snapshots WHERE " + where + " ORDER BY ts", params
    )
    return [_snapshot_row(row) for row in rows]


def diffs(
    con: sqlite3.Connection | None,
    watch_id: str,
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> list[dict[str, Any]]:
    if con is None:
        return []
    where, params = _window_where(watch_id, from_ts, to_ts)
    rows = con.execute("SELECT * FROM watch_diffs WHERE " + where + " ORDER BY ts", params)
    return [dict(row) for row in rows]


def snapshot_count(
    con: sqlite3.Connection | None,
    watch_id: str,
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> int:
    if con is None:
        return 0
    where, params = _window_where(watch_id, from_ts, to_ts)
    row = con.execute("SELECT COUNT(*) FROM watch_snapshots WHERE " + where, params).fetchone()
    return int(row[0])


def diff_count(
    con: sqlite3.Connection | None,
    watch_id: str,
    *,
    from_ts: int | None = None,
    to_ts: int | None = None,
) -> int:
    if con is None:
        return 0
    where, params = _window_where(watch_id, from_ts, to_ts)
    return int(con.execute("SELECT COUNT(*) FROM watch_diffs WHERE " + where, params).fetchone()[0])


def cleanup(path: str = DEFAULT_PATH, *, current_ms: int | None = None) -> dict[str, int]:
    """Истечь зависшие сессии и удалить завершённые данные после retention (B.6)."""
    if not os.path.exists(path):
        return {"expired": 0, "deleted": 0}
    current_ms = now_ms() if current_ms is None else current_ms
    con = connect(path)
    try:
        expired = con.execute(
            "UPDATE order_book_watches SET status = 'expired' "
            "WHERE status = 'active' AND ends_at < ?",
            (current_ms,),
        ).rowcount
        ids = [row["watch_id"] for row in con.execute(
            "SELECT watch_id FROM order_book_watches "
            "WHERE status IN ('expired', 'stopped') "
            "AND ? > COALESCE(stopped_at, ends_at) + retention_after_end_min * 60000",
            (current_ms,),
        )]
        for watch_id in ids:
            con.execute("DELETE FROM order_book_watches WHERE watch_id = ?", (watch_id,))
        con.commit()
        return {"expired": int(expired), "deleted": len(ids)}
    finally:
        con.close()


def _watch_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["depth_pcts"] = tuple(json.loads(result["depth_pcts"]))
    return result


def _snapshot_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in ("bids_json", "asks_json", "ranges_json"):
        result[key.removesuffix("_json")] = json.loads(result.pop(key))
    return result


def _window_where(
    watch_id: str, from_ts: int | None, to_ts: int | None
) -> tuple[str, tuple[Any, ...]]:
    clauses = ["watch_id = ?"]
    params: list[Any] = [watch_id]
    if from_ts is not None:
        clauses.append("ts >= ?")
        params.append(from_ts)
    if to_ts is not None:
        clauses.append("ts <= ?")
        params.append(to_ts)
    return " AND ".join(clauses), tuple(params)


def _levels_json(levels: Iterable[BookLevel]) -> str:
    return json.dumps([asdict(level) for level in levels], separators=(",", ":"))


def _trade(row: dict[str, Any]) -> dict[str, Any]:
    """Нормализовать ответ aggTrades, не выкидывая цену, количество или сторону."""
    return {
        "agg_id": int(row["a"]),
        "ts": int(row["T"]),
        "price": float(row["p"]),
        "qty": float(row["q"]),
        "buyer_is_maker": bool(row["m"]),
    }


def _level_outcome(
    side: str,
    price: float,
    level: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    buyer_taker = side == "ask"
    traded_qty = sum(
        float(row["qty"])
        for row in rows
        if float(row["price"]) == price and bool(row["buyer_is_maker"]) != buyer_taker
    )
    return {
        "side": side,
        "price": price,
        "qty_before": float(level["qty"]),
        "notional_usdt": float(level["notional_usdt"]),
        "traded_qty": traded_qty,
        "from_ts": int(before["ts"]),
        "to_ts": int(after["ts"]),
    }


def _was_touched(side: str, price: float, rows: list[dict[str, Any]]) -> bool:
    if side == "ask":
        return any(float(row["price"]) >= price for row in rows)
    return any(float(row["price"]) <= price for row in rows)


def _vanished_before_touch(
    price: float,
    before: dict[str, Any],
    after: dict[str, Any],
    later: list[dict[str, Any]],
) -> bool:
    def is_far(snapshot: dict[str, Any]) -> bool:
        distance = abs(float(snapshot["mid_price"]) - price) / float(snapshot["mid_price"])
        return distance * 100 > NEAR_PRICE_PCT

    if not (is_far(before) and is_far(after)):
        return False
    return any(
        not is_far(snapshot)
        for snapshot in later[:VANISHED_BEFORE_TOUCH_SNAPSHOTS]
    )


def _diff(
    before_bids: list[dict[str, Any]],
    before_asks: list[dict[str, Any]],
    after_bids: list[dict[str, Any]],
    after_asks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for side, before, after in (
        ("bid", before_bids, after_bids),
        ("ask", before_asks, after_asks),
    ):
        old = {float(item["price"]): float(item["qty"]) for item in before}
        new = {float(item["price"]): float(item["qty"]) for item in after}
        # limit=100 задаёт подвижное ценовое окно. Уровень на дальнем краю,
        # который оказался только в одном ответе Binance, не появился и не
        # исчез — он просто перестал быть виден. Сравниваем лишь пересечение
        # двух окон каждой стороны (приёмка 17.09, блокер 2).
        if not old or not new:
            continue
        lower = max(min(old), min(new))
        upper = min(max(old), max(new))
        if lower > upper:
            continue
        for price in sorted(set(old) | set(new), reverse=side == "bid"):
            if not lower <= price <= upper:
                continue
            qty_before, qty_after = old.get(price, 0.0), new.get(price, 0.0)
            if price not in old:
                event_type = "appeared"
            elif price not in new:
                event_type = "disappeared"
            elif qty_after > qty_before:
                event_type = "grew"
            elif qty_after < qty_before:
                event_type = "shrank"
            else:
                continue
            result.append({
                "side": side,
                "price": price,
                "qty_before": qty_before,
                "qty_after": qty_after,
                "event_type": event_type,
            })
    return result
