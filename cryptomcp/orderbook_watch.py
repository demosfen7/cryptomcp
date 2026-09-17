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
    weight_per_min          REAL NOT NULL
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
    con.commit()
    return con


def read_only(path: str = DEFAULT_PATH) -> sqlite3.Connection | None:
    """Открыть существующий файл без создания: отсутствие сессий штатно."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def weight_per_minute(interval_sec: int, depth_weight: int) -> float:
    return 60.0 / interval_sec * depth_weight


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
    started_at: int | None = None,
) -> dict[str, Any]:
    """Атомарно допустить сессию либо отказать ДО первого снимка (B.4)."""
    if interval_sec < 1:
        raise WatchAdmissionError("interval_sec должен быть не меньше 1")
    if not 1 <= duration_min <= MAX_DURATION_MIN:
        raise WatchAdmissionError(
            f"duration_min должен быть в диапазоне 1..{MAX_DURATION_MIN}"
        )
    started_at = now_ms() if started_at is None else started_at
    requested = weight_per_minute(interval_sec, depth_weight)
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
        }
        con.execute(
            "INSERT INTO order_book_watches "
            "(watch_id, symbol, market, interval_sec, duration_min, depth_pcts, "
            "started_at, ends_at, status, retention_after_end_min, weight_per_min) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                watch_id, row["symbol"], market, interval_sec, duration_min,
                json.dumps(row["depth_pcts"]), started_at, ends_at, "active",
                RETENTION_AFTER_END_MIN, requested,
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
    return [_watch_row(row) for row in con.execute(
        "SELECT * FROM order_book_watches ORDER BY started_at DESC"
    )]


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
