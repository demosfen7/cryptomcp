"""Персистентное хранилище (SQLite).

Существует ради одного класса данных: раздел `/futures/data/` у Binance отдаёт
только последние 30 суток. Открытый интерес, соотношения long/short и доля
тейкер-покупок старше месяца не существуют нигде — проверено отказом -1130 на
запрос за 31 сутки по всем пяти эндпоинтам. Каждый час без записи — час,
потерянный навсегда. Свечи, в отличие от этого, биржа отдаёт за годы, и их
можно докачать когда угодно.

Три решения, которые легко нарушить:

1. **Пишет только загрузчик.** Читателей много (MCP-сервер), писатель один.
   Отсюда WAL: без него читатель ловит `database is locked` на каждой записи.
2. **Все таблицы WITHOUT ROWID.** Замерено на этой самой схеме: 128 байт на
   строку против 98 — четверть объёма на ровном месте. Доступ везде идёт
   диапазонами по первичному ключу, терять нечего.
3. **Запись идемпотентна.** Повторный прогон за тот же период не должен ни
   падать, ни задваивать: перекрытие окон — норма, а не ошибка.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

#: Где лежит база. В контейнере — том, переживающий пересборку образа.
DEFAULT_PATH = os.environ.get("CRYPTOMCP_DB", "data/market.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol          TEXT    NOT NULL,
    tf              TEXT    NOT NULL,
    ts              INTEGER NOT NULL,
    o               REAL    NOT NULL,
    h               REAL    NOT NULL,
    l               REAL    NOT NULL,
    c               REAL    NOT NULL,
    volume          REAL,
    quote_volume    REAL,
    trades          INTEGER,
    taker_buy_base  REAL,
    taker_buy_quote REAL,
    source          TEXT    NOT NULL,
    PRIMARY KEY (symbol, tf, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS derivatives (
    symbol              TEXT    NOT NULL,
    ts                  INTEGER NOT NULL,
    open_interest       REAL,
    open_interest_value REAL,
    ls_global           REAL,
    ls_top_accounts     REAL,
    ls_top_positions    REAL,
    taker_ratio         REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT    NOT NULL,
    ts     INTEGER NOT NULL,
    rate   REAL    NOT NULL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS universe_daily (
    date               TEXT NOT NULL,
    symbol             TEXT NOT NULL,
    quote_volume_24h   REAL,
    open_interest_usdt REAL,
    price              REAL,
    has_spot           INTEGER,
    PRIMARY KEY (date, symbol)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS symbols (
    symbol         TEXT NOT NULL,
    market         TEXT NOT NULL,
    first_kline_ms INTEGER,
    first_seen     TEXT,
    PRIMARY KEY (symbol, market)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS collector_runs (
    ts_ms   INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL,
    symbols INTEGER,
    rows    INTEGER,
    seconds REAL,
    error   TEXT
);
"""

#: Колонки derivatives, кроме ключа. Порядок фиксирован: по нему строится upsert.
DERIVATIVE_COLUMNS = (
    "open_interest",
    "open_interest_value",
    "ls_global",
    "ls_top_accounts",
    "ls_top_positions",
    "taker_ratio",
)


def connect(path: str = DEFAULT_PATH, *, read_only: bool = False) -> sqlite3.Connection:
    """Открыть базу, создав файл и схему при первом обращении."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    # NORMAL вместо FULL: при сбое ОС теряется последняя транзакция, а не база.
    # Для ряда, который дозапишется на следующем прогоне, размен верный.
    con.execute("PRAGMA synchronous=NORMAL")
    if not read_only:
        con.executescript(SCHEMA)
        con.commit()
    return con


#: Колонки свечи, кроме ключа и источника.
OHLCV_COLUMNS = (
    "o", "h", "l", "c", "volume", "quote_volume",
    "trades", "taker_buy_base", "taker_buy_quote",
)


def upsert_ohlcv(
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    source: str,
    rows: Sequence[Sequence[Any]],
) -> int:
    """Записать свечи, перезаписывая совпадающие по времени.

    Перезапись, а не пропуск: биржа доправляет последнюю свечу в первые
    мгновения после закрытия, и инкрементальная догрузка обязана начинаться
    с уже сохранённой границы, а не после неё.
    """
    if not rows:
        return 0
    assignments = ", ".join(f"{c} = excluded.{c}" for c in OHLCV_COLUMNS)
    sql = (
        f"INSERT INTO ohlcv (symbol, tf, ts, {', '.join(OHLCV_COLUMNS)}, source) "
        f"VALUES (?, ?, ?, {', '.join('?' * len(OHLCV_COLUMNS))}, ?) "
        f"ON CONFLICT (symbol, tf, ts) DO UPDATE SET {assignments}"
    )
    con.executemany(sql, [(symbol, tf, *row, source) for row in rows])
    return len(rows)


def last_ohlcv_ts(con: sqlite3.Connection, symbol: str, tf: str) -> int | None:
    row = con.execute(
        "SELECT MAX(ts) AS ts FROM ohlcv WHERE symbol = ? AND tf = ?", (symbol, tf)
    ).fetchone()
    return row["ts"] if row and row["ts"] is not None else None


def archive_source(con: sqlite3.Connection, symbol: str) -> str | None:
    """С какого рынка уже собрана история символа.

    Источник выбирается один раз и дальше не меняется: если монета получит
    спотовую пару позже, переключение задним числом склеило бы в одном ряду
    два разных рынка с разными ценами и объёмами. Реестром служат сами данные —
    отдельная таблица источников рассинхронизировалась бы с ними.
    """
    row = con.execute(
        "SELECT source FROM ohlcv WHERE symbol = ? LIMIT 1", (symbol,)
    ).fetchone()
    return row["source"] if row else None


def ohlcv_coverage(con: sqlite3.Connection) -> list[dict[str, Any]]:
    """Что накоплено по свечам — на символ и таймфрейм."""
    rows = con.execute(
        "SELECT symbol, tf, source, COUNT(*) AS candles, "
        "MIN(ts) AS first_ts, MAX(ts) AS last_ts "
        "FROM ohlcv GROUP BY symbol, tf ORDER BY symbol, tf"
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_derivatives(
    con: sqlite3.Connection, symbol: str, rows: dict[int, dict[str, float]]
) -> int:
    """Записать точки ряда деривативов, дополняя уже сохранённые.

    Пять эндпоинтов отдают свои ряды на общей пятиминутной сетке, но с разной
    свежестью: taker-ratio отстаёт от открытого интереса на пару точек. Поэтому
    колонки пишутся по отдельности, а COALESCE не даёт прогону затереть уже
    записанное значение своим NULL.
    """
    if not rows:
        return 0
    assignments = ", ".join(
        f"{column} = COALESCE(excluded.{column}, {column})"
        for column in DERIVATIVE_COLUMNS
    )
    placeholders = ", ".join("?" * len(DERIVATIVE_COLUMNS))
    sql = (
        f"INSERT INTO derivatives (symbol, ts, {', '.join(DERIVATIVE_COLUMNS)}) "
        f"VALUES (?, ?, {placeholders}) "
        f"ON CONFLICT (symbol, ts) DO UPDATE SET {assignments}"
    )
    payload = [
        (symbol, ts, *(values.get(column) for column in DERIVATIVE_COLUMNS))
        for ts, values in sorted(rows.items())
    ]
    con.executemany(sql, payload)
    return len(payload)


def upsert_funding(
    con: sqlite3.Connection, symbol: str, rows: Iterable[tuple[int, float]]
) -> int:
    payload = [(symbol, ts, rate) for ts, rate in rows]
    con.executemany(
        "INSERT INTO funding (symbol, ts, rate) VALUES (?, ?, ?) "
        "ON CONFLICT (symbol, ts) DO UPDATE SET rate = excluded.rate",
        payload,
    )
    return len(payload)


def upsert_universe(
    con: sqlite3.Connection, date: str, rows: Sequence[dict[str, Any]]
) -> int:
    """Суточный снимок универсума.

    Пишется с первого дня, даже пока сканера нет: отбор по медианному обороту
    за неделю иначе не из чего считать, а задним числом эти снимки не собрать.
    """
    payload = [
        (
            date,
            row["symbol"],
            row.get("quote_volume_24h"),
            row.get("open_interest_usdt"),
            row.get("price"),
            int(bool(row.get("has_spot"))),
        )
        for row in rows
    ]
    con.executemany(
        "INSERT INTO universe_daily "
        "(date, symbol, quote_volume_24h, open_interest_usdt, price, has_spot) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (date, symbol) DO UPDATE SET "
        "quote_volume_24h = excluded.quote_volume_24h, "
        "open_interest_usdt = excluded.open_interest_usdt, "
        "price = excluded.price, has_spot = excluded.has_spot",
        payload,
    )
    return len(payload)


def remember_symbol(
    con: sqlite3.Connection, symbol: str, market: str, first_kline_ms: int | None
) -> None:
    """Запомнить дату листинга один раз: возраст потом считается арифметикой."""
    con.execute(
        "INSERT INTO symbols (symbol, market, first_kline_ms, first_seen) "
        "VALUES (?, ?, ?, ?) ON CONFLICT (symbol, market) DO UPDATE SET "
        "first_kline_ms = COALESCE(symbols.first_kline_ms, excluded.first_kline_ms)",
        (
            symbol,
            market,
            first_kline_ms,
            dt.datetime.now(dt.UTC).isoformat(" ", "seconds"),
        ),
    )


def known_symbols(con: sqlite3.Connection, market: str) -> dict[str, int | None]:
    rows = con.execute(
        "SELECT symbol, first_kline_ms FROM symbols WHERE market = ?", (market,)
    ).fetchall()
    return {row["symbol"]: row["first_kline_ms"] for row in rows}


def last_ts(con: sqlite3.Connection, table: str, symbol: str) -> int | None:
    """Последняя записанная точка — отсюда продолжается инкрементальная догрузка."""
    if table not in {"derivatives", "funding"}:
        raise ValueError(f"Неизвестная таблица: {table!r}")
    row = con.execute(
        f"SELECT MAX(ts) AS ts FROM {table} WHERE symbol = ?", (symbol,)
    ).fetchone()
    return row["ts"] if row and row["ts"] is not None else None


def record_run(
    con: sqlite3.Connection,
    kind: str,
    *,
    symbols: int,
    rows: int,
    seconds: float,
    error: str | None = None,
) -> None:
    """Журнал прогонов.

    Данные невосстановимы, поэтому важно не только писать, но и видеть, что
    запись шла: молчащий сборщик снаружи неотличим от работающего.
    """
    con.execute(
        "INSERT OR REPLACE INTO collector_runs "
        "(ts_ms, kind, symbols, rows, seconds, error) VALUES (?, ?, ?, ?, ?, ?)",
        (
            int(dt.datetime.now(dt.UTC).timestamp() * 1000),
            kind,
            symbols,
            rows,
            round(seconds, 2),
            error,
        ),
    )


def coverage(con: sqlite3.Connection, symbol: str) -> dict[str, Any]:
    """Что накоплено по символу — для проверок и отчётов."""
    row = con.execute(
        "SELECT COUNT(*) AS points, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
        "FROM derivatives WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    return dict(row) if row else {}
