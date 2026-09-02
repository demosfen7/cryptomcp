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
import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

from . import SQUEEZE_FORMULA_VERSION

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

-- Единственная таблица с обычным rowid: на неё ссылаются outcomes, а
-- составной ключ в такой ссылке был бы вчетверо толще целого id.
CREATE TABLE IF NOT EXISTS scan_log (
    id                INTEGER PRIMARY KEY,
    ts_ms             INTEGER NOT NULL,
    symbol            TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    tf                TEXT    NOT NULL,
    formula_version   TEXT    NOT NULL,
    squeeze_index     REAL,
    components        TEXT,
    excluded          TEXT,
    price             REAL,
    range_low         REAL,
    range_high        REAL,
    range_width_pct   REAL,
    narrow_bars       INTEGER,
    atr_pct           REAL,
    rsi               REAL,
    bbw_pct_rank      REAL,
    volume_ratio      REAL,
    taker_buy_mean    REAL,
    ema_state         TEXT,
    structure         TEXT,
    closed_through_ms INTEGER,
    -- Признаки накопления. Считаются на МЛАДШЕМ ряду (§4.19) и пишутся с
    -- первого дня: через два месяца вопрос будет «работает ли накопление», и
    -- ответить на него можно только по журналу.
    absorption_tf     TEXT,
    absorption_bars   INTEGER,
    taker_max         REAL,
    taker_above       INTEGER,
    taker_streak      INTEGER,
    volume_lead       INTEGER,
    lead_state        TEXT,
    oi_change_24h     REAL,
    oi_reading        TEXT,
    funding_annual    REAL
);

-- Одна строка на закрытую свечу И версию формулы. Сканер ходит раз в час, а
-- четырёхчасовая свеча закрывается раз в четыре: без ограничения одно и то же
-- наблюдение попадало бы в статистику четырежды и перевешивало бы остальные.
-- Версия в ключе нужна для перехода между формулами: иначе после её смены
-- текущая свеча осталась бы посчитанной по старой, и несколько часов ранги
-- считались бы по смеси двух версий. Ограничение объявлено в схеме, а не в
-- коде, чтобы его нельзя было обойти по невнимательности.
DROP INDEX IF EXISTS scan_log_candle;
CREATE UNIQUE INDEX IF NOT EXISTS scan_log_candle_version
    ON scan_log (symbol, tf, closed_through_ms, formula_version);
CREATE INDEX IF NOT EXISTS scan_log_symbol_tf_ts ON scan_log (symbol, tf, ts_ms);
CREATE INDEX IF NOT EXISTS scan_log_ts ON scan_log (ts_ms);

CREATE TABLE IF NOT EXISTS outcomes (
    scan_id     INTEGER NOT NULL,
    horizon     TEXT    NOT NULL,
    max_price   REAL,
    min_price   REAL,
    max_pct     REAL,
    min_pct     REAL,
    close_pct   REAL,
    candles     INTEGER,
    computed_at INTEGER,
    PRIMARY KEY (scan_id, horizon)
) WITHOUT ROWID;

-- Эпизод наблюдения: одна строка на «вошла — вышла», а не одна на пару.
-- Монета попадает в список не раз в жизни, и затирать прошлый эпизод новым
-- значило бы стирать ровно ту историю, ради которой список и ведётся.
CREATE TABLE IF NOT EXISTS watchlist (
    id                 INTEGER PRIMARY KEY,
    symbol             TEXT    NOT NULL,
    tf                 TEXT    NOT NULL,
    status             TEXT    NOT NULL,
    entered_at         INTEGER NOT NULL,
    entered_by         TEXT    NOT NULL,
    squeeze_index      REAL,
    accumulation_score REAL,
    rank_at_entry      INTEGER,
    price_at_entry     REAL,
    range_low          REAL,
    range_high         REAL,
    last_rank          INTEGER,
    last_index         REAL,
    exited_at          INTEGER,
    exit_reason        TEXT
);

-- Открытый эпизод на пару может быть только один; закрытых — сколько угодно.
CREATE UNIQUE INDEX IF NOT EXISTS watchlist_open
    ON watchlist (symbol, tf) WHERE exited_at IS NULL;

CREATE TABLE IF NOT EXISTS collector_runs (
    ts_ms   INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL,
    symbols INTEGER,
    rows    INTEGER,
    seconds REAL,
    error   TEXT
);
"""

#: Колонки признаков накопления в scan_log. Список закрытый: опечатка в имени
#: молча писала бы в никуда, а обнаружилось бы это через месяц пустой статистики.
ACCUMULATION_COLUMNS = frozenset({
    "absorption_tf", "absorption_bars", "taker_max", "taker_above",
    "taker_streak", "volume_lead", "lead_state", "oi_change_24h",
    "oi_reading", "funding_annual",
})

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
        _migrate(con)
    return con


#: Колонки, добавленные после первого выпуска схемы: таблица, колонка, тип.
#:
#: `CREATE TABLE IF NOT EXISTS` существующую таблицу не трогает, а база на
#: сервере переживает выкладку — значит новую колонку надо дописывать явно,
#: иначе после деплоя код ждёт поля, которого в файле нет. Пересоздавать
#: таблицу нельзя: в ней лежат эпизоды, ради истории которых она и заведена.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("watchlist", "rank_at_entry", "INTEGER"),
    ("scan_log", "absorption_tf", "TEXT"),
    ("scan_log", "absorption_bars", "INTEGER"),
    ("scan_log", "taker_max", "REAL"),
    ("scan_log", "taker_above", "INTEGER"),
    ("scan_log", "taker_streak", "INTEGER"),
    ("scan_log", "volume_lead", "INTEGER"),
    ("scan_log", "lead_state", "TEXT"),
    ("scan_log", "oi_change_24h", "REAL"),
    ("scan_log", "oi_reading", "TEXT"),
    ("scan_log", "funding_annual", "REAL"),
)


def _migrate(con: sqlite3.Connection) -> None:
    """Дописать недостающие колонки. Идемпотентно: повторный запуск — no-op."""
    for table, column, decl in MIGRATIONS:
        have = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    con.commit()


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


def load_candles(
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    limit: int = 1500,
    before_ms: int | None = None,
) -> list[tuple[Any, ...]]:
    """Последние ``limit`` свечей из архива, по возрастанию времени.

    ``before_ms`` — правая граница для ретроспективы. Отбор делает база, а не
    вызывающий: взять последние ``limit`` свечей и отфильтровать их по дате
    значило бы получить огрызок окна вместо окна на нужный момент.
    """
    sql = f"SELECT ts, {', '.join(OHLCV_COLUMNS)} FROM ohlcv WHERE symbol = ? AND tf = ?"
    params: list[Any] = [symbol, tf]
    if before_ms is not None:
        sql += " AND ts <= ?"
        params.append(before_ms)
    rows = con.execute(sql + " ORDER BY ts DESC LIMIT ?", (*params, limit)).fetchall()
    return [tuple(row) for row in reversed(rows)]


def archived_symbols(con: sqlite3.Connection, tf: str) -> list[tuple[str, str]]:
    """Пары «символ, источник», по которым есть свечи этого таймфрейма."""
    rows = con.execute(
        "SELECT symbol, MIN(source) AS source FROM ohlcv WHERE tf = ? "
        "GROUP BY symbol ORDER BY symbol",
        (tf,),
    ).fetchall()
    return [(row["symbol"], row["source"]) for row in rows]


def finest_tf(con: sqlite3.Connection, symbol: str) -> str | None:
    """Самый мелкий архивный таймфрейм символа.

    Исход считается по High/Low внутри горизонта, и чем мельче свечи, тем
    точнее найдены экстремумы: на дневках цель, задетая внутри дня, попадёт
    в диапазон свечи, а на часовых видно и когда именно.
    """
    order = {"1h": 0, "4h": 1, "1d": 2, "1w": 3}
    rows = con.execute(
        "SELECT DISTINCT tf FROM ohlcv WHERE symbol = ?", (symbol,)
    ).fetchall()
    available = sorted((row["tf"] for row in rows), key=lambda tf: order.get(tf, 99))
    return available[0] if available else None


def price_extremes(
    con: sqlite3.Connection, symbol: str, tf: str, start_ms: int, end_ms: int
) -> dict[str, Any] | None:
    """Максимум, минимум и последнее закрытие в окне.

    По High/Low, а не по закрытию: цель, которую цена достала и с которой
    откатилась, по закрытию не засчиталась бы вовсе.
    """
    row = con.execute(
        "SELECT MAX(h) AS high, MIN(l) AS low, COUNT(*) AS candles, MAX(ts) AS last_ts "
        "FROM ohlcv WHERE symbol = ? AND tf = ? AND ts > ? AND ts <= ?",
        (symbol, tf, start_ms, end_ms),
    ).fetchone()
    if not row or not row["candles"]:
        return None
    close = con.execute(
        "SELECT c FROM ohlcv WHERE symbol = ? AND tf = ? AND ts = ?",
        (symbol, tf, row["last_ts"]),
    ).fetchone()
    return {
        "high": row["high"],
        "low": row["low"],
        "candles": row["candles"],
        "close": close["c"] if close else None,
    }


def record_scan(
    con: sqlite3.Connection,
    symbol: str,
    source: str,
    view: Any,
    *,
    formula_version: str,
    ts_ms: int | None = None,
    accumulation: dict[str, Any] | None = None,
) -> int:
    """Строка журнала сканирования.

    Пишутся не только индекс, но и все его компоненты по отдельности. Через
    два месяца вопрос будет не «работает ли индекс», а «какая из пяти групп
    в нём работает» — и ответить на него можно только по разложению.
    """
    payload = (
        ts_ms if ts_ms is not None else int(dt.datetime.now(dt.UTC).timestamp() * 1000),
        symbol,
        source,
        view.interval,
        formula_version,
        view.squeeze_index,
        json.dumps(view.components, ensure_ascii=False),
        json.dumps(view.excluded, ensure_ascii=False),
        view.price,
        view.range_low,
        view.range_high,
        round(view.range_width * 100, 4),
        view.narrow_bars,
        round(view.atr_pct, 4),
        round(view.rsi_value, 2),
        view.bbw.pct_rank,
        round(view.volume.ratio, 4) if view.volume.ratio == view.volume.ratio else None,
        round(view.volume.taker_buy_mean, 4)
        if view.volume.taker_buy_mean == view.volume.taker_buy_mean
        else None,
        view.ema_state,
        view.structure,
        view.meta.get("closed_through_ms"),
    )
    columns = [
        "ts_ms", "symbol", "source", "tf", "formula_version", "squeeze_index",
        "components", "excluded", "price", "range_low", "range_high",
        "range_width_pct", "narrow_bars", "atr_pct", "rsi", "bbw_pct_rank",
        "volume_ratio", "taker_buy_mean", "ema_state", "structure",
        "closed_through_ms",
    ]
    values = list(payload)
    # Признаки накопления приходят готовыми словарём: они считаются по ДРУГОМУ
    # ряду, и собирать их из view было бы неверно — там свой таймфрейм.
    for column, value in (accumulation or {}).items():
        if column not in ACCUMULATION_COLUMNS:
            raise ValueError(f"Неизвестная колонка накопления: {column!r}")
        columns.append(column)
        values.append(value)

    cursor = con.execute(
        f"INSERT OR IGNORE INTO scan_log ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})",
        values,
    )
    # rowcount == 0 означает, что свеча уже записана предыдущим прогоном.
    return int(cursor.lastrowid or 0) if cursor.rowcount else 0


def pending_outcomes(
    con: sqlite3.Connection, horizon: str, horizon_ms: int, now_ms: int, limit: int = 500
) -> list[dict[str, Any]]:
    """Записи скана, у которых горизонт истёк, а исход не посчитан."""
    rows = con.execute(
        "SELECT s.id, s.symbol, s.tf, s.ts_ms, s.price FROM scan_log s "
        "LEFT JOIN outcomes o ON o.scan_id = s.id AND o.horizon = ? "
        "WHERE o.scan_id IS NULL AND s.ts_ms + ? <= ? AND s.price > 0 "
        "ORDER BY s.ts_ms LIMIT ?",
        (horizon, horizon_ms, now_ms, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def record_outcome(
    con: sqlite3.Connection, scan_id: int, horizon: str, values: dict[str, Any]
) -> None:
    con.execute(
        "INSERT INTO outcomes (scan_id, horizon, max_price, min_price, max_pct, "
        "min_pct, close_pct, candles, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (scan_id, horizon) DO UPDATE SET "
        "max_price = excluded.max_price, min_price = excluded.min_price, "
        "max_pct = excluded.max_pct, min_pct = excluded.min_pct, "
        "close_pct = excluded.close_pct, candles = excluded.candles, "
        "computed_at = excluded.computed_at",
        (
            scan_id, horizon, values["max_price"], values["min_price"],
            values["max_pct"], values["min_pct"], values["close_pct"],
            values["candles"], int(dt.datetime.now(dt.UTC).timestamp() * 1000),
        ),
    )


def latest_scan(
    con: sqlite3.Connection, tf: str, formula_version: str | None = None
) -> list[dict[str, Any]]:
    """Последняя запись скана по каждому символу этого таймфрейма.

    Версия формулы обязательна к учёту: индексы, посчитанные по разным
    формулам, между собой не сравниваются, а ранг — это именно сравнение.
    """
    version = formula_version or SQUEEZE_FORMULA_VERSION
    rows = con.execute(
        "SELECT s.* FROM scan_log s JOIN ("
        "  SELECT symbol, MAX(ts_ms) AS ts FROM scan_log "
        "  WHERE tf = ? AND formula_version = ? GROUP BY symbol"
        ") last ON last.symbol = s.symbol AND last.ts = s.ts_ms "
        "WHERE s.tf = ? AND s.formula_version = ? AND s.squeeze_index IS NOT NULL "
        "ORDER BY s.squeeze_index DESC",
        (tf, version, tf, version),
    ).fetchall()
    return [dict(row) for row in rows]


def scan_history(
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    limit: int = 20,
    formula_version: str | None = None,
) -> list[dict[str, Any]]:
    """История записей скана по паре, свежие сверху.

    Версия формулы по умолчанию текущая: показать в одной таблице индексы,
    посчитанные по разным формулам, значило бы предложить сравнить
    несравнимое — ровно то, ради чего версия и попала в ключ журнала.
    """
    version = formula_version or SQUEEZE_FORMULA_VERSION
    rows = con.execute(
        "SELECT * FROM scan_log WHERE symbol = ? AND tf = ? AND formula_version = ? "
        "ORDER BY ts_ms DESC LIMIT ?",
        (symbol.upper(), tf, version, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def open_episodes(con: sqlite3.Connection, tf: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM watchlist WHERE exited_at IS NULL"
    params: tuple[Any, ...] = ()
    if tf is not None:
        sql += " AND tf = ?"
        params = (tf,)
    return [dict(row) for row in con.execute(sql + " ORDER BY id", params)]


#: Что принимает фильтр статуса: два открытых состояния, два закрытых и два
#: собирательных значения.
EPISODE_STATUSES = ("candidate", "active", "broken_out", "expired", "closed", "all")


def episodes(
    con: sqlite3.Connection,
    *,
    status: str | None = None,
    tf: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Эпизоды наблюдения для выдачи наружу.

    По умолчанию отдаются только открытые: это рабочий список, ради которого
    инструмент и нужен. Закрытые запрашиваются явно — тогда видно, чем
    кончилось, и `exit_reason` перестаёт быть мёртвой колонкой.

    Порядок — по текущему рангу внутри таймфрейма: он и есть порядок внимания.
    Эпизод без ранга (монета выпала из универсума) уезжает вниз, а не наверх,
    как случилось бы при NULL в обычной сортировке SQLite.
    """
    where: list[str] = []
    params: list[Any] = []
    if status is None:
        where.append("exited_at IS NULL")
    elif status == "closed":
        where.append("exited_at IS NOT NULL")
    elif status != "all":
        where.append("status = ?")
        params.append(status)
    if tf is not None:
        where.append("tf = ?")
        params.append(tf)

    sql = "SELECT * FROM watchlist"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += (
        " ORDER BY exited_at IS NOT NULL, tf, "
        "CASE WHEN last_rank IS NULL THEN 1 ELSE 0 END, last_rank, id LIMIT ?"
    )
    params.append(limit)
    return [dict(row) for row in con.execute(sql, params)]


def open_episode(
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    *,
    entered_at: int,
    entered_by: str,
    scan: dict[str, Any] | None = None,
    rank: int | None = None,
) -> int:
    """Завести эпизод наблюдения.

    Границы диапазона запоминаются на входе и потом не пересчитываются: пробой
    определяется относительно того, что было в момент попадания в список, а не
    относительно уехавшего вместе с ценой диапазона.
    """
    scan = scan or {}
    cursor = con.execute(
        "INSERT OR IGNORE INTO watchlist (symbol, tf, status, entered_at, entered_by, "
        "squeeze_index, rank_at_entry, price_at_entry, range_low, range_high, "
        "last_rank, last_index) "
        "VALUES (?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            symbol, tf, entered_at, entered_by,
            scan.get("squeeze_index"), rank, scan.get("price"),
            scan.get("range_low"), scan.get("range_high"),
            rank, scan.get("squeeze_index"),
        ),
    )
    return int(cursor.lastrowid or 0) if cursor.rowcount else 0


def close_episode(
    con: sqlite3.Connection, episode_id: int, *, status: str, reason: str, ts_ms: int
) -> None:
    con.execute(
        "UPDATE watchlist SET status = ?, exit_reason = ?, exited_at = ? "
        "WHERE id = ? AND exited_at IS NULL",
        (status, reason, ts_ms, episode_id),
    )


def touch_episode(
    con: sqlite3.Connection,
    episode_id: int,
    *,
    rank: int | None,
    index: float | None,
    promote: bool = False,
) -> None:
    """Обновить положение эпизода; при promote — перевести кандидата в active."""
    con.execute(
        "UPDATE watchlist SET last_rank = ?, last_index = ?, status = "
        "CASE WHEN ? = 1 AND status = 'candidate' THEN 'active' ELSE status END "
        "WHERE id = ?",
        (rank, index, 1 if promote else 0, episode_id),
    )


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


def derivatives_window(
    con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int
) -> list[tuple[int, float, float]]:
    """Точки открытого интереса за окно, старые→новые: (время, контракты, стоимость).

    Ради ретроспективы архив и заводился: за пределами тридцати суток биржа
    этих чисел не отдаёт вовсе, а здесь они лежат с шагом 5m.
    """
    rows = con.execute(
        "SELECT ts, open_interest, open_interest_value FROM derivatives "
        "WHERE symbol = ? AND ts >= ? AND ts <= ? "
        "AND open_interest IS NOT NULL ORDER BY ts",
        (symbol.upper(), start_ms, end_ms),
    ).fetchall()
    return [
        (int(row["ts"]), float(row["open_interest"]), float(row["open_interest_value"]))
        for row in rows
        if row["open_interest_value"] is not None
    ]


def funding_window(
    con: sqlite3.Connection, symbol: str, end_ms: int, limit: int = 1000
) -> list[tuple[int, float]]:
    """Последние начисления фандинга до момента, старые→новые."""
    rows = con.execute(
        "SELECT ts, rate FROM funding WHERE symbol = ? AND ts <= ? "
        "ORDER BY ts DESC LIMIT ?",
        (symbol.upper(), end_ms, limit),
    ).fetchall()
    return [(int(row["ts"]), float(row["rate"])) for row in reversed(rows)]


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
