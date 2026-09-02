"""Фоновый сборщик невосстановимых данных (этап 1 плана хранилища).

Чем этот процесс отличается от MCP-сервера: сервер отвечает на вопросы и
ничего не копит, сборщик копит и ни на что не отвечает. Разделены они потому,
что у них разные режимы отказа. Сервер может молчать час — никто не пострадает.
Сборщик, промолчавший час, теряет час данных навсегда: раздел `/futures/data/`
отдаёт только последние 30 суток, дальше -1130 (проверено на всех пяти
эндпоинтах).

Отсюда три правила:

1. **Сначала бэкфилл, потом дозапись.** Тридцать суток на бирже ещё лежат, и
   забрать их надо сегодня. Стартовать «с текущего момента» — выбросить месяц,
   который пока существует.
2. **Один упавший символ не отменяет прогон.** Тот же принцип, что и в
   снапшоте: пишем что смогли, а что не смогли — записываем в журнал прогонов.
3. **Каждый прогон оставляет след.** Молчащий сборщик снаружи неотличим от
   работающего, а обнаружить это через месяц будет поздно.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import sqlite3
import time
from typing import Any

from . import storage
from .client import BinanceClient
from .errors import ToolError
from .markets import FUTURES, SPOT
from .series import build_series

log = logging.getLogger("cryptomcp.collector")

#: Шаг ряда деривативов. Пятиминутный, а не часовой: агрегировать до часа можно
#: в любой момент, восстановить час до пяти минут — никогда. Именно эта
#: детализация показала на CAKE, что суточная разгрузка OI на 8% — одно
#: событие в 20:00, а не равномерный фон.
PERIOD = "5m"
PERIOD_MS = 5 * 60 * 1000

#: Глубина истории, которую биржа вообще отдаёт. Ровно 30 суток: запрос за 29
#: проходит, за 31 — HTTP 400 с кодом -1130.
HISTORY_DAYS = 30

#: Точек за один запрос. Проверено: 1000 отдаётся, это 83 часа с шагом 5m,
#: то есть тридцать суток забираются девятью запросами на эндпоинт.
PAGE = 1000

#: Как поле ответа превращается в колонку таблицы.
SOURCES: tuple[tuple[str, str, str], ...] = (
    ("open_interest_hist", "sumOpenInterest", "open_interest"),
    ("open_interest_hist", "sumOpenInterestValue", "open_interest_value"),
    ("global_account", "longShortRatio", "ls_global"),
    ("top_account", "longShortRatio", "ls_top_accounts"),
    ("top_position", "longShortRatio", "ls_top_positions"),
    ("taker", "buySellRatio", "taker_ratio"),
)

#: Сколько символов обрабатывать параллельно. Клиент и так держит семафор и
#: бюджет веса; здесь ограничение нужно, чтобы не плодить тысячи задач разом.
SYMBOL_CONCURRENCY = 3

#: Лестница архива свечей: таймфрейм и глубина в сутках.
#:
#: 15m и мельче не хранятся сознательно. Год пятнадцатиминуток — 35 040 свечей
#: на монету, это больше, чем вся остальная лестница вместе (26 175), то есть
#: архив ровно удваивается. При этом свечи, в отличие от открытого интереса,
#: биржа отдаёт за годы: докачать 15m за любой период — три минуты и один
#: проход. Асимметрия и решает: у деривативов «потом» означает «никогда», у
#: свечей — «когда понадобится».
LADDER: tuple[tuple[str, int], ...] = (
    ("1w", 5 * 365),
    ("1d", 5 * 365),
    ("4h", 3 * 365),
    ("1h", 2 * 365),
)

#: Средний слой: монеты вне ядра архивируются только по старшим ТФ.
MID_LADDER: tuple[tuple[str, int], ...] = (("1d", 5 * 365), ("4h", 3 * 365))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


#: Порог оборота для ядра, по которому пишутся деривативы. Пока не накоплен
#: universe_daily, отбор идёт по текущему обороту; когда накопится — заменить
#: на медиану за неделю, ради которой снимки и пишутся.
CORE_MIN_VOLUME = _env_float("COLLECTOR_CORE_MIN_VOLUME", 50_000_000)
CORE_MAX_SYMBOLS = _env_int("COLLECTOR_MAX_SYMBOLS", 60)

#: Порог попадания в архив свечей и в суточный снимок универсума. Ниже ядра:
#: интересные сжатия чаще встречаются в диапазоне 10–50M, а хранение свечей
#: по старшим ТФ стоит копейки — 0.9 МБ на монету за пять лет.
ARCHIVE_MIN_VOLUME = _env_float("COLLECTOR_ARCHIVE_MIN_VOLUME", 10_000_000)

INTERVAL_S = _env_int("COLLECTOR_INTERVAL_S", 3600)


async def _page(
    client: BinanceClient, source: str, symbol: str, end_ms: int
) -> list[dict[str, Any]]:
    """Один запрос к одному эндпоинту раздела /futures/data/."""
    if source == "open_interest_hist":
        return await client.open_interest_hist(
            symbol, PERIOD, limit=PAGE, end_time=end_ms
        )
    if source == "taker":
        return await client.taker_long_short_ratio(symbol, PERIOD, limit=PAGE, end_time=end_ms)
    return await client.long_short_ratio(
        symbol, source, PERIOD, limit=PAGE, end_time=end_ms
    )


async def _series(
    client: BinanceClient, source: str, symbol: str, start_ms: int, end_ms: int
) -> dict[int, dict[str, Any]]:
    """Весь ряд одного эндпоинта за окно, постранично НАЗАД.

    Листать вперёд по startTime нельзя: раздел /futures/data/ его для позиции
    не использует. Проверено — запрос с startTime за 25 суток назад вернул не
    начало окна, а последние 1000 точек, то есть свежий хвост. А вот endTime
    честно задаёт правую границу: ответ — последние ``limit`` точек до неё.
    Поэтому курсор идёт справа налево, от свежего к старому.
    """
    out: dict[int, dict[str, Any]] = {}
    cursor = end_ms
    while cursor > start_ms:
        rows = await _page(client, source, symbol, cursor)
        if not rows:
            break
        stamps = [int(row["timestamp"]) for row in rows]
        for row, ts in zip(rows, stamps, strict=True):
            if ts >= start_ms:
                out[ts] = row
        low = min(stamps)
        if low <= start_ms or len(rows) < PAGE:
            break
        cursor = low - 1
    return out


async def fetch_window(
    client: BinanceClient, symbol: str, start_ms: int, end_ms: int
) -> dict[int, dict[str, float]]:
    """Пять рядов за окно, сведённые в одну точку на отметку времени.

    Ряды идут по общей пятиминутной сетке, но с разной свежестью: taker-ratio
    отстаёт от открытого интереса на пару точек. Поэтому сведение — внешнее
    объединение по времени, а недостающие колонки остаются пустыми и
    дозаписываются следующим прогоном.
    """
    wanted = sorted({source for source, _, _ in SOURCES})
    fetched = await asyncio.gather(
        *(_series(client, source, symbol, start_ms, end_ms) for source in wanted),
        return_exceptions=True,
    )

    merged: dict[int, dict[str, float]] = {}
    for source, result in zip(wanted, fetched, strict=True):
        if isinstance(result, BaseException):
            # Один эндпоинт мог отвалиться — остальные всё равно надо сохранить.
            log.warning("%s %s: %s", symbol, source, result)
            continue
        for ts, row in result.items():
            point = merged.setdefault(ts, {})
            for row_source, field, column in SOURCES:
                if row_source == source and field in row:
                    point[column] = float(row[field])
    return merged


async def collect_symbol(
    client: BinanceClient,
    con: sqlite3.Connection,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> int:
    rows = await fetch_window(client, symbol, start_ms, end_ms)
    written = storage.upsert_derivatives(con, symbol, rows)
    con.commit()
    return written


async def collect(
    client: BinanceClient,
    con: sqlite3.Connection,
    symbols: list[str],
    *,
    days: float | None = None,
) -> tuple[int, list[str]]:
    """Собрать деривативы по списку символов.

    ``days`` задаёт окно принудительно (бэкфилл). Без него окно начинается от
    последней сохранённой точки символа, а если её нет — от границы доступной
    истории: пропускать месяц, который биржа ещё отдаёт, незачем.
    """
    now_ms = await client.now_ms()
    horizon = now_ms - HISTORY_DAYS * 86_400_000
    total = 0
    failed: list[str] = []
    guard = asyncio.Semaphore(SYMBOL_CONCURRENCY)

    async def one(symbol: str) -> None:
        nonlocal total
        async with guard:
            if days is not None:
                start = now_ms - int(days * 86_400_000)
            else:
                saved = storage.last_ts(con, "derivatives", symbol)
                start = saved + 1 if saved is not None else horizon
            start = max(start, horizon)
            if start >= now_ms:
                return
            try:
                total += await collect_symbol(client, con, symbol, start, now_ms)
            except ToolError as error:
                log.warning("%s: %s", symbol, error.message)
                failed.append(symbol)

    await asyncio.gather(*(one(symbol) for symbol in symbols))
    return total, failed


async def load_candles(
    client: BinanceClient,
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    depth_days: int,
    source: str,
) -> int:
    """Догрузить свечи одного символа и таймфрейма.

    Листается ВПЕРЁД по startTime — в отличие от раздела /futures/data/,
    обычный /klines честно отдаёт окно от заданного момента, страницы стыкуются
    без разрывов (проверено на обоих рынках). Курсор ставится на последнюю
    СОХРАНЁННУЮ свечу, а не после неё: биржа доправляет её в первые мгновения
    после закрытия, и перезаписать дешевле, чем сохранить смещённую.
    """
    now_ms = await client.now_ms()
    saved = storage.last_ohlcv_ts(con, symbol, tf)
    cursor = saved if saved is not None else now_ms - depth_days * 86_400_000
    page = client.market.max_limit
    written = 0

    while True:
        raw = await client.klines(symbol, tf, limit=page, start_time=cursor)
        if not raw:
            break
        # Незакрытая свеча отсекается тем же кодом, что и в онлайновом пути:
        # её объём и диапазон неполны, а в архиве это осталось бы навсегда.
        series = build_series(raw, symbol, tf, now_ms)
        if len(series):
            frame = series.df
            # tolist(), а не itertuples: sqlite3 не умеет привязывать numpy.int64
            # (float64 проходит как подкласс float, а целые — нет).
            columns = (
                "open_time", "open", "high", "low", "close", "volume_base",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
            )
            rows = list(zip(*(frame[c].tolist() for c in columns), strict=True))
            written += storage.upsert_ohlcv(con, symbol, tf, source, rows)
            con.commit()
        last_open = int(raw[-1][0])
        if len(raw) < page or last_open <= cursor:
            break
        cursor = last_open + 1
    return written


async def collect_candles(
    clients: dict[str, BinanceClient],
    con: sqlite3.Connection,
    plan: list[tuple[str, str, tuple[tuple[str, int], ...]]],
) -> tuple[int, list[str]]:
    """Пройти по плану «символ, источник, лестница» и догрузить свечи."""
    total = 0
    failed: list[str] = []
    guard = asyncio.Semaphore(SYMBOL_CONCURRENCY)

    async def one(symbol: str, source: str, ladder: tuple[tuple[str, int], ...]) -> None:
        nonlocal total
        async with guard:
            client = clients[source]
            for tf, depth in ladder:
                try:
                    total += await load_candles(client, con, symbol, tf, depth, source)
                except ToolError as error:
                    log.warning("%s %s (%s): %s", symbol, tf, source, error.message)
                    failed.append(f"{symbol} {tf}")

    await asyncio.gather(*(one(*item) for item in plan))
    return total, failed


async def archive_plan(
    con: sqlite3.Connection,
    core: list[str],
    mid: list[str],
    spot_symbols: set[str],
) -> list[tuple[str, str, tuple[tuple[str, int], ...]]]:
    """Кому какой источник и какая лестница.

    Источник выбирается один раз и потом берётся из уже накопленных данных.
    Спот предпочтительнее: перпетуал ценово производен от него, а история
    глубже (у CAKE спот с 2021 года против фьючерса с 2023). Но четверть
    ликвидных перпетуалов спотовой пары не имеет вовсе — для них источником
    остаётся фьючерс, и колонка source не даёт их потом перепутать.
    """
    plan = []
    for symbols, ladder in ((core, LADDER), (mid, MID_LADDER)):
        for symbol in symbols:
            source = storage.archive_source(con, symbol)
            if source is None:
                source = SPOT.name if symbol in spot_symbols else FUTURES.name
            plan.append((symbol, source, ladder))
    return plan


async def universe_rows(client: BinanceClient) -> list[dict[str, Any]]:
    """Все торгуемые перпетуалы к USDT, по убыванию оборота за сутки."""
    from .symbols import SymbolRegistry

    registry = SymbolRegistry(client)
    tradable = {info.symbol for info in await registry.tradable()}
    tickers = await client.ticker_24hr()
    rows = [
        {
            "symbol": row["symbol"],
            "quote_volume_24h": float(row["quoteVolume"]),
            "price": float(row["lastPrice"]),
        }
        for row in tickers
        if row["symbol"] in tradable
    ]
    rows.sort(key=lambda row: -row["quote_volume_24h"])
    return rows


def core_symbols(rows: list[dict[str, Any]]) -> list[str]:
    """Ядро: полная лестница свечей плюс деривативы."""
    return [
        row["symbol"]
        for row in rows
        if row["quote_volume_24h"] >= CORE_MIN_VOLUME
    ][:CORE_MAX_SYMBOLS]


def archived_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Всё, что вообще попадает в архив и в суточный снимок универсума."""
    return [row for row in rows if row["quote_volume_24h"] >= ARCHIVE_MIN_VOLUME]


async def snapshot_universe(
    client: BinanceClient,
    spot: BinanceClient,
    con: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> tuple[int, set[str]]:
    """Суточный снимок универсума.

    Пишется с первого дня, даже пока сканера нет: отбор по устойчивому обороту
    требует истории оборотов, а задним числом её взять негде. Заодно
    отмечается наличие спотовой пары — четверть ликвидных перпетуалов её не
    имеет, и для них история свечей берётся с фьючерсов.
    """
    from .symbols import SymbolRegistry

    spot_symbols = {info.symbol for info in await SymbolRegistry(spot).tradable()}
    known = storage.known_symbols(con, FUTURES.name)

    for row in rows:
        symbol = row["symbol"]
        row["has_spot"] = symbol in spot_symbols
        try:
            current = await client.open_interest(symbol)
            row["open_interest_usdt"] = float(current["openInterest"]) * row["price"]
        except ToolError as error:
            log.warning("%s: открытый интерес недоступен: %s", symbol, error.message)
        if symbol not in known:
            # Дата листинга спрашивается один раз на символ, дальше возраст
            # считается арифметикой.
            first = await client.klines(symbol, "1d", limit=1, start_time=0)
            storage.remember_symbol(
                con, symbol, FUTURES.name, int(first[0][0]) if first else None
            )

    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    written = storage.upsert_universe(con, today, rows)
    con.commit()
    return written, spot_symbols


async def run_once(con: sqlite3.Connection, *, backfill_days: float | None) -> None:
    """Один проход: деривативы, снимок универсума, свечи.

    Порядок не случаен. Деривативы первыми, потому что только они пропадают
    безвозвратно; свечи последними, потому что их можно догрузить и завтра, и
    через месяц. Если прогон оборвётся на середине, потеряно будет самое
    дешёвое.
    """
    client = BinanceClient(FUTURES)
    spot = BinanceClient(SPOT)
    try:
        rows = await universe_rows(client)
        core = core_symbols(rows)
        archived = archived_rows(rows)
        kind = "backfill" if backfill_days else "incremental"
        log.info("%s: ядро %d, архив %d символов", kind, len(core), len(archived))

        started = time.monotonic()
        written, failed = await collect(client, con, core, days=backfill_days)
        storage.record_run(
            con, kind, symbols=len(core), rows=written,
            seconds=time.monotonic() - started,
            error=("не собраны: " + ", ".join(failed)) if failed else None,
        )
        con.commit()
        log.info("%s: записано точек %d, не собрано символов %d", kind, written, len(failed))

        started = time.monotonic()
        count, spot_symbols = await snapshot_universe(client, spot, con, archived)
        storage.record_run(
            con, "universe", symbols=count, rows=count,
            seconds=time.monotonic() - started,
        )
        con.commit()
        log.info("universe_daily: %d строк", count)

        started = time.monotonic()
        core_set = set(core)
        mid = [row["symbol"] for row in archived if row["symbol"] not in core_set]
        plan = await archive_plan(con, core, mid, spot_symbols)
        candles, candle_failures = await collect_candles(
            {SPOT.name: spot, FUTURES.name: client}, con, plan
        )
        storage.record_run(
            con, "ohlcv", symbols=len(plan), rows=candles,
            seconds=time.monotonic() - started,
            error=("не собраны: " + ", ".join(candle_failures[:20]))
            if candle_failures else None,
        )
        con.commit()
        log.info("свечи: записано %d, не собрано рядов %d", candles, len(candle_failures))
    finally:
        await client.aclose()
        await spot.aclose()


async def loop(con: sqlite3.Connection) -> None:
    """Бесконечный цикл дозаписи.

    Первый проход при пустой базе забирает всю доступную историю: окно и так
    начинается от границы 30 суток, если сохранённых точек нет.
    """
    while True:
        try:
            await run_once(con, backfill_days=None)
        except Exception as error:  # noqa: BLE001 — цикл не должен умирать
            log.exception("прогон не удался: %s", error)
            storage.record_run(
                con, "incremental", symbols=0, rows=0, seconds=0.0, error=str(error)
            )
            con.commit()
        await asyncio.sleep(INTERVAL_S)


def health(con: sqlite3.Connection, *, interval_s: int = INTERVAL_S) -> tuple[bool, str]:
    """Был ли успешный прогон за последние два интервала.

    Признак живости именно такой, а не «процесс запущен»: процесс может быть
    жив и час за часом ловить отказ биржи, а снаружи это выглядит нормальной
    работой. Здесь спрашивается результат, а не наличие.
    """
    row = con.execute(
        "SELECT ts_ms, kind, rows, error FROM collector_runs "
        "WHERE kind IN ('incremental', 'backfill') ORDER BY ts_ms DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return False, "прогонов ещё не было"
    age_s = dt.datetime.now(dt.UTC).timestamp() - row["ts_ms"] / 1000
    when = dt.datetime.fromtimestamp(row["ts_ms"] / 1000, dt.UTC).isoformat(" ", "seconds")
    if age_s > 2 * interval_s:
        return False, f"последний прогон {when} UTC — {age_s / 3600:.1f} ч назад"
    return True, f"последний прогон {when} UTC, точек {row['rows']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборщик рыночных данных")
    parser.add_argument(
        "command",
        choices=("once", "backfill", "loop", "health"),
        nargs="?",
        default="loop",
    )
    parser.add_argument("--days", type=float, default=HISTORY_DAYS)
    parser.add_argument("--db", default=storage.DEFAULT_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("CRYPTOMCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    con = storage.connect(args.db)
    try:
        if args.command == "health":
            alive, message = health(con)
            log.info("%s", message)
            raise SystemExit(0 if alive else 1)
        if args.command == "loop":
            asyncio.run(loop(con))
        else:
            days = args.days if args.command == "backfill" else None
            asyncio.run(run_once(con, backfill_days=days))
    finally:
        con.close()


if __name__ == "__main__":
    main()
