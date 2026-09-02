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


async def core_universe(client: BinanceClient) -> list[dict[str, Any]]:
    """Ядро: перпетуалы к USDT, отсортированные по обороту за сутки."""
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
    return [row for row in rows if row["quote_volume_24h"] >= CORE_MIN_VOLUME][
        :CORE_MAX_SYMBOLS
    ]


async def snapshot_universe(
    client: BinanceClient, spot: BinanceClient, con: sqlite3.Connection
) -> int:
    """Суточный снимок универсума.

    Пишется с первого дня, даже пока сканера нет: отбор по устойчивому обороту
    требует истории оборотов, а задним числом её взять негде. Заодно
    отмечается наличие спотовой пары — четверть ликвидных перпетуалов её не
    имеет, и для них история свечей берётся с фьючерсов.
    """
    from .symbols import SymbolRegistry

    rows = await core_universe(client)
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
    return written


async def run_once(con: sqlite3.Connection, *, backfill_days: float | None) -> None:
    """Один проход: деривативы, затем снимок универсума."""
    client = BinanceClient(FUTURES)
    spot = BinanceClient(SPOT)
    try:
        started = time.monotonic()
        universe = await core_universe(client)
        symbols = [row["symbol"] for row in universe]
        kind = "backfill" if backfill_days else "incremental"
        log.info("%s: %d символов", kind, len(symbols))

        written, failed = await collect(client, con, symbols, days=backfill_days)
        storage.record_run(
            con, kind, symbols=len(symbols), rows=written,
            seconds=time.monotonic() - started,
            error=("не собраны: " + ", ".join(failed)) if failed else None,
        )
        con.commit()
        log.info("%s: записано точек %d, не собрано символов %d", kind, written, len(failed))

        started = time.monotonic()
        count = await snapshot_universe(client, spot, con)
        storage.record_run(
            con, "universe", symbols=count, rows=count,
            seconds=time.monotonic() - started,
        )
        con.commit()
        log.info("universe_daily: %d строк", count)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборщик рыночных данных")
    parser.add_argument(
        "command", choices=("once", "backfill", "loop"), nargs="?", default="loop"
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
        if args.command == "loop":
            asyncio.run(loop(con))
        else:
            days = args.days if args.command == "backfill" else None
            asyncio.run(run_once(con, backfill_days=days))
    finally:
        con.close()


if __name__ == "__main__":
    main()
