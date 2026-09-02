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

from . import SQUEEZE_FORMULA_VERSION, storage
from .analysis import MIN_CANDLES, analyse_timeframe
from .client import BinanceClient
from .config import Config
from .errors import ToolError
from .markets import FUTURES, SPOT
from .series import build_series, series_from_records

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

#: На каких таймфреймах считать индекс в фоновом скане.
SCAN_TIMEFRAMES = ("4h", "1d")

#: Горизонты, на которых замеряется исход. По High/Low внутри периода, а не по
#: цене закрытия: цель, которую цена достала и с которой откатилась, по
#: закрытию не засчиталась бы вовсе.
HORIZONS: tuple[tuple[str, int], ...] = (
    ("24h", 24 * 3_600_000),
    ("72h", 72 * 3_600_000),
    ("7d", 7 * 86_400_000),
)

config = Config.load()


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

#: Границы ранга для списка наблюдения. Вход выше выхода — гистерезис: при
#: одинаковых порогах монеты у границы входили бы и выходили каждый прогон.
WATCH_ENTER_RANK = _env_int("WATCH_ENTER_RANK", 15)
WATCH_EXIT_RANK = _env_int("WATCH_EXIT_RANK", 40)

#: Сколько эпизод живёт без развязки, суток.
WATCH_MAX_DAYS = _env_int("WATCH_MAX_DAYS", 30)


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


def run_scan(con: sqlite3.Connection) -> int:
    """Посчитать индекс по всему архиву и записать в scan_log.

    Ни одного запроса к бирже: свечи уже лежат в базе, а перцентилям хватает
    её глубины (4h за три года при требуемых шестидесяти сутках). Поэтому скан
    стоит только процессорного времени, и его не жалко гонять каждый час.

    Строка пишется одна на закрытую свечу — это обеспечено уникальным
    индексом в схеме. Иначе четырёхчасовое наблюдение попадало бы в статистику
    четыре раза и перевешивало бы дневное.
    """
    written = 0
    for tf in SCAN_TIMEFRAMES:
        for symbol, source in storage.archived_symbols(con, tf):
            records = storage.load_candles(con, symbol, tf, limit=1500)
            series = series_from_records(records, symbol, tf)
            if len(series) < MIN_CANDLES:
                continue
            view = analyse_timeframe(series, config)
            if storage.record_scan(
                con, symbol, source, view,
                formula_version=SQUEEZE_FORMULA_VERSION,
            ):
                written += 1
    con.commit()
    return written


def settle_outcomes(con: sqlite3.Connection, now_ms: int | None = None) -> int:
    """Досчитать исходы у записей скана, чей горизонт истёк.

    Считается по самому мелкому архивному таймфрейму символа: на дневках цель,
    задетую внутри дня, видно только как диапазон свечи. И считается по
    High/Low, а не по закрытию — иначе достигнутая и откатившаяся цель не
    засчитывается вовсе.

    Запись, для которой архив ещё не покрыл весь горизонт, пропускается и
    будет взята следующим прогоном: половина окна дала бы заниженный размах.
    """
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    settled = 0
    for horizon, span in HORIZONS:
        for row in storage.pending_outcomes(con, horizon, span, now_ms):
            tf = storage.finest_tf(con, row["symbol"])
            if tf is None:
                continue
            end_ms = row["ts_ms"] + span
            last = storage.last_ohlcv_ts(con, row["symbol"], tf)
            if last is None or last < end_ms:
                continue
            window = storage.price_extremes(con, row["symbol"], tf, row["ts_ms"], end_ms)
            if not window:
                continue
            price = row["price"]
            storage.record_outcome(con, row["id"], horizon, {
                "max_price": window["high"],
                "min_price": window["low"],
                "max_pct": round((window["high"] / price - 1) * 100, 4),
                "min_pct": round((window["low"] / price - 1) * 100, 4),
                "close_pct": (
                    round((window["close"] / price - 1) * 100, 4)
                    if window["close"] else None
                ),
                "candles": window["candles"],
            })
            settled += 1
    con.commit()
    return settled


def update_watchlist(con: sqlite3.Connection, now_ms: int | None = None) -> dict[str, list]:
    """Пересчитать состав списка наблюдения и вернуть дельту к прошлому прогону.

    **Отбор рангом, а не порогом.** Измерено на 58 ликвидных монетах: порог
    индекса 0.5 не пропускал никого, 0.4 — четверых, 0.3 — треть рынка.
    Естественной границы между ними нет, а взять её сейчас неоткуда — исходы
    ещё не накоплены. Любое число было бы угадыванием, и через месяц
    выяснилось бы, что сканер месяц молчал или месяц шумел. Ранг
    самонормируется: при общем сжатии рынка список не переполняется, при
    расширении не пустеет. Когда outcomes покажут, при каких значениях индекс
    что-то предсказывал, ранг заменится на измеренный порог.

    **Гистерезис по рангу.** Вход в топ-15, выход из топ-40. Одинаковые пороги
    заставляли бы монеты у границы входить и выходить на каждом прогоне.

    **Длительность сжатия — не ворота, а колонка.** Из 58 монет `узк > 0`
    было у двух, и это BTC с BNB — самые ликвидные и наименее интересные для
    игры на сжатии. Требование «все три условия сразу» отобрало бы их одних.

    Порядок проверок при закрытии важен: пробой раньше ранга. Монета, которая
    выстрелила и на этом вылетела из топа, должна попасть в статистику как
    сработавшая, а не как «выпала по рангу».
    """
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    changes: dict[str, list] = {"entered": [], "exited": [], "promoted": []}

    for tf in SCAN_TIMEFRAMES:
        rows = storage.latest_scan(con, tf)
        rank = {row["symbol"]: i + 1 for i, row in enumerate(rows)}
        by_symbol = {row["symbol"]: row for row in rows}

        for episode in storage.open_episodes(con, tf):
            symbol = episode["symbol"]
            scan = by_symbol.get(symbol)
            position = rank.get(symbol)
            manual = episode["entered_by"] == "manual"

            if scan and _broke_out(episode, scan):
                storage.close_episode(
                    con, episode["id"], status="broken_out",
                    reason="пробой диапазона входа", ts_ms=now_ms,
                )
                changes["exited"].append((symbol, tf, "пробой"))
                continue

            if now_ms - episode["entered_at"] > WATCH_MAX_DAYS * 86_400_000:
                storage.close_episode(
                    con, episode["id"], status="expired",
                    reason=f"{WATCH_MAX_DAYS} суток без развязки", ts_ms=now_ms,
                )
                changes["exited"].append((symbol, tf, "истёк срок"))
                continue

            if not manual and (position is None or position > WATCH_EXIT_RANK):
                storage.close_episode(
                    con, episode["id"], status="expired",
                    reason=(
                        f"ранг {position} ниже {WATCH_EXIT_RANK}" if position
                        else "выпала из универсума"
                    ),
                    ts_ms=now_ms,
                )
                changes["exited"].append((symbol, tf, "выпала по рангу"))
                continue

            promote = (
                episode["status"] == "candidate"
                and position is not None
                and position <= WATCH_EXIT_RANK
            )
            storage.touch_episode(
                con, episode["id"], rank=position,
                index=scan["squeeze_index"] if scan else None, promote=promote,
            )
            if promote:
                changes["promoted"].append((symbol, tf, position))

        for position, row in enumerate(rows[:WATCH_ENTER_RANK], start=1):
            episode_id = storage.open_episode(
                con, row["symbol"], tf, entered_at=now_ms,
                entered_by="scanner", scan=row, rank=position,
            )
            if episode_id:
                changes["entered"].append((row["symbol"], tf, position))

    con.commit()
    return changes


def _broke_out(episode: dict[str, Any], scan: dict[str, Any]) -> bool:
    """Вышла ли цена за диапазон, зафиксированный на входе."""
    price = scan.get("price")
    low, high = episode.get("range_low"), episode.get("range_high")
    if price is None or low is None or high is None:
        return False
    return bool(price > high or price < low)


def add_to_watchlist(
    con: sqlite3.Connection, symbol: str, tf: str = "4h", now_ms: int | None = None
) -> int:
    """Ручное добавление.

    Обязательная часть: сканер видит только то, что умеет измерять. Ручные
    записи не выбывают по рангу — только по пробою, сроку или вручную.
    """
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    rows = {row["symbol"]: row for row in storage.latest_scan(con, tf)}
    episode_id = storage.open_episode(
        con, symbol.upper(), tf, entered_at=now_ms, entered_by="manual",
        scan=rows.get(symbol.upper(), {}),
    )
    con.commit()
    return episode_id


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

        # Скан и исходы считаются из базы, к бирже не ходят вовсе, поэтому
        # стоят копейки и идут последними — после того как архив пополнен.
        started = time.monotonic()
        scanned = run_scan(con)
        settled = settle_outcomes(con)
        changes = update_watchlist(con)
        storage.record_run(
            con, "scan", symbols=scanned, rows=scanned + settled,
            seconds=time.monotonic() - started,
        )
        con.commit()
        log.info("скан: новых записей %d, посчитано исходов %d", scanned, settled)
        log.info(
            "watchlist: вошло %d, вышло %d, подтверждено %d",
            len(changes["entered"]), len(changes["exited"]), len(changes["promoted"]),
        )
        for symbol, tf, extra in changes["entered"]:
            log.info("  + %s %s (ранг %s)", symbol, tf, extra)
        for symbol, tf, reason in changes["exited"]:
            log.info("  - %s %s (%s)", symbol, tf, reason)
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


def render_watchlist(con: sqlite3.Connection) -> str:
    """Открытые эпизоды столбиком.

    Ширина диапазона и длительность сжатия печатаются отдельными колонками, а
    не сворачиваются в индекс: ранг сам по себе пропускает и широкие диапазоны
    (BULLA с 40% попала в топ на первой же выдаче), и решать, что с этим
    делать, должен человек.
    """
    episodes = storage.open_episodes(con)
    if not episodes:
        return "список наблюдения пуст"

    lines = [
        f"открытых эпизодов: {len(episodes)}",
        f"{'символ':<14}{'ТФ':>4}{'статус':>11}{'ранг':>6}{'индекс':>8}"
        f"{'вход':>10}{'сейчас':>9}{'дней':>6}  кем",
    ]
    now_ms = int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    for episode in sorted(episodes, key=lambda e: (e["tf"], e["last_rank"] or 999)):
        scan = {row["symbol"]: row for row in storage.latest_scan(con, episode["tf"])}
        price = (scan.get(episode["symbol"]) or {}).get("price")
        move = (
            f"{(price / episode['price_at_entry'] - 1) * 100:+.1f}%"
            if price and episode["price_at_entry"] else "n/a"
        )
        lines.append(
            f"{episode['symbol']:<14}{episode['tf']:>4}{episode['status']:>11}"
            f"{episode['last_rank'] or 0:>6}{episode['last_index'] or 0:>8.2f}"
            f"{episode['price_at_entry'] or 0:>10.4g}{move:>9}"
            f"{(now_ms - episode['entered_at']) / 86_400_000:>6.1f}  "
            f"{episode['entered_by']}"
        )
    return "\n".join(lines)


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
        choices=("once", "backfill", "loop", "health", "watch", "add"),
        nargs="?",
        default="loop",
    )
    parser.add_argument("symbol", nargs="?", help="символ для команды add")
    parser.add_argument("--tf", default="4h")
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
        if args.command == "watch":
            print(render_watchlist(con))
            return
        if args.command == "add":
            if not args.symbol:
                raise SystemExit("нужен символ: add CAKEUSDT [--tf 4h]")
            added = add_to_watchlist(con, args.symbol, args.tf)
            print(
                f"{args.symbol.upper()} {args.tf}: "
                + ("добавлена вручную" if added else "уже в списке")
            )
            return
        if args.command == "loop":
            asyncio.run(loop(con))
        else:
            days = args.days if args.command == "backfill" else None
            asyncio.run(run_once(con, backfill_days=days))
    finally:
        con.close()


if __name__ == "__main__":
    main()
