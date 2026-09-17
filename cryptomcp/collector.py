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
import statistics
import time
from typing import Any

from . import (
    SQUEEZE_FORMULA_VERSION,
    distribution,
    flow,
    manual,
    orderbook_watch,
    render,
    storage,
)
from .analysis import MIN_CANDLES, analyse_timeframe, required_candles
from .client import BinanceClient
from .config import Config
from .derivatives import build_open_interest, side_of_flow
from .errors import ToolError
from .indicators import atr
from .markets import FUTURES, SPOT
from .notify import Telegram, notify_accumulation, notify_watchlist
from .reader import WARMUP
from .series import build_series, interval_ms, series_from_records
from .volume import absorption

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

#: Начислений фандинга за запрос — потолок эндпоинта.
FUNDING_PAGE = 1000

#: Предохранитель от бесконечного листания, если биржа перестанет двигать
#: курсор. Года хватает даже часовому интервалу начислений: 8760 точек — девять
#: страниц.
MAX_FUNDING_PAGES = 12

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

#: Средний слой: монеты вне ядра архивируются мельче ядра, но 1h им нужен.
#:
#: Без часовок скан по 1h видел бы только ядро — 52 монеты из 121, причём
#: отобранные по обороту, то есть ровно не ту половину: сжатия чаще
#: встречаются в полосе 10–50M. Год часовок стоит 0.92 МБ на монету, на 69
#: монет — 63 МБ; перцентилю 1h нужно 60 суток, так что года хватает с запасом.
MID_LADDER: tuple[tuple[str, int], ...] = (
    ("1d", 5 * 365), ("4h", 3 * 365), ("1h", 365)
)

#: На каких таймфреймах считать индекс в фоновом скане.
#:
#: 1h добавлен, чтобы проверить гипотезу, а не потому, что она принята.
#: Довод за него: подготовка к резкому движению часто длится 1–3 суток, а на
#: дневке это две-три свечи. Довод против: единственный принесённый пример
#: (UAI 28–29 августа) при ретроспективной проверке показал обратное — 1d дал
#: индекс 0.41 при BBW на 22-м перцентиле, а 1h 0.40 при BBW на 69-м, то есть
#: сжатие увидела как раз дневка, имея данных на полтора суток меньше.
#:
#: Спор решается журналом, а не аргументами: записи 1h копятся и получают
#: исходы наравне с остальными, и через месяц будет видно, предсказывал ли
#: 1h-индекс размах лучше старших ТФ. До тех пор 1h НЕ открывает эпизодов —
#: см. WATCH_TIMEFRAMES.
SCAN_TIMEFRAMES = ("4h", "1d", "1h")

#: Ряд, по которому считаются признаки накопления, независимо от ТФ записи.
ACCUMULATION_TF = "1h"

#: На каких таймфреймах ведётся НАБОР в список наблюдения.
#:
#: Уже́ скана сознательно, и с 04.09.2026 (§4.35) — только дневка. Довод тот
#: же, по которому эпизоды не открываются на 1h: лишний ТФ поднимает потолок
#: списка на 40 эпизодов (топ-40 на выход), а список и без того признан
#: длинным. К нему добавился второй, измеренный: окно ширины на 4h — 3.3
#: суток, оно проезжает внутри месячного боковика, и любой всплеск внутри
#: диапазона обнуляет счётчик. Серии длиннее нескольких суток там почти не
#: выживают, то есть эпизоды на 4h открывались по величине, которая на вопрос
#: задачи — «стоит ли монета неделями» — не отвечает.
#:
#: Скан по 4h и 1h при этом идёт и пишется в журнал как прежде: вернуть набор
#: можно будет одной строкой, если статистика покажет, что зря убрали.
WATCH_TIMEFRAMES = ("1d",)

#: На каких таймфреймах считается соседний рынок (§4.32). Шире набора: журнал
#: по 4h читается человеком на шаге 2 протокола, и расхождение спота с перпом
#: там нужно видеть так же, как на дневке.
TWIN_TIMEFRAMES = ("1d", "4h")

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


#: Порог оборота для ядра: кому пишется полная лестница свечей (1w/1d/4h/1h).
#: Деривативы по ядру НЕ ограничены — они собираются по всему архивному слою
#: (см. run_once). Пока не накоплен universe_daily, отбор идёт по текущему
#: обороту; когда накопится — заменить на медиану за неделю, ради которой
#: снимки и пишутся.
CORE_MIN_VOLUME = _env_float("COLLECTOR_CORE_MIN_VOLUME", 50_000_000)
CORE_MAX_SYMBOLS = _env_int("COLLECTOR_MAX_SYMBOLS", 60)

#: Порог попадания в архив свечей и в суточный снимок универсума. Ниже ядра:
#: интересные сжатия чаще встречаются в диапазоне 10–50M, а хранение свечей
#: по старшим ТФ стоит копейки — 0.9 МБ на монету за пять лет.
#:
#: 3M вместо 10M с 03.09.2026 (§4.30). Замерено в тот день: 231 монета против
#: 121, и 15 из топ-20 по индексу на 4h — из полосы 3–10M, которую архив не
#: видел вовсе; HOME, с которого начался разбор происхождения сжатия, тоже
#: оттуда. Ниже 2M сознательно не идём: там стакан тонкий, takerB считается по
#: десяткам сделок, и метрики шумят.
ARCHIVE_MIN_VOLUME = _env_float("COLLECTOR_ARCHIVE_MIN_VOLUME", 3_000_000)

#: Порог удержания: монета, уже попавшая в архив, выбывает только ниже него, и
#: только если держится ниже ``ARCHIVE_KEEP_DAYS`` суток подряд (PLAN §4.40).
#:
#: Причина замерена на ONEUSDT: 06–10.09 оборот 3.0–3.7M, дальше 2.27, 1.87,
#: 2.07, 2.07M — монета выпала из скана ровно на тихой фазе перед выстрелом
#: 16.09 (+56% за сутки, к утру вдвое), а 15.09 вернулась с 3.05M. Тишина в
#: обороте — то самое состояние, которое сканер и ищет, поэтому выселять за
#: неё нельзя. Симметричный порог входа оставлен на 3M: пускать в архив по
#: 2M значило бы принять ещё 65 пар разом.
ARCHIVE_KEEP_VOLUME = _env_float("COLLECTOR_ARCHIVE_KEEP_VOLUME", 2_000_000)

#: Сколько суток подряд оборот должен быть ниже порога удержания, чтобы монета
#: выбыла. Два дня истории универсума плюс сегодняшний оборот.
ARCHIVE_KEEP_DAYS = _env_int("COLLECTOR_ARCHIVE_KEEP_DAYS", 3)

#: Сколько НОВЫХ символов впускать в архив за один прогон.
#:
#: Понижение порога сразу приводит сотню незнакомых монет, и первый же прогон
#: попытался бы взять по каждой 30 суток деривативов (36 запросов) плюс
#: лестницу свечей (около 13 страниц). Это единовременный залп в несколько
#: тысяч запросов там, где обычный прогон обходится сотнями.
#:
#: Растягивание ничего не стоит, и это ключевое свойство: окно деривативов у
#: новичка всё равно начинается от границы доступных 30 суток, а не от момента
#: приёма. Монета, впущенная через пять часов, получит ровно ту же историю, что
#: и впущенная сейчас. Терять нечего, а пик нагрузки исчезает.
NEW_SYMBOLS_PER_RUN = _env_int("COLLECTOR_NEW_SYMBOLS_PER_RUN", 25)

INTERVAL_S = _env_int("COLLECTOR_INTERVAL_S", 3600)

#: Глубина истории фандинга при первом заходе. Ограничена не биржей, а смыслом:
#: перцентилю хватает года, а строка в таблице стоит десятки байт. Обратный
#: случай к деривативам, где глубину определяет биржа своими 30 сутками.
FUNDING_BACKFILL_DAYS = _env_float("COLLECTOR_FUNDING_DAYS", 365.0)

#: Скольким кандидатам считается соседний рынок (§4.32). Ровно тем, кто
#: претендует на эпизод: это 15 символов на таймфрейм, а не 120, и потому
#: свойство «основной проход не делает запросов к бирже» остаётся в силе —
#: запросы делает короткий второй проход, а не скан универсума.
TWIN_LIMIT = _env_int("SCAN_TWIN_LIMIT", 15)

#: Отсечка шага 1 протокола: монета в движении не кандидат на накопление.
#: Считается по ЗАКРЫТЫМ свечам, как и всё остальное. Касается только
#: открытия эпизода: индекс всё равно считается и пишется в журнал, иначе
#: через два месяца нечем будет проверить, верен ли сам порог.
WATCH_MAX_CHANGE_24H = _env_float("WATCH_MAX_CHANGE_24H", 15.0)

#: Пол по длительности сжатия, в сутках. На 1d это `узк ≥ 7`.
#:
#: Абсолютный порог здесь не противоречит перцентилю в группе `duration`:
#: там вопрос «длинное ли это сжатие ДЛЯ ЭТОЙ монеты», здесь — «относится ли
#: монета к задаче вообще». Мы ищем недельные и месячные базы; сжатие в
#: полтора дня не относится к задаче по постановке, а перцентиль собственной
#: истории на это не отвечает: у монеты, которая никогда не стояла дольше
#: двух суток, полтора дня окажутся в 95-м перцентиле и пройдут любой
#: относительный фильтр.
#:
#: Число стартовое: взято из определения задачи и одного кейса (ASTER, 36
#: суток сжатия на дневке). Менять — когда `outcomes` наберут статистику по
#: длительности.
WATCH_MIN_SQUEEZE_DAYS = _env_float("WATCH_MIN_SQUEEZE_DAYS", 7.0)

#: Пол по индексу. Ноль означает «порога нет» — сегодняшнее поведение.
#:
#: Параметр заведён заранее, значение будет выбрано замером: убрав из свёртки
#: 0.20 «бесплатного веса» и добавив `duration`, распределение индекса почти
#: наверняка сдвинули вниз, и любое число, выбранное до недели наблюдений,
#: окажется неверным. Ранг остаётся: пол добавляется вместе с ним, а не
#: вместо него.
WATCH_MIN_INDEX = _env_float("WATCH_MIN_INDEX", 0.0)

#: Границы ранга для списка наблюдения. Вход выше выхода — гистерезис: при
#: одинаковых порогах монеты у границы входили бы и выходили каждый прогон.
#: Второй список — «накопление» (PLAN §4.43). Отбор не по тишине, а по
#: кластеру набора: замер 17.09.2026 показал, что верхушка индекса на исходах
#: не лучше базы (72ч, доля от +10%: 1d 44% против 45% у всех, 4h 29% против
#: 42%), а кластер — лучше (51% против 39% на своевременных строках).
#: Кластер редок: в верхушке индекса он есть у 6% строк, поэтому воротами к
#: первому списку его делать нельзя — там осталась бы одна монета из
#: пятнадцати. Отсюда отдельный список: свои правила, свои уведомления и
#: честное сравнение двух подходов по `outcomes` через два-три месяца.
ACCUMULATION_TIMEFRAMES = ("1d",)
ACCUMULATION_ENTER_RANK = _env_int("ACCUMULATION_ENTER_RANK", 15)
ACCUMULATION_EXIT_RANK = _env_int("ACCUMULATION_EXIT_RANK", 25)

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


async def funding_window(
    client: BinanceClient, symbol: str, start_ms: int, end_ms: int
) -> dict[int, float]:
    """Начисления фандинга за окно.

    **Листание здесь ПРЯМОЕ — в отличие от /futures/data/.** Тот раздел не
    использует startTime для позиции и листается назад по endTime (§4.13);
    fundingRate честно отдаёт окно вперёд от startTime. Две противоположные
    семантики границ живут в одном процессе, и перепутать их — значит молча
    забрать не тот кусок истории, что уже случалось.
    """
    rows: dict[int, float] = {}
    cursor = start_ms
    for _ in range(MAX_FUNDING_PAGES):
        page = await client.funding_rate(
            symbol, limit=FUNDING_PAGE, start_time=cursor, end_time=end_ms
        )
        if not page:
            break
        for row in page:
            rows[int(row["fundingTime"])] = float(row["fundingRate"])
        last = int(page[-1]["fundingTime"])
        # Конец окна определяется по КУРСОРУ, а не по короткой странице.
        # «Вернулось меньше, чем просили» означает конец истории только если
        # биржа не режет limit молча, а она режет: на klines с limit=1500
        # молча возвращалось 1000. Здесь цена ошибки — тихо недобранный кусок
        # истории, поэтому платим одним лишним пустым запросом на символ.
        if last <= cursor or last >= end_ms:
            break
        cursor = last + 1
    return rows


async def collect_funding(
    client: BinanceClient,
    con: sqlite3.Connection,
    symbols: list[str],
    *,
    days: float | None = None,
) -> tuple[int, list[str]]:
    """Сложить историю ставок фандинга в архив.

    Зачем хранить то, что биржа отдаёт и так: знак фандинга и его перцентиль
    нужны СТРОКОЙ СКАНА, то есть сразу по всем монетам универсума. Живым
    запросом это по обращению на монету за прогон против одного прохода по
    локальной базе. Без архива нельзя ни подписать сторону набора позиций, ни
    поставить фандинг в пакетную выдачу.

    Глубина ограничена не биржей, а смыслом: история фандинга отдаётся за годы
    (проверено на 900 сутках), перцентилю хватает года, а хранение стоит
    копейки. Это ровно обратный случай к деривативам, где решает биржа.
    """
    now_ms = await client.now_ms()
    horizon = now_ms - int((days or FUNDING_BACKFILL_DAYS) * 86_400_000)
    total = 0
    failed: list[str] = []
    guard = asyncio.Semaphore(SYMBOL_CONCURRENCY)

    async def one(symbol: str) -> None:
        nonlocal total
        async with guard:
            saved = storage.last_ts(con, "funding", symbol)
            start = saved + 1 if saved is not None else horizon
            if start >= now_ms:
                return
            try:
                rows = await funding_window(client, symbol, start, now_ms)
            except ToolError as error:
                log.warning("%s: фандинг не собран: %s", symbol, error.message)
                failed.append(symbol)
                return
            if rows:
                total += storage.upsert_funding(con, symbol, sorted(rows.items()))

    await asyncio.gather(*(one(symbol) for symbol in symbols))
    con.commit()
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


def funding_annual(settlements: list[tuple[int, float]]) -> float | None:
    """Годовая ставка по архивным начислениям.

    Интервал начисления берётся из САМИХ данных — по медианному промежутку
    между начислениями. Он различается у монет (1, 4 или 8 часов), а спросить
    его у биржи значило бы сделать запрос там, где скан их не делает вовсе.
    """
    if len(settlements) < 3:
        return None
    stamps = [ts for ts, _ in settlements[-30:]]
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
    if not gaps:
        return None
    interval_h = statistics.median(gaps) / 3_600_000
    if interval_h <= 0:
        return None
    return settlements[-1][1] * (24 * 365 / interval_h) * 100.0


def accumulation_context(
    con: sqlite3.Connection, symbol: str, as_of_ms: int
) -> dict[str, Any]:
    """Признаки накопления по символу на момент ``as_of_ms`` — всё из архива.

    ``as_of_ms`` — закрытие свечи, к которой пишется строка журнала, а не время
    прогона. Раньше сюда шло время прогона, и строка получала признаки из
    будущего своей свечи: половина дневных строк пишется позже часа после
    закрытия, а при доборе истории — через дни. У ONEUSDT 16.09 строка по
    свече 20:00 записана в 23:45 с «притоком новых денег +37%»: это был сам
    выстрел, начавшийся в 20:00, а до него открытый интерес снижался. На
    поздних строках признак «OI за сутки > +10%» давал 57% исходов от +10%
    против 45% на своевременных — калибровка по журналу училась бы на
    заглядывании вперёд.

    Пишутся с первого дня по той же причине, по которой с первого дня пишется
    разложение индекса: через два месяца вопрос будет не «работает ли
    накопление», а «какой из его признаков работает», и ответить на него можно
    только по журналу. Вес ни одному из них пока не назначен.

    Считается на 1h независимо от таймфрейма записи: дневное разрешение стирает
    поглощение внутри свечи целиком (§4.19). Для записей самого 1h признаки не
    пишутся — там младшего ряда в архиве нет, а считать поглощение по своему же
    ряду значило бы повторить объёмную группу.
    """
    context: dict[str, Any] = {}

    # Свеча входит в окно, только если закрылась к ``as_of_ms``: граница по
    # времени открытия, потому что ts в архиве — открытие свечи.
    records = storage.load_candles(
        con, symbol, ACCUMULATION_TF, required_candles(ACCUMULATION_TF) + WARMUP,
        before_ms=as_of_ms - interval_ms(ACCUMULATION_TF) + 1,
    )
    series = series_from_records(records, symbol, ACCUMULATION_TF)
    if len(series) >= MIN_CANDLES:
        atr_values = atr(series.high, series.low, series.close, 14)
        data = absorption(series, atr_values)
        context.update(
            absorption_tf=data.interval,
            absorption_bars=data.bars,
            taker_max=round(data.taker_max, 4),
            taker_above=data.taker_above,
            taker_streak=data.taker_streak,
            volume_lead=data.lead_bars,
            lead_state=data.lead_state,
            absorption_clusters=data.clusters,
            cluster_longest=data.cluster_longest,
            wick_streak=data.wick_streak,
        )

    settlements = storage.funding_window(con, symbol, as_of_ms, limit=60)
    annual = funding_annual(settlements)
    if annual is not None:
        context["funding_annual"] = round(annual, 2)

    rows = storage.derivatives_window(con, symbol, as_of_ms - 2 * 86_400_000, as_of_ms)
    if len(rows) > 1:
        view = build_open_interest(
            symbol, rows, PERIOD_MS // 60_000, rows, PERIOD_MS // 60_000,
            windows=("24h",), source="архив",
        )
        if "24h" in view.change:
            context["oi_change_24h"] = round(view.change["24h"] * 100, 4)
            context["oi_reading"] = side_of_flow(view.quadrant("24h"), annual)
    return context


def _flow_row(series: Any) -> dict[str, Any]:
    """Поток тейкеров колонками журнала (SPEC-flow-and-absorption-v2 §9).

    Окно 30 — основное; 60 и 90 печатаются в выдаче, но в журнал не идут:
    вопрос «не окажется ли окно 30 слишком коротким» ТЗ само выносит в
    калибровку (§12), а для неё хватает одного окна плюс исходов.
    """
    data = flow.flow(series, 30)
    ratio = flow.vol_ratio(series)
    row: dict[str, Any] = {"vol_ratio_12_30": _round(ratio.value, 4)}
    if data is None:
        return row
    row.update(
        delta_sum_30=_round(data.delta_sum, 2),
        delta_share_30=_round(data.delta_share, 4),
        delta_slope=_round(data.delta_slope, 6),
        delta_quadrant=data.quadrant,
        absorption_ratio=_round(data.absorption_ratio, 3),
        flow_change_30=_round(data.price_change, 4),
    )
    return row


def _distribution_row(
    detector: distribution.Detector, stream: dict[str, Any]
) -> dict[str, Any]:
    """Детектор и поток колонками журнала.

    Пишется с первого дня по той же причине, что и признаки набора: через два
    месяца вопрос будет не «работает ли детектор», а «на каких порогах он
    работает», и ответить на него можно только по накопленным строкам. Свои
    пороги ТЗ само называет стартовыми и неоткалиброванными (§10).
    """
    dist, absorp = detector.distribution, detector.accumulation
    return {
        **stream,
        "dist_events": dist.count,
        "dist_slope": _round(dist.slope, 8),
        "dist_drop": _round(dist.shift_atr, 3),
        "dist_verdict": dist.verdict,
        "absorp_events": absorp.count,
        "absorp_slope": _round(absorp.slope, 8),
        "absorp_rise": _round(absorp.shift_atr, 3),
        "absorp_verdict": absorp.verdict,
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None or value != value else round(value, digits)


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
    accumulation: dict[tuple[str, int], dict[str, Any]] = {}
    for tf in SCAN_TIMEFRAMES:
        for symbol, source in storage.archived_symbols(con, tf):
            # То же каноническое окно, что у сервера: иначе один и тот же
            # символ получал бы в scan_log и в снапшоте разные индексы,
            # отличающиеся только глубиной ряда.
            records = storage.load_candles(
                con, symbol, tf, required_candles(tf) + WARMUP
            )
            series = series_from_records(records, symbol, tf)
            if len(series) < MIN_CANDLES:
                continue
            view = analyse_timeframe(series, config)
            detector = distribution.analyse(
                series, atr(series.high, series.low, series.close, 14)
            )
            # Контекст накопления снимается на закрытие СВОЕЙ свечи: у 4h и 1d
            # моменты разные, поэтому ключ — символ и закрытие. Совпавшие
            # закрытия (полночь у 4h и 1d) читают ряд один раз.
            context = None
            closed = view.meta.get("closed_through_ms")
            if tf != ACCUMULATION_TF and closed is not None:
                key = (symbol, int(closed))
                if key not in accumulation:
                    accumulation[key] = accumulation_context(con, symbol, int(closed))
                context = accumulation[key]
            if storage.record_scan(
                con, symbol, source, view,
                formula_version=SQUEEZE_FORMULA_VERSION,
                accumulation=context,
                distribution=_distribution_row(detector, _flow_row(series)),
            ):
                written += 1
    con.commit()
    return written


async def scan_twin(
    clients: dict[str, BinanceClient],
    con: sqlite3.Connection,
    *,
    timeframes: tuple[str, ...] = TWIN_TIMEFRAMES,
    limit: int = TWIN_LIMIT,
    now_ms: int | None = None,
) -> int:
    """Посчитать те же величины на соседнем рынке для верхушки списка.

    **Зачем.** Архив хранит один ряд на символ, и `archive_plan` предпочитает
    спотовый — история глубже. Перпетуал при этом ценово производен от спота,
    но не равен ему: замер 03.09.2026 на HOMEUSDT 4h дал `узк 17` по споту
    против `36` по перпу на одной и той же свече, то есть вдвое. Пока в
    выдаче стояла одна величина, разница была невидима.

    **Справка, а не фильтр.** Величина соседнего рынка не участвует ни в
    ранге, ни в решении открыть эпизод. Гипотеза «открывать, только если
    сжатие подтвердилось на обоих» не проверена, и вводить её сейчас значило
    бы угадывать. Решить, какой рынок предсказательнее, можно будет по
    `outcomes`, накопленным по обеим величинам.

    **Ряд берётся одним запросом.** Каноническое окно на 4h и 1d (611 и 361
    свеча) помещается в одну страницу обоих рынков. Если окно в страницу не
    влезло, символ пропускается: короткая база дала бы перцентиль, не
    сравнимый с основным — то же расхождение, ради устранения которого весь
    этот проход и заводится.
    """
    written = 0
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    for tf in timeframes:
        window = required_candles(tf) + WARMUP
        # Верхушка живого списка, а не журнала: соседний рынок считается
        # запросом к бирже, и платить им за символ, который уже не
        # сканируется, незачем.
        for row in storage.latest_scan(con, tf, fresh_as_of_ms=now_ms)[:limit]:
            symbol = row["symbol"]
            twin = SPOT.name if row["source"] == FUTURES.name else FUTURES.name
            client = clients[twin]
            if window > client.market.max_limit:
                log.debug("%s %s: окно %d не влезает в страницу %s",
                          symbol, tf, window, twin)
                continue
            try:
                raw = await client.klines(symbol, tf, limit=window)
                series = build_series(raw, symbol, tf, await client.now_ms())
            except ToolError as error:
                # Четверть перпов не имеет спотовой пары, и наоборот — это
                # штатный ответ биржи, а не сбой прогона.
                log.debug("%s %s (%s): %s", symbol, tf, twin, error.message)
                continue
            if len(series) < MIN_CANDLES:
                continue
            storage.record_twin(
                con, int(row["id"]), market=twin,
                view=analyse_timeframe(series, config),
                # Объём и поток соседнего рынка — то, ради чего проход и
                # расширен: у 152 архивных монет из 183 спот даёт меньше 30%
                # оборота фьючерсов, и объёмные признаки считаются по
                # меньшинству торговли (PLAN §4.42).
                flow=_flow_row(series),
            )
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


def _delta_entry(
    symbol: str, tf: str, market: str | None, scan: dict[str, Any] | None, **fields: Any
) -> dict[str, Any]:
    """Запись дельты списка наблюдения.

    Словарь, а не кортеж: полей стало пять — символ, ТФ, рынок ряда, ранг или
    причина выхода и длительность сжатия. На пятом поле позиционное чтение
    начинает ошибаться молча.
    """
    return {
        "symbol": symbol, "tf": tf, "market": market,
        "narrow_bars": (scan or {}).get("narrow_bars"), **fields,
    }


def _frozen_days(scan: dict[str, Any] | None, tf: str, now_ms: int) -> float | None:
    """На сколько суток запись скана отстала, если отстала совсем.

    None у живой записи и у той, чьего возраста не знаем, — чтобы причина
    закрытия эпизода никогда не выдумывалась из отсутствующего основания.
    """
    if scan is None or storage.scan_is_fresh(scan, tf, now_ms):
        return None
    age = storage.scan_age_ms(scan, now_ms)
    return None if age is None else age / 86_400_000


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

    **Длительность сжатия — теперь ворота (§4.35).** Прежде она была только
    колонкой, и это давало список, в котором 32 эпизода из 40 отобраны без
    единой свечи сжатия. Порог — календарные семь суток, а не число свечей:
    сканер, написанный для поиска недельных и месячных баз, не должен
    открывать эпизод на полутора днях.

    **Отсечка по ходу за сутки.** Монета в движении не кандидат на
    накопление — это шаг 1 протокола, и в сканере его не было: ZAMA вошла в
    список на +15% за сутки, H на +22%, ZEST за 53 минуты до пробоя.

    Оба порога — ворота только на ВХОД. Индекс считается и пишется в журнал
    по всем монетам, включая отсеянные.

    Порядок проверок при закрытии важен: пробой раньше ранга. Монета, которая
    выстрелила и на этом вылетела из топа, должна попасть в статистику как
    сработавшая, а не как «выпала по рангу».

    **В ранге участвуют только живые записи.** Правило «выпала из универсума»
    было в коде с самого начала, но не срабатывало ни разу: `latest_scan`
    отдавал последнюю строку символа независимо от её возраста, поэтому
    переставший сканироваться символ сохранял ранг навсегда. Эпизод при этом
    замерзал целиком — индекс, `узк` и цена «сейчас» оставались от последнего
    прогона, а выдача печатала их как текущие.
    """
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    # Записи дельты — словари, а не кортежи: полей стало пять (рынок ряда и
    # длительность сжатия добавились к символу, ТФ и рангу), и позиционное
    # чтение на пятом поле начинает ошибаться молча.
    changes: dict[str, list] = {"entered": [], "exited": [], "promoted": []}

    # Обслуживаются эпизоды всех таймфреймов, на которых они открыты, а
    # набираются — только на тех, где ведётся список. Иначе эпизоды, открытые
    # до сужения набора до дневки, зависли бы в выдаче навсегда: цикл до них
    # просто не доходил бы.
    timeframes = dict.fromkeys((*WATCH_TIMEFRAMES, *storage.open_timeframes(con)))
    for tf in timeframes:
        # Ранг — только по живым записям. Символ, выпавший из универсума,
        # перестаёт сканироваться, но его последняя строка остаётся в журнале
        # навсегда: без проверки возраста она держит ранг вечно, и эпизод по
        # ней не закрывается никогда. Замер 14.09.2026: AIOTUSDT стоял первым
        # с индексом от 08.09 и печатал «+0.0%» при фактических +19.7%.
        #
        # Факты по самому эпизоду при этом берутся из последней известной
        # записи, даже замороженной: пробой, случившийся до заморозки, —
        # состоявшийся факт, и терять его значило бы записать сработавшую
        # монету в «выпала по рангу».
        rows = storage.latest_scan(con, tf)
        fresh = storage.latest_scan(con, tf, fresh_as_of_ms=now_ms)
        rank = {row["symbol"]: i + 1 for i, row in enumerate(fresh)}
        by_symbol = {row["symbol"]: row for row in rows}

        for episode in storage.open_episodes(con, tf):
            symbol = episode["symbol"]
            scan = by_symbol.get(symbol)
            position = rank.get(symbol)
            manual = episode["entered_by"] == "manual"
            # Рынок эпизода, а не текущий источник архива: числа эпизода
            # посчитаны по тому ряду, с которым он открывался.
            market = episode["market"] or (scan or {}).get("source")

            if scan and _broke_out(episode, scan):
                storage.close_episode(
                    con, episode["id"], status="broken_out",
                    reason="пробой диапазона входа", ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, market, scan, reason="пробой")
                )
                continue

            # Распределение проверяется ПОСЛЕ пробоя, но ДО ранга (§6.1).
            # Монета, снятая по признаку, должна попасть в журнал как
            # отсеянная по признаку, а не как «выпала по рангу», — иначе
            # статистика систематически потеряет ровно те случаи, ради
            # которых детектор написан.
            #
            # Подтверждённое распределение снимает монету независимо от
            # squeeze_index и узк (§6.2): сжатие описывает форму,
            # распределение — механизм, и механизм важнее. OAX в июле был
            # одновременно сжат и распределяем.
            if not manual and _dismissed(scan):
                storage.close_episode(
                    con, episode["id"], status="dismissed",
                    reason="распределение подтверждено", ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, market, scan, reason="распределение")
                )
                continue

            if now_ms - episode["entered_at"] > WATCH_MAX_DAYS * 86_400_000:
                storage.close_episode(
                    con, episode["id"], status="expired",
                    reason=f"{WATCH_MAX_DAYS} суток без развязки", ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, market, scan, reason="истёк срок")
                )
                continue

            if not manual and (position is None or position > WATCH_EXIT_RANK):
                frozen = _frozen_days(scan, tf, now_ms)
                if position:
                    reason = f"ранг {position} ниже {WATCH_EXIT_RANK}"
                    short = "выпала по рангу"
                elif frozen is not None:
                    reason = f"данные не обновлялись {frozen:.1f} сут."
                    short = "перестала обновляться"
                else:
                    reason = "выпала из универсума"
                    short = "выпала по рангу"
                storage.close_episode(
                    con, episode["id"], status="expired",
                    reason=reason, ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, market, scan, reason=short)
                )
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
                changes["promoted"].append(
                    _delta_entry(symbol, tf, market, scan, rank=position)
                )

        if tf not in WATCH_TIMEFRAMES:
            continue

        for position, row in enumerate(fresh[:WATCH_ENTER_RANK], start=1):
            if not _admissible(row, tf):
                continue
            episode_id = storage.open_episode(
                con, row["symbol"], tf, entered_at=now_ms,
                entered_by="scanner", scan=row, rank=position,
            )
            if episode_id:
                changes["entered"].append(
                    _delta_entry(row["symbol"], tf, row["source"], row, rank=position)
                )

    con.commit()
    return changes


def update_accumulation(
    con: sqlite3.Connection, now_ms: int | None = None
) -> dict[str, list]:
    """Пересчитать второй список и вернуть дельту (PLAN §4.43).

    Кандидаты — монеты, у которых в последней живой записи скана есть кластер
    набора на младшем ряду. Порядок внутри — по индексу сжатия: тишина сама по
    себе выбирает не лучше базы, но среди монет с признаком набора она остаётся
    разумным способом расставить их по очереди.

    Гистерезис тот же по смыслу, что у первого списка, но уже: вход в топ-15,
    выход из топ-25 множества монет С КЛАСТЕРОМ. Множество меньше — значит и
    границы теснее.

    Ворота на вход одни: монета, уже прошедшая за сутки больше
    ``WATCH_MAX_CHANGE_24H``, не кандидат на накопление. Порог семи суток
    сжатия сюда НЕ переносится: накопление бывает и коротким, а тишина здесь
    не является основанием отбора.
    """
    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    changes: dict[str, list] = {"entered": [], "exited": []}

    for tf in ACCUMULATION_TIMEFRAMES:
        fresh = [
            row for row in storage.latest_scan(con, tf, fresh_as_of_ms=now_ms)
            if (row.get("absorption_clusters") or 0) > 0
        ]
        ranks = {row["symbol"]: position for position, row in enumerate(fresh, start=1)}
        by_symbol = {row["symbol"]: row for row in fresh}

        for entry in storage.accumulation_entries(con, status="active", limit=200):
            if entry["tf"] != tf:
                continue
            symbol = entry["symbol"]
            scan = by_symbol.get(symbol)
            position = ranks.get(symbol)
            if position is None:
                # Кластер исчез или монета перестала сканироваться — для
                # второго списка это одно и то же: признака, по которому она
                # отобрана, больше нет.
                storage.close_accumulation(
                    con, entry["id"], reason="кластера набора больше нет", ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, entry.get("market"), None,
                                 reason="кластер пропал")
                )
                continue
            if position > ACCUMULATION_EXIT_RANK:
                storage.close_accumulation(
                    con, entry["id"],
                    reason=f"ранг {position} ниже {ACCUMULATION_EXIT_RANK}", ts_ms=now_ms,
                )
                changes["exited"].append(
                    _delta_entry(symbol, tf, entry.get("market"), scan,
                                 reason="выпала по рангу")
                )
                continue
            storage.touch_accumulation(
                con, entry["id"], rank=position,
                index=scan.get("squeeze_index"),
                clusters=scan.get("absorption_clusters"),
            )

        for position, row in enumerate(fresh[:ACCUMULATION_ENTER_RANK], start=1):
            change = row.get("change_24h_pct")
            if change is not None and abs(change) > WATCH_MAX_CHANGE_24H:
                continue
            entry_id = storage.open_accumulation(
                con, row["symbol"], tf, entered_at=now_ms, scan=row, rank=position,
            )
            if entry_id:
                changes["entered"].append(
                    _delta_entry(row["symbol"], tf, row["source"], row, rank=position)
                )

    con.commit()
    return changes


def _admissible(scan: dict[str, Any], tf: str) -> bool:
    """Проходит ли кандидат ворота открытия эпизода (§4.35).

    Ворота — только на вход в список. Индекс и все группы уже посчитаны и
    записаны в журнал, включая отсеянные случаи: без них через два месяца
    нечем будет проверить сами пороги.

    Неизвестная величина не запрещает вход: `change_24h_pct` появился вместе с
    этой правкой, и у записей прошлых версий его нет. Молча не пускать по
    отсутствующему полю — это тот же отказ, что и по превышенному порогу, но
    без основания.
    """
    change = scan.get("change_24h_pct")
    if change is not None and abs(change) > WATCH_MAX_CHANGE_24H:
        return False

    bars = scan.get("narrow_bars")
    days = (bars or 0) * interval_ms(tf) / 86_400_000
    if days < WATCH_MIN_SQUEEZE_DAYS:
        return False

    index = scan.get("squeeze_index")
    return not (index is not None and index < WATCH_MIN_INDEX)


def _dismissed(scan: dict[str, Any] | None) -> bool:
    """Подтверждено ли распределение по последней записи скана.

    Ручные записи сюда не попадают: они детектором не снимаются, только
    помечаются (§6.3) — то же правило, что уже действует для ранга.
    """
    return bool(scan) and scan.get("dist_verdict") == distribution.VERDICT_CONFIRMED


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
    # Только живые: границы диапазона запоминаются на входе и потом не
    # пересчитываются, и засеять их числами недельной давности значило бы
    # считать пробой от диапазона, которого уже нет.
    rows = {
        row["symbol"]: row
        for row in storage.latest_scan(con, tf, fresh_as_of_ms=now_ms)
    }
    episode_id = storage.open_episode(
        con, symbol.upper(), tf, entered_at=now_ms, entered_by="manual",
        scan=rows.get(symbol.upper(), {}),
        market=storage.archive_source(con, symbol.upper()),
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
    """Ядро: полная лестница свечей 1w/1d/4h/1h.

    Деривативы сюда больше не привязаны — они пишутся по всему архивному слою.
    """
    return [
        row["symbol"]
        for row in rows
        if row["quote_volume_24h"] >= CORE_MIN_VOLUME
    ][:CORE_MAX_SYMBOLS]


def archived_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Всё, что вообще попадает в архив и в суточный снимок универсума."""
    return [row for row in rows if row["quote_volume_24h"] >= ARCHIVE_MIN_VOLUME]


def kept_rows(
    rows: list[dict[str, Any]],
    known: dict[str, int | None],
    peaks: dict[str, float],
) -> list[dict[str, Any]]:
    """Архивный слой с гистерезисом: вход по 3M, выход ниже 2M и не сразу.

    Новая монета входит по ``ARCHIVE_MIN_VOLUME``. Знакомая остаётся, пока её
    оборот не опустится ниже ``ARCHIVE_KEEP_VOLUME`` и сегодня, и в последних
    снимках универсума. Выселение стоит дорого: по выбывшей монете перестают
    писаться деривативы, которых биржа не отдаёт задним числом, и обрывается
    журнал скана — именно это случилось с ONEUSDT 11–14.09.
    """
    kept = []
    for row in rows:
        volume = row["quote_volume_24h"]
        if volume >= ARCHIVE_MIN_VOLUME:
            kept.append(row)
            continue
        if row["symbol"] not in known:
            continue
        # Сегодня ниже порога удержания — смотрим, был ли выше в последние
        # снимки. Монеты без истории снимков (принята сегодня) остаются.
        if volume >= ARCHIVE_KEEP_VOLUME or peaks.get(row["symbol"], 0.0) >= ARCHIVE_KEEP_VOLUME:
            kept.append(row)
    return kept


def admit(
    con: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    limit: int = NEW_SYMBOLS_PER_RUN,
) -> tuple[list[dict[str, Any]], int]:
    """Знакомые символы плюс не более ``limit`` новых.

    Возвращает пару «кого берём в этот прогон» и «сколько осталось в очереди».
    Новые берутся по обороту, потому что ``rows`` уже отсортирован по нему:
    крупная монета из очереди дождётся своего часа раньше мелкой.

    Знакомым считается символ, попавший в таблицу `symbols`, — туда его пишет
    снимок универсума при первом появлении в архивном слое.
    """
    known = storage.known_symbols(con, FUTURES.name)
    familiar = [row for row in rows if row["symbol"] in known]
    fresh = [row for row in rows if row["symbol"] not in known]
    return familiar + fresh[:limit], max(0, len(fresh) - limit)


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
        cleanup_order_book_watches()
        rows = await universe_rows(client)
        core = core_symbols(rows)
        known = storage.known_symbols(con, FUTURES.name)
        peaks = storage.recent_universe_volume(con, days=ARCHIVE_KEEP_DAYS - 1)
        archived, waiting = admit(con, kept_rows(rows, known, peaks))
        if waiting:
            log.info(
                "новых символов в очереди: %d — впускаются по %d за прогон, "
                "история от этого не теряется",
                waiting, NEW_SYMBOLS_PER_RUN,
            )
        # Деривативы пишутся по всему архивному слою, а не по ядру. Замерено на
        # живой выдаче: из топ-15 по индексу восемь монет имели оборот ниже
        # порога ядра, то есть больше половины кандидатов в список наблюдения
        # оставались без открытого интереса — а компонент накопления считается
        # именно по нему. Полоса 10–50M и есть самая интересная для сжатий.
        # Цена вопроса — 0.6 ГБ в год; цена промедления — сутки истории за
        # каждые сутки, потому что /futures/data/ отдаёт только 30 суток.
        derivative_symbols = [row["symbol"] for row in archived]
        kind = "backfill" if backfill_days else "incremental"
        log.info(
            "%s: деривативы %d, ядро (полная лестница) %d, архив %d символов",
            kind, len(derivative_symbols), len(core), len(archived),
        )

        started = time.monotonic()
        written, failed = await collect(
            client, con, derivative_symbols, days=backfill_days
        )
        storage.record_run(
            con, kind, symbols=len(derivative_symbols), rows=written,
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

        # Фандинг после свечей и деривативов: он восстановим, как свечи, — биржа
        # отдаёт его за годы. Если прогон оборвётся здесь, потеряно ничего.
        started = time.monotonic()
        rates, rate_failures = await collect_funding(
            client, con, derivative_symbols, days=backfill_days
        )
        storage.record_run(
            con, "funding", symbols=len(derivative_symbols), rows=rates,
            seconds=time.monotonic() - started,
            error=("не собраны: " + ", ".join(rate_failures[:20]))
            if rate_failures else None,
        )
        con.commit()
        log.info(
            "фандинг: записано начислений %d, не собрано символов %d",
            rates, len(rate_failures),
        )

        # Скан и исходы считаются из базы, к бирже не ходят вовсе, поэтому
        # стоят копейки и идут последними — после того как архив пополнен.
        started = time.monotonic()
        scanned = run_scan(con)
        # Соседний рынок — сразу после скана и до списка наблюдения: величина
        # справочная, но попасть она должна на ту же строку журнала.
        twins = await scan_twin({SPOT.name: spot, FUTURES.name: client}, con)
        settled = settle_outcomes(con)
        changes = update_watchlist(con)
        accumulated = update_accumulation(con)
        storage.record_run(
            con, "scan", symbols=scanned, rows=scanned + settled,
            seconds=time.monotonic() - started,
        )
        con.commit()
        log.info(
            "скан: новых записей %d, соседний рынок посчитан у %d, исходов %d",
            scanned, twins, settled,
        )
        log.info(
            "watchlist: вошло %d, вышло %d, подтверждено %d",
            len(changes["entered"]), len(changes["exited"]), len(changes["promoted"]),
        )
        for entry in changes["entered"]:
            log.info(
                "  + %s %s (ранг %s, узк %s)",
                entry["symbol"], entry["tf"], entry["rank"], entry["narrow_bars"],
            )
        for entry in changes["exited"]:
            log.info(
                "  - %s %s (%s)", entry["symbol"], entry["tf"], entry["reason"],
            )

        # Только после commit: доставка не должна стоить данных. Молчащий
        # Telegram — потеря уведомления, молчащий сборщик — потеря часа
        # открытого интереса навсегда. notify_watchlist исключений не
        # поднимает и молчит, когда состав списка не изменился.
        log.info(
            "накопление: вошло %d, вышло %d",
            len(accumulated["entered"]), len(accumulated["exited"]),
        )
        if await notify_watchlist(changes):
            log.info("telegram: дельта отправлена")
        if await notify_accumulation(accumulated):
            log.info("telegram: дельта накопления отправлена")
    finally:
        await client.aclose()
        await spot.aclose()


def cleanup_order_book_watches() -> dict[str, int]:
    """Убрать временные данные стакана, не рискуя часовым прогоном (B.6, PLAN §4.38).

    Отдельный файл может ещё не существовать — это нормальный случай до
    первого start. Любая другая ошибка также только логируется: цена ошибки
    здесь — лишняя временная запись, а цена падения collector — потерянный час
    невосстановимых деривативов.
    """
    try:
        result = orderbook_watch.cleanup()
    except Exception:
        log.exception("уборка сессий стакана не выполнена")
        return {"expired": 0, "deleted": 0}
    if result["expired"] or result["deleted"]:
        log.info(
            "стакан: истекло сессий %d, удалено по retention %d",
            result["expired"], result["deleted"],
        )
    return result


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


def merge_manual(
    episodes: list[dict[str, Any]], manual: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Слить эпизоды сканера с ручными записями в один список.

    Монета, стоящая в обеих базах, показывается ОДНОЙ строкой: ранг и индекс
    берутся у сканера, заметка — у человека, источник печатается как
    «scanner+manual». Две строки на одну пару читались бы как два разных
    наблюдения, а это одно и то же (§4.28).

    Ручная запись без сканерного эпизода идёт своей строкой и без ранга. Ранга
    у неё нет не потому, что он потерян, а потому что её никто не ранжировал —
    и по рангу её не снимут.
    """
    by_key = {(row["symbol"], row["tf"]): row for row in episodes}
    merged = list(episodes)
    for entry in manual:
        key = (entry["symbol"], entry["tf"])
        twin = by_key.get(key)
        if twin is not None and not twin.get("exited_at"):
            twin["note"] = entry.get("note")
            twin["entered_by"] = "scanner+manual"
            continue
        merged.append({
            "symbol": entry["symbol"],
            "tf": entry["tf"],
            "status": entry["status"],
            "entered_at": entry["entered_at"],
            "entered_by": "manual",
            "exited_at": entry.get("removed_at"),
            "exit_reason": entry.get("remove_reason"),
            "price_at_entry": entry.get("price_at_entry"),
            "note": entry.get("note"),
            "rank_at_entry": None,
            "last_rank": None,
            "squeeze_index": None,
            "last_index": None,
            "accumulation_score": None,
        })
    return merged


def watchlist_view(
    con: sqlite3.Connection,
    *,
    status: str | None = None,
    tf: str | None = None,
    limit: int = 100,
    now_ms: int | None = None,
    manual_rows: list[dict[str, Any]] | None = None,
    prices: dict[str, float] | None = None,
) -> str:
    """Список наблюдения текстом: один путь для CLI сборщика и MCP-сервера.

    Общий рендер здесь не про экономию строк. Две функции с одинаковым смыслом
    в этом проекте уже трижды расходились в третьем знаке — в колонке объёма
    get_klines, между архивом и биржей, между сервером и сканером. Список
    наблюдения будет читаться и из терминала, и из чата, и расхождение между
    ними обнаружилось бы не сразу.
    """
    stamp = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    rows = merge_manual(
        storage.episodes(con, status=status, tf=tf, limit=limit),
        manual_rows or [],
    )
    scans = {
        (scan["symbol"], timeframe): scan
        for timeframe in {row["tf"] for row in rows}
        for scan in storage.latest_scan(con, timeframe)
    }
    timeframes = {row["tf"] for row in rows}
    earlier = sorted({
        version
        for timeframe in timeframes
        for version in storage.earlier_versions(con, timeframe)
    })
    return render.render_watchlist(
        rows[:limit], scans, now_ms=stamp, prices=prices,
        earlier=earlier, version=SQUEEZE_FORMULA_VERSION,
    )


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


async def _notify_test() -> bool:
    sender = Telegram.from_env()
    if sender is None:
        log.error("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID пусты")
        return False
    stamp = dt.datetime.now(dt.UTC).strftime("%d.%m %H:%M")
    if await sender.send(f"cryptomcp: проверка связи · {stamp} UTC"):
        log.info("отправлено")
        return True
    log.error("не отправлено — причина выше")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборщик рыночных данных")
    parser.add_argument(
        "command",
        choices=("once", "backfill", "loop", "health", "watch", "add", "notify-test"),
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
            # Ручные записи читаются ТОЛЬКО на чтение и только если файл уже
            # есть: пишет в него сервер, и создавать его здесь нельзя. Без
            # этого чтения терминал и чат показывали бы разные списки — то
            # самое расхождение двух путей, из-за которого рендер общий.
            manual_con = manual.read_only()
            try:
                print(watchlist_view(
                    con,
                    manual_rows=manual.entries(
                        manual_con,
                        now_ms=int(dt.datetime.now(dt.UTC).timestamp() * 1000),
                    ),
                ))
            finally:
                if manual_con is not None:
                    manual_con.close()
            return
        if args.command == "notify-test":
            # Проверка связки «токен + chat id + бот в группе». Без неё
            # настройку не подтвердить иначе как дождавшись смены состава
            # списка, а она может не случиться сутками.
            raise SystemExit(0 if asyncio.run(_notify_test()) else 1)
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
