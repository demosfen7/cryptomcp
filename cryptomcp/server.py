"""MCP-сервер: девять инструментов, три уровня выдачи (PLAN §5).

Транспорт задаётся переменной CRYPTOMCP_TRANSPORT: stdio для локальной работы,
streamable-http для сервера за Caddy. В MCP 2.x транспорт передаётся в run(),
а не в конструктор.

Все инструменты возвращают текст. Ошибки — структурированным payload, а не
исключением: сырой traceback модель не поймёт и, скорее всего, зациклится на
повторах (PLAN §6.5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys

from mcp.server.mcpserver import MCPServer

from . import SQUEEZE_FORMULA_VERSION, storage
from .analysis import (
    MIN_CANDLES,
    TimeframeView,
    accumulation_interval,
    analyse_timeframe,
    weekly_pivots_from,
)
from .client import BinanceClient
from .collector import watchlist_view
from .config import Config
from .derivatives import DerivativesReader
from .errors import ErrorKind, ToolError, bad_params
from .fetcher import CandleFetcher
from .indicators import MIN_PERCENTILE_SPAN_DAYS, atr
from .journal import Journal
from .markets import MARKETS, Market
from .reader import ArchiveReader, archive_path
from .render import (
    closed_through,
    render_absorption,
    render_derivatives,
    render_klines,
    render_levels,
    render_pivots,
    render_scan_history,
    render_screen,
    render_snapshot,
    render_squeeze_metrics,
    skip_label,
    utc,
)
from .series import INTERVAL_MS
from .symbols import SymbolRegistry, format_price
from .volume import absorption

#: Максимум сырых свечей на запрос: третий уровень предназначен для чтения
#: формы, а не для выгрузки истории (PLAN §5).
#:
#: На младших ТФ потолок выше, и это не послабление, а следствие того же
#: правила. Пятьдесят часовых свечей — двое суток, то есть меньше, чем длится
#: фаза поглощения; разбор ASTER 19.08 потребовал четырёх вызовов с ручным
#: пересчётом as_of_ms. Пятьдесят дневных — это уже два месяца, и читать по ним
#: форму бессмысленно, поэтому там потолок остаётся прежним.
MAX_RAW_KLINES = 50
MAX_RAW_KLINES_INTRADAY = 200

#: Граница «младшего»: до 1h включительно.
RAW_INTRADAY_CEILING = "1h"

#: Таймфреймы, на которых пагинация не окупается: 60 суток требуют 58 страниц.
NO_PAGINATION = {"1m"}

config = Config.load()
journal = Journal(config.journal_path)

server = MCPServer(
    name="cryptomcp",
    version="0.1.0",
    instructions=(
        "Рыночный контекст Binance USDⓈ-M Futures для анализа.\n\n"
        "Начинать с get_watchlist. Фоновый сканер каждый час проходит весь "
        "ликвидный универсум, считает индекс сжатия и ведёт список отобранных "
        "пар с историей входов и выходов; get_scan_history показывает, как "
        "индекс и его группы менялись по конкретной паре. Эта работа уже "
        "сделана — перебирать пары руками через list_symbols и scan_pairs "
        "стоит только тогда, когда нужен свой срез universe.\n\n"
        "Три уровня разбора, идти сверху вниз и останавливаться, как только "
        "хватит:\n"
        "1) get_market_snapshot — общая картина по лестнице таймфреймов;\n"
        "2) get_squeeze_metrics — пять групп признаков с базой сравнения;\n"
        "3) get_klines — сырые свечи, когда нужно увидеть ФОРМУ "
        "(последовательность экстремумов, отбои от уровня), которую агрегаты "
        "не показывают.\n\n"
        "Каждый инструмент работает и по фьючерсам (market=\"futures\", по "
        "умолчанию), и по споту (market=\"spot\"). Перпетуал ценово производен "
        "от спота: когда открытый интерес мал и сигналы шумные, подтверждение "
        "объёмом стоит проверять на споте. Деривативов у спота нет, и списки "
        "символов у рынков не совпадают в обе стороны.\n\n"
        "Все метрики считаются по ЗАКРЫТЫМ свечам. Числа сопровождаются базой "
        "сравнения; где базы не хватило, честно стоит n/a — это не сбой.\n"
        "Инструменты не предсказывают направление и не дают рекомендаций: "
        "squeeze_index измеряет готовность к движению, а не его сторону."
    ),
)

#: Клиент на рынок: у фьючерсов и спота разные хосты и раздельные пулы веса,
#: поэтому и бюджеты должны быть разными объектами.
_clients: dict[str, BinanceClient] = {}
_lock = asyncio.Lock()


def _market(name: str) -> Market:
    market = MARKETS.get(name)
    if market is None:
        raise bad_params(
            f"Неизвестный рынок {name!r}. Доступны: {', '.join(MARKETS)}",
            market=name,
        )
    return market


async def _ctx(
    market_name: str = "futures",
) -> tuple[BinanceClient, ArchiveReader, SymbolRegistry, DerivativesReader | None, Market]:
    market = _market(market_name)
    async with _lock:
        client = _clients.get(market.name)
        if client is None:
            client = _clients[market.name] = BinanceClient(market)
    derivatives = DerivativesReader(client) if market.has_derivatives else None
    # Ряды идут из архива с догрузкой хвоста; где архива нет — прямо из биржи.
    # Контракт get() у обёртки тот же, поэтому инструменты о ней не знают.
    reader = ArchiveReader(CandleFetcher(client), market.name)
    return client, reader, SymbolRegistry(client), derivatives, market


def _fail(error: ToolError) -> str:
    return json.dumps(error.to_payload(), ensure_ascii=False, indent=2)


def _merged(values: list[str] | None, single: str | None) -> list[str]:
    """Слить список и то же значение в единственном числе.

    Три инструмента принимают список там, где у соседних инструментов стоит
    одиночное значение (symbols/symbol, timeframes/timeframe). Вызывающий
    пишет привычное единственное число, а MCP-сервер лишний ключ отбрасывает
    молча: модель аргументов собирается через create_model без
    extra="forbid", то есть с умолчанием pydantic extra="ignore". Ошибки не
    возникает — инструмент отвечает по умолчанию, и подмена незаметна.
    Поэтому принимаются оба написания; порядок сохраняется, дубли отсеиваются.
    """
    merged = list(values) if values else []
    if single is not None and single not in merged:
        merged.append(single)
    return merged


def _validate_timeframes(values: list[str] | None) -> tuple[str, ...]:
    timeframes = tuple(values) if values else config.timeframes
    unknown = [tf for tf in timeframes if tf not in INTERVAL_MS]
    if unknown:
        raise bad_params(
            f"Неизвестные таймфреймы: {', '.join(unknown)}. "
            f"Доступны: {', '.join(INTERVAL_MS)}",
            unknown=unknown,
        )
    return timeframes


def max_raw_klines(interval: str) -> int:
    """Потолок сырых свечей для этого таймфрейма."""
    return (
        MAX_RAW_KLINES_INTRADAY
        if INTERVAL_MS[interval] <= INTERVAL_MS[RAW_INTRADAY_CEILING]
        else MAX_RAW_KLINES
    )


def _target_span(interval: str) -> float | None:
    """Сколько суток истории догружать ради перцентилей (PLAN §4.2)."""
    return None if interval in NO_PAGINATION else MIN_PERCENTILE_SPAN_DAYS


async def _views(
    fetcher: ArchiveReader,
    symbol: str,
    timeframes: tuple[str, ...],
    as_of_ms: int | None,
    *,
    market: str = "futures",
    paginate: bool = True,
) -> tuple[dict[str, TimeframeView], dict[str, ToolError]]:
    """Разбор по таймфреймам, устойчивый к нехватке истории на отдельном ТФ.

    Свежий листинг — не краевой случай, а целевая категория: высокая
    волатильность, тонкая ликвидность, именно там сканер и интересен. Ронять
    весь снапшот из-за того, что недельных свечей набралось 43 из 60, значит
    не работать ровно там, где инструмент нужнее всего.

    Поэтому недоступный ТФ помечается строкой, а остальные считаются — та же
    логика, по которой метрика без достаточной базы печатает причину, а не
    отменяет всю выдачу (PLAN §4.2). Ошибка возвращается целиком только когда
    не посчитан ни один ТФ: тогда возвращать действительно нечего.
    """
    weekly = await fetcher.get(symbol, "1w", limit=10, as_of_ms=as_of_ms)
    pivots = weekly_pivots_from(weekly)

    views: dict[str, TimeframeView] = {}
    skipped: dict[str, ToolError] = {}
    for interval in timeframes:
        try:
            series = await fetcher.get(
                symbol,
                interval,
                limit=500,
                as_of_ms=as_of_ms,
                min_candles=MIN_CANDLES,
                target_span_days=_target_span(interval) if paginate else None,
            )
            view = analyse_timeframe(series, config, weekly_pivots=pivots)
        except ToolError as error:
            skipped[interval] = error
            continue
        journal.record(symbol, view, market=market, as_of_ms=as_of_ms)
        views[interval] = view

    if not views:
        # Единственный запрошенный ТФ не посчитался — это и есть ответ.
        raise next(iter(skipped.values()))
    return views, skipped


@server.tool(
    description=(
        "Уровень 1. Общая картина по паре: тренд и структура на лестнице "
        "таймфреймов, положение в диапазоне, RSI, ATR, объём с поправкой на "
        "время суток, ближайшие уровни и недельные пивоты, фандинг и открытый "
        "интерес, squeeze_index. Начинать разбор отсюда. По умолчанию "
        "1w/1d/4h/1h/15m; 5m и 1m доступны, но запрашиваются явно. Лестница "
        "задаётся как timeframes списком, либо timeframe одним значением. "
        "Параметр market: futures (перпетуал, по умолчанию) или spot. Спот "
        "нужен, когда перпетуал тонкий: подтверждение пробоя объёмом честнее "
        "искать там, где происходит поставка. Деривативов у спота нет."
    )
)
async def get_market_snapshot(
    symbol: str,
    timeframes: list[str] | None = None,
    timeframe: str | None = None,
    as_of_ms: int | None = None,
    market: str = "futures",
) -> str:
    try:
        client, fetcher, registry, derivatives, mkt = await _ctx(market)
        info = await registry.get(symbol)
        # Потерянный ключ возвращал всю лестницу по умолчанию вместо одного
        # запрошенного ТФ.
        intervals = _validate_timeframes(_merged(timeframes, timeframe) or None)

        views, skipped = await _views(
            fetcher, info.symbol, intervals, as_of_ms, market=mkt.name
        )
        ticker = await client.ticker_24hr(info.symbol)

        funding = oi = None
        if derivatives is not None and as_of_ms is None:
            # Деривативы существуют только «сейчас»: восстановить их состояние
            # на историческую дату биржа не даёт.
            funding = await derivatives.funding(info.symbol)
            oi = await derivatives.open_interest(info.symbol)

        live_price = (
            float(ticker["lastPrice"]) if as_of_ms is None
            else next(iter(views.values())).price
        )
        return render_snapshot(
            info,
            views,
            live_price=live_price,
            change_24h=float(ticker["priceChangePercent"]),
            quote_volume_24h=float(ticker["quoteVolume"]),
            now_ms=await client.now_ms(),
            funding=funding,
            open_interest=oi,
            as_of_ms=as_of_ms,
            skipped=skipped,
            order=intervals,
            market=mkt,
        )
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Уровень 2. Пять групп признаков сжатия по одному таймфрейму, каждое "
        "число с базой сравнения: волатильность (BBW, ATR), объём (сезонная "
        "поправка, затухание, бары набора позиции, доля тейкер-покупок), "
        "диапазон и его длительность, объёмный профиль, дивергенции RSI. "
        "Вызывать, когда снапшот показал, куда смотреть. Параметр market: "
        "futures (по умолчанию) или spot."
    )
)
async def get_squeeze_metrics(
    symbol: str,
    timeframe: str = "4h",
    as_of_ms: int | None = None,
    market: str = "futures",
) -> str:
    try:
        _, fetcher, registry, _, mkt = await _ctx(market)
        info = await registry.get(symbol)
        interval = _validate_timeframes([timeframe])[0]

        views, _ = await _views(
            fetcher, info.symbol, (interval,), as_of_ms, market=mkt.name
        )
        view = views[interval]
        absorption_block = await _absorption(
            fetcher, info.symbol, interval, as_of_ms
        )
        return (
            f"{info.symbol}{mkt.suffix} ({mkt.label})\n\n"
            + render_squeeze_metrics(view)
            + absorption_block
        )
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Уровень 3. Сырые ЗАКРЫТЫЕ свечи (до 200 на 1h и мельче, до 50 на "
        "старших ТФ) с производными по "
        "каждой: тело, фитили, объём к базе (та же, что в снапшоте: уровень "
        "20 предыдущих свечей с поправкой на слот суток — от limit не зависит), "
        "доля тейкер-покупок. Нужен, "
        "когда важна ФОРМА, которую агрегаты не передают: последовательность "
        "экстремумов, сужение подходов к уровню, характер отбоев. Это не "
        "запасной вариант, а полноправный третий уровень. Параметр market: "
        "futures (по умолчанию) или spot."
    )
)
async def get_klines(
    symbol: str,
    timeframe: str = "4h",
    limit: int = 30,
    as_of_ms: int | None = None,
    market: str = "futures",
) -> str:
    try:
        _, fetcher, registry, _, mkt = await _ctx(market)
        info = await registry.get(symbol)
        interval = _validate_timeframes([timeframe])[0]
        cap = max_raw_klines(interval)
        if not 1 <= limit <= cap:
            raise bad_params(
                f"limit={limit} вне диапазона 1..{cap} для {interval}", limit=limit
            )

        # 500 свечей, а не запрошенные limit: база объёма должна быть той же,
        # что в снапшоте, иначе одна и та же свеча снова получит два числа.
        series = await fetcher.get(
            info.symbol, interval, limit=max(limit, 500), as_of_ms=as_of_ms
        )
        return render_klines(series, info, limit, market=mkt)
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Уровни поддержки и сопротивления: кластеры swing-экстремумов с числом "
        "касаний, недельные и дневные пивоты, POC и Value Area. Все расстояния "
        "и в процентах, и в ATR. Когда цена стоит вплотную к уровню, "
        "показывается и следующий за ним. Параметр market: futures "
        "(по умолчанию) или spot."
    )
)
async def get_key_levels(
    symbol: str,
    timeframe: str = "4h",
    as_of_ms: int | None = None,
    market: str = "futures",
) -> str:
    try:
        _, fetcher, registry, _, mkt = await _ctx(market)
        info = await registry.get(symbol)
        interval = _validate_timeframes([timeframe])[0]

        views, _ = await _views(
            fetcher, info.symbol, (interval,), as_of_ms, market=mkt.name
        )
        view = views[interval]
        precision = info.price_precision

        lines = [
            f"{info.symbol}{mkt.suffix} ({mkt.label}) {interval} · "
            f"цена {format_price(view.price, precision)} "
            f"— закрытие свечи на {closed_through(view)} UTC, не live "
            f"(в снапшоте цена live, и числа расходятся)",
            f"ATR {format_price(view.atr_value, precision)} ({view.atr_pct:.2f}%)",
            f"найдено уровней: {len(view.levels)}",
            "",
        ]
        lines += render_levels(view.levels, view.price, view.atr_value, precision)
        if view.pivots_weekly:
            lines.append(render_pivots(view.pivots_weekly, view.price, view.atr_value, precision))
        if view.profile:
            inside = "ВНУТРИ" if view.profile.contains(view.price) else "вне"
            lines.append(
                f"  POC~ (прибл. из OHLCV) {format_price(view.profile.poc, precision)} · "
                f"Value Area~ {format_price(view.profile.value_area_low, precision)} – "
                f"{format_price(view.profile.value_area_high, precision)}, цена {inside}"
            )
        return "\n".join(lines)
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Фандинг и открытый интерес с рядами: почасовая динамика OI вместе с "
        "ценой за последние сутки и последние начисления фандинга. Три дельты "
        "показывают итог окна, ряд — момент, когда поток развернулся. "
        "Ставка приводится к годовым — без этого "
        "сравнение между монетами бессмысленно, интервал начисления у разных "
        "символов 1, 4 или 8 часов. Динамика OI считается по числу контрактов, "
        "а не по стоимости в USDT, и классифицируется: приток новых денег, "
        "закрытие шортов, набор позиций без движения цены и так далее. "
        "Параметр as_of_ms показывает состояние на момент в прошлом: ряды и "
        "дельты сдвигаются к нему. Ретроспектива читается из архива сборщика, "
        "а где он момент не покрывает — с биржи, у которой история открытого "
        "интереса обрезана 30 сутками; источник подписан в выдаче. Базиса в "
        "ретроспективе нет: маркировочная и индексная цены существуют только "
        "для текущего момента."
    )
)
async def get_derivatives(symbol: str, as_of_ms: int | None = None) -> str:
    try:
        # Фандинга и открытого интереса у спота не существует: инструмент
        # всегда работает по фьючерсам, параметра market у него нет.
        _, _, registry, derivatives, _ = await _ctx("futures")
        info = await registry.get(symbol)
        funding = await derivatives.funding(  # type: ignore[union-attr]
            info.symbol, as_of_ms=as_of_ms
        )
        oi = await derivatives.open_interest(  # type: ignore[union-attr]
            info.symbol, as_of_ms=as_of_ms
        )

        head = info.symbol
        if as_of_ms is not None:
            sources = " и ".join(sorted({funding.source, oi.source}))
            head += (
                f" · состояние на {utc(as_of_ms)} UTC · источник: {sources}"
                "\nбазис в ретроспективе недоступен: маркировочная и индексная "
                "цены существуют только для текущего момента"
            )
        return f"{head}\n\n" + render_derivatives(
            funding, oi, history=True, precision=info.price_precision
        )
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Отбор монет по сжатию и накоплению. Два режима. БЕЗ symbols — отбор "
        "по фильтру из журнала сканера: он каждый час проходит весь ликвидный "
        "универсум, поэтому выдача не стоит ни одного запроса к бирже и "
        "содержит признаки накопления (бары набора, тейкеры, лид объёма над "
        "ценой, сторона набора по фандингу). Доступные ТФ в этом режиме — "
        "4h, 1d, 1h; строка пишется на закрытие свечи, поэтому может отставать "
        "почти на таймфрейм, и «закрыта по» печатается в выдаче. С явными "
        "парами (symbols списком, либо symbol одним значением) — пересчёт по "
        "бирже, до 100 пар, любой ТФ: нужен для монет вне "
        "архива или когда важна свежесть, а не охват. Фильтры: оборот, возраст "
        "листинга, максимальное движение за сутки (монета в движении — не "
        "кандидат на накопление), длительность сжатия, исключения. Сортировка: "
        "squeeze, duration, accumulation. Несколько таймфреймов за вызов: "
        "timeframes списком, либо timeframe одним значением."
    )
)
async def scan_pairs(
    symbols: list[str] | None = None,
    symbol: str | None = None,
    timeframes: list[str] | None = None,
    timeframe: str | None = None,
    min_volume_usdt: float = 10_000_000,
    max_volume_usdt: float | None = None,
    min_age_days: int | None = None,
    max_abs_change_24h: float | None = None,
    exclude: list[str] | None = None,
    min_narrow_bars: int | None = None,
    sort_by: str = "squeeze",
    limit: int = 20,
    market: str = "futures",
) -> str:
    try:
        # Потерянный таймфрейм уводил выдачу на дефолтные 4h, причём в шапке
        # честно стояло 4h — признака подмены не было нигде.
        intervals = _validate_timeframes(_merged(timeframes, timeframe) or ["4h"])
        if sort_by not in storage.SCAN_SORTS:
            raise bad_params(
                f"Неизвестная сортировка {sort_by!r}. Доступны: "
                + ", ".join(storage.SCAN_SORTS),
                sort_by=sort_by,
            )
        # Здесь потеря ключа дороже: она не меняла ТФ, а тихо переключала
        # режим — вместо пересчёта по бирже по названной монете уходил полный
        # отбор по журналу.
        wanted = _merged(symbols, symbol)
        if wanted:
            return await _scan_explicit(wanted, intervals, market)
        return await _scan_screen(
            intervals,
            min_volume_usdt=min_volume_usdt,
            max_volume_usdt=max_volume_usdt,
            min_age_days=min_age_days,
            max_abs_change_24h=max_abs_change_24h,
            exclude={s.upper() for s in exclude} if exclude else None,
            min_narrow_bars=min_narrow_bars,
            sort_by=sort_by,
            limit=max(1, min(limit, 100)),
            market=market,
        )
    except ToolError as error:
        return _fail(error)


async def _scan_explicit(
    symbols: list[str], intervals: tuple[str, ...], market: str
) -> str:
    """Пересчёт по явному списку — как было до фильтров.

    Остаётся ради монет вне архива и ради случая, когда важна свежесть: этот
    путь считает по бирже и не отстаёт на свечу.
    """
    _, fetcher, registry, _, mkt = await _ctx(market)
    if len(symbols) > 100:
        raise bad_params(f"За раз не более 100 пар, передано {len(symbols)}")

    blocks: list[str] = []
    for interval in intervals:
        rows: list[tuple[float, str]] = []
        problems: list[str] = []
        for symbol in symbols:
            try:
                info = await registry.get(symbol)
                series = await fetcher.get(
                    info.symbol, interval, limit=500, min_candles=MIN_CANDLES
                )
                view = analyse_timeframe(series, config)
                journal.record(info.symbol, view, market=mkt.name)
                index = view.squeeze_index if view.squeeze_index is not None else -1.0
                bbw = f"{view.bbw.pct_rank:>3.0f}" if view.bbw.has_context else "n/a"
                rows.append((
                    index,
                    f"{info.symbol:<14}{index:>6.2f}{bbw:>7}"
                    f"{view.range_width * 100:>8.2f}%{view.narrow_bars:>6}"
                    f"{view.volume.ratio:>8.2f}x  {view.ema_state}/{view.structure}",
                ))
            except ToolError as error:
                problems.append(f"{symbol}: {error.message}")

        rows.sort(key=lambda item: -item[0])
        lines = [
            f"{interval} · {mkt.label} · пересчёт по бирже · "
            f"сортировка по squeeze_index",
            f"{'символ':<14}{'индекс':>6}{'BBW':>7}{'диап':>9}"
            f"{'узк':>6}{'объём':>9}  EMA/структура",
        ]
        lines += [text for _, text in rows]
        if problems:
            lines += ["", "пропущены — остальные посчитаны:"] + [
                f"  {p}" for p in problems
            ]
        blocks.append("\n".join(lines))

    blocks.append(
        "узк — свечей подряд с шириной диапазона(20) ниже 20-го перцентиля "
        "своей истории\nобъём — к сезонной базе; окно то же каноническое, что "
        "в снапшоте, поэтому числа сопоставимы напрямую"
    )
    return "\n\n".join(blocks)


async def _scan_screen(
    intervals: tuple[str, ...],
    *,
    min_volume_usdt: float,
    max_volume_usdt: float | None,
    min_age_days: int | None,
    max_abs_change_24h: float | None,
    exclude: set[str] | None,
    min_narrow_bars: int | None,
    sort_by: str,
    limit: int,
    market: str,
) -> str:
    """Отбор по фильтру из журнала сканера.

    Один запрос к бирже на весь отбор — тикер по всем символам сразу, ради
    оборота и суточного движения. Всё остальное берётся из базы: пересчёт
    стоил бы запроса за хвостом на КАЖДЫЙ символ (§4.15).
    """
    client, _, registry, _, mkt = await _ctx(market)
    tradable = {info.symbol for info in await registry.tradable()}
    tickers = await client.ticker_24hr()

    allowed: set[str] = set()
    for row in tickers:
        symbol = row["symbol"]
        if symbol not in tradable:
            continue
        volume = float(row["quoteVolume"])
        if volume < min_volume_usdt:
            continue
        if max_volume_usdt is not None and volume > max_volume_usdt:
            continue
        # Монета в движении — не кандидат на накопление. Фильтр выключен по
        # умолчанию: он же прячет монету, которая только что выстрелила.
        if (
            max_abs_change_24h is not None
            and abs(float(row["priceChangePercent"])) > max_abs_change_24h
        ):
            continue
        allowed.add(symbol)

    con = _archive()
    try:
        if min_age_days is not None:
            now_ms = await client.now_ms()
            edge = now_ms - min_age_days * 86_400_000
            listed = storage.listing_dates(con, mkt.name)
            allowed = {
                s for s in allowed
                if s in listed and listed[s] <= edge
            }
        passed = len(allowed)
        blocks = []
        for interval in intervals:
            matched = storage.screen_scan(
                con, interval, allowed=allowed, exclude=exclude,
                min_narrow_bars=min_narrow_bars, sort_by=sort_by,
            )
            logged = len(storage.latest_scan(con, interval))
            blocks.append(render_screen(
                interval, matched[:limit], version=SQUEEZE_FORMULA_VERSION,
                sort_by=sort_by, filtered=passed, logged=logged,
                matched=len(matched),
            ))
    finally:
        con.close()
    return "\n\n".join(blocks)


@server.tool(
    description=(
        "Список торгуемых пар к USDT по обороту за сутки, со стейблкоин-парами, "
        "исключёнными из выдачи. Нужен, чтобы получить список для scan_pairs. "
        "Порог оборота: min_quote_volume_usdt, либо min_volume_usdt — как в "
        "scan_pairs. Параметр market: futures (только бессрочные контракты, "
        "по умолчанию) или spot."
    )
)
async def list_symbols(
    min_quote_volume_usdt: float = 50_000_000,
    min_volume_usdt: float | None = None,
    limit: int = 50,
    market: str = "futures",
) -> str:
    try:
        # Тот же порог оборота в scan_pairs зовётся min_volume_usdt. Разные
        # имена одной величины — та же тихая потеря ключа: планка молча
        # оставалась дефолтной, а в шапке печаталось «≥ 50M», из-за чего
        # выдача выглядела правдоподобно и при потерянном фильтре.
        if min_volume_usdt is not None:
            min_quote_volume_usdt = min_volume_usdt
        client, _, registry, _, mkt = await _ctx(market)
        tradable = {info.symbol for info in await registry.tradable()}
        tickers = await client.ticker_24hr()

        rows = [
            (float(row["quoteVolume"]), row["symbol"], float(row["priceChangePercent"]))
            for row in tickers
            if row["symbol"] in tradable
            and float(row["quoteVolume"]) >= min_quote_volume_usdt
        ]
        rows.sort(reverse=True)
        rows = rows[:limit]

        kind = "перпетуалов" if mkt.has_derivatives else "спотовых пар"
        lines = [
            f"{mkt.label} · торгуемых {kind} с оборотом "
            f"≥ {min_quote_volume_usdt / 1e6:.0f}M USDT: {len(rows)}",
            f"{'символ':<14}{'оборот 24ч':>14}{'24ч %':>9}",
        ]
        lines += [
            f"{symbol:<14}{volume / 1e6:>12,.0f}M{change:>9.2f}"
            for volume, symbol, change in rows
        ]
        return "\n".join(lines)
    except ToolError as error:
        return _fail(error)


async def _absorption(
    fetcher: ArchiveReader, symbol: str, interval: str, as_of_ms: int | None
) -> str:
    """Блок поглощения по младшему ряду.

    Отдельный запрос ряда, и он оправдан: на дневном разрешении поглощение
    внутри свечи не видно вовсе (§4.19), то есть без него главный признак
    накопления просто отсутствует. Нехватка истории на младшем ТФ — не повод
    отменить всю выдачу: печатается причина, как и у любой метрики без базы.
    """
    lower = accumulation_interval(interval)
    if lower == interval:
        return ""
    try:
        series = await fetcher.get(
            symbol, lower, as_of_ms=as_of_ms, min_candles=MIN_CANDLES,
            target_span_days=_target_span(lower),
        )
    except ToolError as error:
        return render_absorption(None, skipped=skip_label(error))

    atr_values = atr(series.high, series.low, series.close, 14)
    return render_absorption(absorption(series, atr_values))


def _archive() -> sqlite3.Connection:
    """Соединение с архивом только на чтение.

    Сервер и сборщик монтируют один том, поэтому список наблюдения и журнал
    скана доступны серверу без единого запроса к бирже. Открывается на чтение
    не из осторожности, а по правилу схемы: писатель в базе один — сборщик.
    """
    path = archive_path()
    if path is None:
        raise ToolError(
            ErrorKind.DATA_GAP,
            "Архив недоступен: сборщик рядом не запущен. Список наблюдения и "
            "журнал скана ведёт он, из биржи их взять неоткуда.",
        )
    try:
        return storage.connect(path, read_only=True)
    except sqlite3.Error as error:
        raise ToolError(ErrorKind.DATA_GAP, f"Архив не открылся: {error}") from error


@server.tool(
    description=(
        "Список наблюдения, который ведёт фоновый сканер: какая пара и на "
        "каком таймфрейме отобрана, ранг при входе и сейчас, индекс и его "
        "изменение с входа, длительность сжатия, цена входа и ход от неё, "
        "чем кончился закрытый эпизод. Смотреть ПЕРВЫМ: сканер каждый час "
        "проходит весь универсум и уже отобрал кандидатов — перебирать пары "
        "руками через list_symbols и scan_pairs для этого не нужно. "
        "status: candidate, active, broken_out, expired, closed, all; по "
        "умолчанию только открытые эпизоды. Отбор идёт рангом индекса с "
        "гистерезисом (вход в топ-15, выход из топ-40), а не порогом."
    )
)
async def get_watchlist(
    status: str | None = None,
    timeframe: str | None = None,
    limit: int = 100,
) -> str:
    try:
        if status is not None and status not in storage.EPISODE_STATUSES:
            raise bad_params(
                f"Неизвестный статус {status!r}. Доступны: "
                + ", ".join(storage.EPISODE_STATUSES),
                status=status,
            )
        if timeframe is not None:
            timeframe = _validate_timeframes([timeframe])[0]
        con = _archive()
        try:
            return watchlist_view(
                con, status=status, tf=timeframe, limit=max(1, min(limit, 500))
            )
        finally:
            con.close()
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "История сканирования одной пары: как менялся squeeze_index и каждая "
        "из пяти групп от свечи к свече. Отвечает на вопрос, который по одному "
        "снимку не решается — вошла в сжатие вчера или сжимается третью неделю "
        "и признак усиливается. Запись одна на закрытую свечу; сравниваются "
        "только записи текущей версии формулы, потому что индексы разных "
        "версий между собой несопоставимы."
    )
)
async def get_scan_history(symbol: str, timeframe: str = "4h", limit: int = 20) -> str:
    try:
        interval = _validate_timeframes([timeframe])[0]
        con = _archive()
        try:
            rows = storage.scan_history(
                con, symbol.upper(), interval, limit=max(1, min(limit, 200))
            )
        finally:
            con.close()
        return render_scan_history(
            symbol.upper(), interval, rows, SQUEEZE_FORMULA_VERSION
        )
    except ToolError as error:
        return _fail(error)


def _configure_logging() -> None:
    """Увести весь лог в stderr и приглушить httpx.

    В stdio-транспорте сервер пишет JSON-RPC в stdout, и любая посторонняя
    строка рвёт протокол. httpx по умолчанию логирует каждый запрос — при
    снапшоте это полтора десятка строк, которые сделали бы поток нечитаемым
    для клиента.
    """
    logging.basicConfig(
        level=os.environ.get("CRYPTOMCP_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    _configure_logging()
    transport = os.environ.get("CRYPTOMCP_TRANSPORT", "stdio")

    if transport == "stdio":
        server.run(transport="stdio")
        return

    # Удалённый режим: своё Starlette-приложение вместо server.run(), чтобы
    # навесить проверку токена и health, не трогая общий Caddy (PLAN §7.3).
    import uvicorn

    from .app import build_app

    token = os.environ.get("MCP_AUTH_TOKEN")
    uvicorn.run(
        build_app(server, token=token),
        host=os.environ.get("CRYPTOMCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("CRYPTOMCP_PORT", "8000")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
