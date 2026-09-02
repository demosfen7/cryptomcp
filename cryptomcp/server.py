"""MCP-сервер: семь инструментов, три уровня выдачи (PLAN §5).

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
import sys

from mcp.server.mcpserver import MCPServer

from .analysis import MIN_CANDLES, TimeframeView, analyse_timeframe, weekly_pivots_from
from .client import BinanceClient
from .config import Config
from .derivatives import DerivativesReader
from .errors import ToolError, bad_params
from .fetcher import CandleFetcher
from .indicators import MIN_PERCENTILE_SPAN_DAYS
from .journal import Journal
from .markets import MARKETS, Market
from .render import (
    closed_through,
    render_derivatives,
    render_klines,
    render_levels,
    render_pivots,
    render_snapshot,
    render_squeeze_metrics,
)
from .series import INTERVAL_MS
from .symbols import SymbolRegistry, format_price

#: Максимум сырых свечей на запрос: третий уровень предназначен для чтения
#: формы, а не для выгрузки истории (PLAN §5).
MAX_RAW_KLINES = 50

#: Таймфреймы, на которых пагинация не окупается: 60 суток требуют 58 страниц.
NO_PAGINATION = {"1m"}

config = Config.load()
journal = Journal(config.journal_path)

server = MCPServer(
    name="cryptomcp",
    version="0.1.0",
    instructions=(
        "Рыночный контекст Binance USDⓈ-M Futures для анализа.\n\n"
        "Три уровня, идти сверху вниз и останавливаться, как только хватит:\n"
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
) -> tuple[BinanceClient, CandleFetcher, SymbolRegistry, DerivativesReader | None, Market]:
    market = _market(market_name)
    async with _lock:
        client = _clients.get(market.name)
        if client is None:
            client = _clients[market.name] = BinanceClient(market)
    derivatives = DerivativesReader(client) if market.has_derivatives else None
    return client, CandleFetcher(client), SymbolRegistry(client), derivatives, market


def _fail(error: ToolError) -> str:
    return json.dumps(error.to_payload(), ensure_ascii=False, indent=2)


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


def _target_span(interval: str) -> float | None:
    """Сколько суток истории догружать ради перцентилей (PLAN §4.2)."""
    return None if interval in NO_PAGINATION else MIN_PERCENTILE_SPAN_DAYS


async def _views(
    fetcher: CandleFetcher,
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
        "1w/1d/4h/1h/15m; 5m и 1m доступны, но запрашиваются явно. "
        "Параметр market: futures (перпетуал, по умолчанию) или spot. Спот "
        "нужен, когда перпетуал тонкий: подтверждение пробоя объёмом честнее "
        "искать там, где происходит поставка. Деривативов у спота нет."
    )
)
async def get_market_snapshot(
    symbol: str,
    timeframes: list[str] | None = None,
    as_of_ms: int | None = None,
    market: str = "futures",
) -> str:
    try:
        client, fetcher, registry, derivatives, mkt = await _ctx(market)
        info = await registry.get(symbol)
        intervals = _validate_timeframes(timeframes)

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
        return (
            f"{info.symbol}{mkt.suffix} ({mkt.label})\n\n"
            + render_squeeze_metrics(view)
        )
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Уровень 3. Сырые ЗАКРЫТЫЕ свечи (не более 50) с производными по "
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
        if not 1 <= limit <= MAX_RAW_KLINES:
            raise bad_params(
                f"limit={limit} вне диапазона 1..{MAX_RAW_KLINES}", limit=limit
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
        "закрытие шортов, набор позиций без движения цены и так далее."
    )
)
async def get_derivatives(symbol: str) -> str:
    try:
        # Фандинга и открытого интереса у спота не существует: инструмент
        # всегда работает по фьючерсам, параметра market у него нет.
        _, _, registry, derivatives, _ = await _ctx("futures")
        info = await registry.get(symbol)
        funding = await derivatives.funding(info.symbol)  # type: ignore[union-attr]
        oi = await derivatives.open_interest(info.symbol)  # type: ignore[union-attr]
        return f"{info.symbol}\n\n" + render_derivatives(
            funding, oi, history=True, precision=info.price_precision
        )
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Пакетное сканирование списка пар по squeeze_index на одном "
        "таймфрейме, по строке на монету. Пагинация истории отключена ради "
        "веса запросов, поэтому на младших таймфреймах часть перцентилей будет "
        "недоступна — для разбора конкретной пары вызывать get_squeeze_metrics. "
        "Параметр market: futures (по умолчанию) или spot."
    )
)
async def scan_pairs(
    symbols: list[str], timeframe: str = "4h", market: str = "futures"
) -> str:
    try:
        _, fetcher, registry, _, mkt = await _ctx(market)
        interval = _validate_timeframes([timeframe])[0]
        if not symbols:
            raise bad_params("Список symbols пуст")
        if len(symbols) > 100:
            raise bad_params(f"За раз не более 100 пар, передано {len(symbols)}")

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
                bbw = (
                    f"{view.bbw.pct_rank:>3.0f}" if view.bbw.has_context else "n/a"
                )
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
            f"{interval} · {mkt.label} · сортировка по squeeze_index",
            f"{'символ':<14}{'индекс':>6}{'BBW':>7}{'диап':>9}"
            f"{'узк':>6}{'объём':>9}  EMA/структура",
        ]
        lines += [text for _, text in rows]
        lines += [
            "",
            "узк — свечей подряд с шириной диапазона(20) ниже 20-го перцентиля "
            "своей истории",
            "объём — к сезонной базе, но история здесь мельче снапшотной "
            "(пагинация отключена ради веса), поэтому число приблизительное",
        ]
        if problems:
            lines += ["", "пропущены — остальные посчитаны:"] + [f"  {p}" for p in problems]
        return "\n".join(lines)
    except ToolError as error:
        return _fail(error)


@server.tool(
    description=(
        "Список торгуемых пар к USDT по обороту за сутки, со стейблкоин-парами, "
        "исключёнными из выдачи. Нужен, чтобы получить список для scan_pairs. "
        "Параметр market: futures (только бессрочные контракты, по умолчанию) "
        "или spot."
    )
)
async def list_symbols(
    min_quote_volume_usdt: float = 50_000_000,
    limit: int = 50,
    market: str = "futures",
) -> str:
    try:
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
