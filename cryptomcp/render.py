"""Текстовый рендер для модели (PLAN §4.2, §4.3, §5).

Два правила, которые здесь соблюдаются жёстко:

1. Ни одно число не печатается без базы сравнения. Метрика без перцентиля
   печатает причину, а не голое значение.
2. Расстояния идут и в процентах, и в ATR. «Сопротивление в 1.31%» на BTC при
   ATR 1.2% означает вплотную, а на альте с ATR 9% — шум внутри свечи.

Вывода здесь нет. Строка «согласованность ТФ» перечисляет факты по
таймфреймам; формулировки вида «сетап на продолжение» — работа модели.
"""

from __future__ import annotations

import datetime as dt

from .analysis import TimeframeView
from .derivatives import Funding, OpenInterest
from .errors import ErrorKind, ToolError
from .indicators import Metric
from .levels import Level, Pivots
from .markets import FUTURES, Market
from .series import Series
from .symbols import SymbolInfo, format_price
from .volume import MIN_SAMPLES_PER_SLOT, baseline_series

#: Ближе этого расстояния уровень считается «под ценой», и показывается
#: следующий за ним — иначе видно, что цена на уровне, но не видно, куда ход.
NEAR_LEVEL_ATR = 0.25


def utc(ms: int) -> str:
    if not ms:
        return "n/a"
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


def closed_through(view: TimeframeView, *, short: bool = False) -> str:
    """Момент, по который у таймфрейма есть закрытые свечи.

    Печатается граница, а не время закрытия последней свечи: «2026-09-02 00:00»
    вместо «2026-09-01 23:59:59». Без этой строки было не понять, каким
    закрытием заканчивается каждая строка лестницы, а недельная метрика
    выглядела так же «свежо», как пятнадцатиминутная.
    """
    ms = view.meta.get("closed_through_ms")
    if not ms:
        return "n/a"
    stamp = utc(int(ms) + 1)
    return stamp[5:16] if short else stamp[:16]


def skip_label(error: ToolError) -> str:
    """Короткая причина, по которой строка таймфрейма не посчиталась."""
    if error.kind is ErrorKind.INSUFFICIENT_HISTORY and "have" in error.details:
        return f"недостаточно истории ({error.details['have']}/{error.details['need']})"
    return error.message


def render_metric(metric: Metric, *, precision: int = 4) -> str:
    """Значение с базой сравнения — или с причиной её отсутствия."""
    value = f"{metric.value:.{precision}f}{metric.unit}"
    if not metric.has_context:
        return f"{value}  → {metric.base_note or 'нет базы для сравнения'}"
    flag = "  ⚑" if metric.flagged else ""
    tail = f"{metric.pct_rank:.0f}-й перцентиль из {metric.n_obs}"
    if metric.base_note:
        tail += f", {metric.base_note}"
    return f"{value}  → {tail}{flag}"


def _distance(target: float, price: float, atr_value: float, precision: int) -> str:
    pct = (target - price) / price * 100.0 if price else float("nan")
    atr_units = (target - price) / atr_value if atr_value else float("nan")
    return f"{format_price(target, precision)} ({pct:+.2f}%, {atr_units:+.1f} ATR)"


def render_levels(
    levels: list[Level], price: float, atr_value: float, precision: int
) -> list[str]:
    """Ближайшие уровни, а при касании — и следующий за ним.

    Когда цена стоит вплотную к уровню, одно только «поддержка −0.0 ATR»
    сообщает, что цена на уровне, но не сообщает, куда есть ход. Печатаются оба.
    """
    below = sorted((lv for lv in levels if lv.price <= price), key=lambda lv: -lv.price)
    above = sorted((lv for lv in levels if lv.price > price), key=lambda lv: lv.price)

    lines: list[str] = []
    for label, group in (("поддержка", below), ("сопротивление", above)):
        if not group:
            continue
        first = group[0]
        touching = abs(first.price - price) < NEAR_LEVEL_ATR * atr_value
        chosen = group[:2] if touching else group[:1]
        for i, level in enumerate(chosen):
            name = label if i == 0 else f"{label}, следующая"
            note = " · цена НА уровне" if touching and i == 0 else ""
            lines.append(
                f"  {name:<22} {_distance(level.price, price, atr_value, precision)}"
                f"  {level.touches} касаний{note}"
            )
    return lines


def render_pivots(pv: Pivots, price: float, atr_value: float, precision: int) -> str:
    below, above = pv.nearest(price)
    parts = [
        f"{name} {_distance(value, price, atr_value, precision)}"
        for name, value in (x for x in (below, above) if x)
    ]
    return "  пивоты (нед.)          " + " · ".join(parts) if parts else ""


def render_snapshot(
    info: SymbolInfo,
    views: dict[str, TimeframeView],
    *,
    live_price: float,
    change_24h: float,
    quote_volume_24h: float,
    now_ms: int,
    funding: Funding | None = None,
    open_interest: OpenInterest | None = None,
    as_of_ms: int | None = None,
    skipped: dict[str, ToolError] | None = None,
    order: tuple[str, ...] | None = None,
    market: Market = FUTURES,
) -> str:
    """Уровень L1: общая картина, 20–25 строк.

    ``skipped`` — таймфреймы, которые не удалось посчитать. Они печатаются
    строкой с причиной, а не исчезают из лестницы: пропавшая без объяснения
    строка читается как «на 1w сжатия нет», хотя означает «не считали».
    """
    precision = info.price_precision
    lines: list[str] = []

    head = (
        f"{info.symbol}{market.suffix} ({market.label}) · "
        f"цена {format_price(live_price, precision)}"
    )
    if as_of_ms is None:
        head += f" (live {utc(now_ms)} UTC)"
    else:
        head += f" (состояние на {utc(as_of_ms)} UTC)"
    lines.append(head)
    turnover = (
        f"{quote_volume_24h / 1e9:.2f}B" if quote_volume_24h >= 1e9
        else f"{quote_volume_24h / 1e6:.1f}M"
    )
    lines.append(
        f"24h {change_24h:+.2f}% · оборот {turnover} USDT · "
        f"метрики по ЗАКРЫТЫМ свечам"
    )
    lines.append("")
    lines.append(
        f"{'ТФ':<5}{'EMA':<8}{'структ':<9}{'поз':>5}{'RSI':>6}{'ATR%':>7}"
        f"{'объём':>8}{'takerB':>8}  сжатие"
    )

    skipped = skipped or {}
    ladder = order or tuple(views) + tuple(k for k in skipped if k not in views)
    for interval in ladder:
        view = views.get(interval)
        if view is None:
            error = skipped.get(interval)
            lines.append(f"{interval:<5}— {skip_label(error) if error else 'не считался'}")
            continue
        if view.bbw.has_context:
            squeeze = f"BBW {view.bbw.pct_rank:.0f} pct"
            if view.bbw.flagged:
                squeeze = f"ДА · {squeeze}"
        else:
            squeeze = view.bbw.base_note or "n/a"
        # Длительность сжатия — отдельный признак ТЗ §4.2, и критерий у неё свой
        # (порог ширины диапазона), а не перцентиль BBW. Раньше оба числа стояли
        # в одной фразе через запятую и читались как одно.
        squeeze += f" · узк {view.narrow_bars}"
        volume = f"{view.volume.ratio:.2f}x" + ("~" if view.volume.weak_basis else "")
        lines.append(
            f"{interval:<5}{view.ema_state:<8}{view.structure:<9}"
            f"{view.position_in_range:>5.2f}{view.rsi_value:>6.1f}"
            f"{view.atr_pct:>6.2f}%{volume:>8}{view.volume.taker_buy_mean:>8.2f}  {squeeze}"
        )

    lines.append("")
    if skipped:
        lines.append("недоступные ТФ — остальные посчитаны:")
        lines += [f"  {tf}: {error.message}" for tf, error in skipped.items()]
        lines.append("")
    lines.append("закрыты по (UTC): " + " · ".join(
        f"{tf} {closed_through(v, short=True)}" for tf, v in views.items()
    ))
    gaps = [
        f"{tf} {v.meta['missing']} св."
        for tf, v in views.items() if v.meta.get("missing")
    ]
    if gaps:
        lines.append("пропуски в истории: " + " · ".join(gaps) + " — метрики на них смещены")
    lines.append("объём — к уровню последних свечей с поправкой на слот суток; ~ слабая база")
    lines.append("EMA — цена против EMA50/EMA200; структ — два последних swing-экстремума")
    lines.append(
        "узк N — свечей подряд с шириной диапазона(20) ниже 20-го перцентиля "
        "СВОЕЙ истории"
    )

    anchor = views.get("4h") or next(iter(views.values()))
    lines.append("")
    lines.append(
        f"диапазон {anchor.interval} (20 св.): {format_price(anchor.range_low, precision)} – "
        f"{format_price(anchor.range_high, precision)} · "
        f"{anchor.range_width * 100:.2f}% = {anchor.range_width_atr:.1f} ATR · "
        + (
            f"{anchor.range_metric.pct_rank:.0f}-й перцентиль своей истории, "
            f"узким был {anchor.narrow_bars} св. подряд"
            if anchor.range_metric.has_context
            else (anchor.range_metric.base_note or "перцентиль n/a")
        )
    )
    lines.extend(render_levels(anchor.levels, live_price, anchor.atr_value, precision))
    if anchor.pivots_weekly:
        pivot_line = render_pivots(anchor.pivots_weekly, live_price, anchor.atr_value, precision)
        if pivot_line:
            lines.append(pivot_line)
    if anchor.profile:
        inside = "ВНУТРИ" if anchor.profile.contains(live_price) else "вне"
        lines.append(
            f"  POC~ (прибл. из OHLCV)  "
            f"{_distance(anchor.profile.poc, live_price, anchor.atr_value, precision)}"
            f"  · Value Area~ {format_price(anchor.profile.value_area_low, precision)}"
            f" – {format_price(anchor.profile.value_area_high, precision)}, цена {inside}"
        )

    if funding or open_interest:
        lines.append("")
        lines.append(render_derivatives(funding, open_interest, precision=precision))
    elif not market.has_derivatives:
        lines.append("")
        lines.append(
            "фандинга и открытого интереса у спота не существует — "
            "смотреть по фьючерсу (get_derivatives)"
        )

    lines.append("")
    lines.append("согласованность ТФ: " + " · ".join(
        f"{tf} {v.ema_state}, {v.structure}" for tf, v in views.items()
    ))
    indices = " · ".join(
        f"{tf} {v.squeeze_index:.2f}" for tf, v in views.items()
        if v.squeeze_index is not None
    )
    lines.append(f"squeeze_index: {indices or 'n/a'}")
    return "\n".join(lines)


def _metric_context(metric: Metric | None, unit: str = "сут.") -> str:
    """Хвост «, 65 pct за 83 сут.» — или причина, по которой его нет."""
    if metric is None:
        return ""
    if not metric.has_context:
        return f", {metric.base_note or 'нет базы для сравнения'}"
    text = f", {metric.pct_rank:.0f} pct за {metric.span_days:.0f} {unit}"
    if metric.base_note:
        text += f" ({metric.base_note})"
    return text


def render_derivatives(
    funding: Funding | None,
    open_interest: OpenInterest | None,
    *,
    history: bool = False,
    precision: int = 4,
) -> str:
    """Фандинг и открытый интерес.

    ``history`` включает ряды. В снапшоте их нет: он обязан оставаться на
    двух десятках строк. В get_derivatives есть, потому что три дельты
    сообщают итог окна, а ряд — момент, когда поток развернулся.
    """
    lines: list[str] = []
    if funding:
        basis = f" · базис {funding.basis_pct:+.3f}%" if funding.index_price else ""
        lines.append(
            f"фандинг {funding.rate * 100:+.4f}% / {funding.interval_hours}ч "
            f"= {funding.annualized_pct:+.2f}% годовых"
            f"{_metric_context(funding.percentile)}{basis}"
        )
        if history and funding.history:
            since = utc(funding.history[0][0])[:16]
            rates = " ".join(f"{rate * 100:+.4f}" for _, rate in funding.history)
            lines.append(f"  начисления, % за период, старые→новые (с {since} UTC):")
            lines.append(f"    {rates}")

    if open_interest:
        notional = open_interest.notional_usdt
        scale = f"{notional / 1e9:.2f}B" if notional >= 1e9 else f"{notional / 1e6:.0f}M"
        lines.append(
            f"OI {scale} USDT{_metric_context(open_interest.percentile)}:"
        )
        for window in open_interest.change:
            lines.append(
                f"  {window:>4}: OI {open_interest.change[window] * 100:+6.2f}% · "
                f"цена {open_interest.price_change[window] * 100:+6.2f}% "
                f"→ {open_interest.quadrant(window)}"
            )
        if history and open_interest.history:
            base_ms, base_oi, base_price = open_interest.history[0]
            lines.append(f"  по часам, Δ от начала окна ({utc(base_ms)[:16]} UTC):")
            lines.append(
                f"    {'время':<12}{'OI, контр.':>14}{'ΔOI%':>8}"
                f"{'цена':>11}{'Δцены%':>9}"
            )
            for stamp, contracts, price in open_interest.history:
                d_oi = (contracts / base_oi - 1) * 100 if base_oi else float("nan")
                d_price = (price / base_price - 1) * 100 if base_price else float("nan")
                lines.append(
                    f"    {utc(stamp)[5:16]:<12}{contracts:>14,.0f}{d_oi:>+8.2f}"
                    f"{format_price(price, precision):>11}{d_price:>+9.2f}"
                )
            lines.append(
                "    цена восстановлена из sumOpenInterestValue/sumOpenInterest — "
                "моменты замеров совпадают с OI точно"
            )
    return "\n".join(lines)


def render_squeeze_metrics(view: TimeframeView) -> str:
    """Уровень L2: пять групп признаков ТЗ §4.2 с базой сравнения."""
    volume = view.volume
    lines = [
        f"{view.interval} · свечи закрыты по {closed_through(view)} UTC",
        f"база перцентилей: {view.bbw.n_obs} наблюдений, "
        f"охват {view.bbw.span_days:.0f} сут.",
        "",
        "1. Волатильность",
        f"   BBW(20,2)        {render_metric(view.bbw)}",
        f"   ATR/price        {render_metric(view.atr_metric, precision=2)}",
        f"   тренд ATR        снижается {view.atr_declining_bars} свечей подряд",
        "",
        "2. Объём (quote, USDT)",
        f"   текущий/база     {volume.ratio:.2f}x  → {volume.basis}, {volume.samples} набл."
        + ("  (слабая база)" if volume.weak_basis else ""),
        f"   MA20/MA100       {volume.ma_ratio:.2f}x  → "
        + ("затухание" if volume.ma_ratio < 1 else "рост"),
        f"   аномальные бары  {volume.anomalous_bars} за 30 свечей "
        f"(объём ≥3x при |Δцены| <0.5 ATR)",
        f"   taker buy доля   {volume.taker_buy_mean:.2f} средняя за 30 (нейтраль 0.50)",
        "",
        "3. Диапазон",
        f"   ширина(20)       {view.range_width * 100:.2f}% = "
        f"{view.range_width_atr:.1f} ATR",
        f"   перцентиль       {render_metric(view.range_metric, precision=2)}",
        f"   ниже порога      {view.narrow_bars} свечей подряд"
        + (
            f"  (порог {view.range_threshold * 100:.1f}% — 20-й перцентиль "
            f"своей истории)"
            if view.range_threshold == view.range_threshold else ""
        ),
        "",
    ]

    if view.profile:
        inside = "ВНУТРИ" if view.profile.contains(view.price) else "вне"
        lines += [
            "4. Объёмный профиль (прибл. из OHLCV, quote)",
            f"   POC~             {view.profile.poc:,.6g}",
            f"   Value Area~      {view.profile.value_area_low:,.6g} – "
            f"{view.profile.value_area_high:,.6g} · цена {inside}"
            + ("  ⚑" if inside == "ВНУТРИ" else ""),
            "",
        ]
    else:
        lines += ["4. Объёмный профиль — недостаточно данных", ""]

    lines += [
        "5. RSI(14) дивергенции",
        f"   {view.divergence or 'не обнаружено'}",
        "",
    ]

    if view.squeeze_index is None:
        lines.append("squeeze_index: не рассчитан — ни одна группа не измерима")
    else:
        parts = " + ".join(f"{k}={v:.2f}" for k, v in view.components.items())
        lines.append(f"squeeze_index = {view.squeeze_index:.2f}   ({parts})")
        if view.excluded:
            lines.append(
                f"исключено из индекса: {', '.join(view.excluded)} — "
                "веса оставшихся групп перенормированы"
            )
        lines.append("деривативы в индекс не входят (см. get_derivatives)")
    return "\n".join(lines)


def render_klines(
    series: Series, info: SymbolInfo, limit: int, *, market: Market = FUTURES
) -> str:
    """Уровень L3: сырые закрытые свечи с производными по каждой."""
    tail = series.tail(limit)
    precision = info.price_precision
    taker = tail.taker_buy_ratio

    # База объёма — та же, что в снапшоте. Среднее по показанному окну, стоявшее
    # здесь раньше, зависело от limit: свеча CAKE 30.08 давала 4.10x при
    # limit=50 и 2.25x при limit=14, а снапшот по ней же — третье число.
    offset = len(series) - len(tail)
    baseline = baseline_series(series, len(tail))
    ratios = baseline.ratio(series.quote_volume)[offset:]
    samples = baseline.samples[offset:]

    lines = [
        f"{info.symbol}{market.suffix} ({market.label}) {series.interval} · "
        f"последние {len(tail)} ЗАКРЫТЫХ свечей · "
        f"время UTC · объём в USDT",
        f"{'время':<17}{'open':>12}{'high':>12}{'low':>12}{'close':>12}"
        f"{'тело%':>8}{'верх%':>7}{'низ%':>7}{'объём':>8}{'takerB':>7}",
    ]

    opens, highs, lows, closes = (
        tail.col("open"), tail.high, tail.low, tail.close
    )
    for i in range(len(tail)):
        span = highs[i] - lows[i]
        body = (closes[i] - opens[i]) / opens[i] * 100 if opens[i] else 0.0
        upper = (highs[i] - max(opens[i], closes[i])) / span * 100 if span else 0.0
        lower = (min(opens[i], closes[i]) - lows[i]) / span * 100 if span else 0.0
        ratio = float(ratios[i])
        volume = "n/a" if ratio != ratio else (
            f"{ratio:.2f}x" + ("~" if samples[i] < MIN_SAMPLES_PER_SLOT else "")
        )
        lines.append(
            f"{utc(int(tail.df['open_time'].iloc[i]))[:16]:<17}"
            f"{format_price(opens[i], precision):>12}"
            f"{format_price(highs[i], precision):>12}"
            f"{format_price(lows[i], precision):>12}"
            f"{format_price(closes[i], precision):>12}"
            f"{body:>+8.2f}{upper:>7.0f}{lower:>7.0f}{volume:>8}{taker[i]:>7.2f}"
        )

    lines.append("")
    lines.append(
        "тело% — изменение от open к close; "
        "верх%/низ% — доля фитилей в диапазоне свечи"
    )
    lines.append(
        "объём — к той же базе, что в колонке «объём» снапшота: уровень 20 предыдущих "
        "свечей × сезонность слота суток; ~ слабая база. От limit не зависит."
    )
    lines.append(f"база последней свечи: {baseline.basis}")
    return "\n".join(lines)
