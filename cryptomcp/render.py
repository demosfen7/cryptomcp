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
from .indicators import Metric
from .levels import Level, Pivots
from .series import Series
from .symbols import SymbolInfo, format_price

#: Ближе этого расстояния уровень считается «под ценой», и показывается
#: следующий за ним — иначе видно, что цена на уровне, но не видно, куда ход.
NEAR_LEVEL_ATR = 0.25


def utc(ms: int) -> str:
    if not ms:
        return "n/a"
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M:%S")


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
) -> str:
    """Уровень L1: общая картина, 20–25 строк."""
    precision = info.price_precision
    lines: list[str] = []

    head = f"{info.symbol}.P (USDⓈ-M perp) · цена {format_price(live_price, precision)}"
    if as_of_ms is None:
        head += f" (live {utc(now_ms)} UTC)"
    else:
        head += f" (состояние на {utc(as_of_ms)} UTC)"
    lines.append(head)
    lines.append(
        f"24h {change_24h:+.2f}% · оборот {quote_volume_24h / 1e9:.2f}B USDT · "
        f"метрики по ЗАКРЫТЫМ свечам"
    )
    lines.append("")
    lines.append(
        f"{'ТФ':<5}{'EMA':<8}{'структ':<9}{'поз':>5}{'RSI':>6}{'ATR%':>7}"
        f"{'объём':>8}{'takerB':>8}  сжатие"
    )

    for interval, view in views.items():
        if view.bbw.has_context:
            squeeze = f"BBW {view.bbw.pct_rank:.0f} pct"
            if view.bbw.flagged:
                squeeze = f"ДА · {squeeze}, {view.range_duration} св."
        else:
            squeeze = view.bbw.base_note or "n/a"
        volume = f"{view.volume.ratio:.2f}x" + ("~" if view.volume.weak_basis else "")
        lines.append(
            f"{interval:<5}{view.ema_state:<8}{view.structure:<9}"
            f"{view.position_in_range:>5.2f}{view.rsi_value:>6.1f}"
            f"{view.atr_pct:>6.2f}%{volume:>8}{view.volume.taker_buy_mean:>8.2f}  {squeeze}"
        )

    lines.append("")
    lines.append("объём — к уровню последних свечей с поправкой на слот суток; ~ слабая база")
    lines.append("EMA — цена против EMA50/EMA200; структ — два последних swing-экстремума")

    anchor = views.get("4h") or next(iter(views.values()))
    lines.append("")
    lines.append(
        f"диапазон {anchor.interval}: {format_price(anchor.range_low, precision)} – "
        f"{format_price(anchor.range_high, precision)} "
        f"({anchor.range_width * 100:.2f}%, {anchor.range_width_atr:.1f} ATR, "
        f"держится {anchor.range_duration} св.)"
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
        lines.append(render_derivatives(funding, open_interest))

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


def render_derivatives(
    funding: Funding | None, open_interest: OpenInterest | None
) -> str:
    lines: list[str] = []
    if funding:
        context = ""
        if funding.percentile and funding.percentile.has_context:
            context = (
                f", {funding.percentile.pct_rank:.0f} pct за "
                f"{funding.percentile.span_days:.0f} сут."
            )
        lines.append(
            f"фандинг {funding.rate * 100:+.4f}% / {funding.interval_hours}ч "
            f"= {funding.annualized_pct:+.2f}% годовых{context}"
        )
    if open_interest:
        notional = open_interest.notional_usdt
        scale = f"{notional / 1e9:.2f}B" if notional >= 1e9 else f"{notional / 1e6:.0f}M"
        lines.append(f"OI {scale} USDT:")
        for window in open_interest.change:
            lines.append(
                f"  {window:>4}: OI {open_interest.change[window] * 100:+6.2f}% · "
                f"цена {open_interest.price_change[window] * 100:+6.2f}% "
                f"→ {open_interest.quadrant(window)}"
            )
    return "\n".join(lines)


def render_squeeze_metrics(view: TimeframeView, threshold_pct: float) -> str:
    """Уровень L2: пять групп признаков ТЗ §4.2 с базой сравнения."""
    volume = view.volume
    lines = [
        f"{view.interval} · база перцентилей: {view.bbw.n_obs} наблюдений, "
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
        f"   ширина(20)       {view.range_width * 100:.2f}% = {view.range_width_atr:.1f} ATR"
        f"  → порог {threshold_pct * 100:.1f}%"
        + ("  ⚑" if view.range_width < threshold_pct else ""),
        f"   длительность     {view.range_duration} свечей подряд",
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


def render_klines(series: Series, info: SymbolInfo, limit: int) -> str:
    """Уровень L3: сырые закрытые свечи с производными по каждой."""
    tail = series.tail(limit)
    precision = info.price_precision
    volumes = tail.quote_volume
    mean_volume = float(volumes.mean()) if len(volumes) else 0.0
    taker = tail.taker_buy_ratio

    lines = [
        f"{info.symbol} {series.interval} · последние {len(tail)} ЗАКРЫТЫХ свечей · "
        f"время UTC · объём в USDT",
        f"{'время':<17}{'open':>12}{'high':>12}{'low':>12}{'close':>12}"
        f"{'тело%':>8}{'верх%':>7}{'низ%':>7}{'об./ср':>8}{'takerB':>7}",
    ]

    opens, highs, lows, closes = (
        tail.col("open"), tail.high, tail.low, tail.close
    )
    for i in range(len(tail)):
        span = highs[i] - lows[i]
        body = (closes[i] - opens[i]) / opens[i] * 100 if opens[i] else 0.0
        upper = (highs[i] - max(opens[i], closes[i])) / span * 100 if span else 0.0
        lower = (min(opens[i], closes[i]) - lows[i]) / span * 100 if span else 0.0
        ratio = volumes[i] / mean_volume if mean_volume else float("nan")
        lines.append(
            f"{utc(int(tail.df['open_time'].iloc[i]))[:16]:<17}"
            f"{format_price(opens[i], precision):>12}"
            f"{format_price(highs[i], precision):>12}"
            f"{format_price(lows[i], precision):>12}"
            f"{format_price(closes[i], precision):>12}"
            f"{body:>+8.2f}{upper:>7.0f}{lower:>7.0f}{ratio:>8.2f}{taker[i]:>7.2f}"
        )

    lines.append("")
    lines.append(
        "тело% — изменение от open к close; верх%/низ% — доля фитилей в диапазоне свечи; "
        "об./ср — объём к среднему по показанному окну"
    )
    return "\n".join(lines)
