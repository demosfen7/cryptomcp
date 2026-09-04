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
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .analysis import SHOCK_RANGE_ATR, SHOCK_VOLUME_MULTIPLE, TimeframeView
from .derivatives import Funding, OpenInterest, side_of_flow
from .errors import ErrorKind, ToolError
from .indicators import Metric
from .levels import Level, Pivots
from .markets import FUTURES, Market, market_short
from .series import Series
from .symbols import SymbolInfo, format_price
from .volume import (
    MIN_SAMPLES_PER_SLOT,
    TAKER_PRESSURE,
    Absorption,
    baseline_series,
    candles,
)

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
        f"{'объём':>8}{'MA20/100':>10}{'takerB':>8}  сжатие"
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
        squeeze += f" · узк {view.narrow_bars} ({view.narrow_days:.1f} сут)"
        # Событие внутри окна сжатия печатается рядом с длительностью, потому
        # что оно эту длительность и обесценивает (§4.26).
        if view.shock is not None and view.shock.loud:
            squeeze += f" · ШОК {view.shock.range_atr:.1f} ATR"
        volume = f"{view.volume.ratio:.2f}x" + ("~" if view.volume.weak_basis else "")
        # Затухание длинного объёма при оживающем коротком видно только в
        # сравнении строк лестницы, поэтому колонка стоит здесь, а не в L2.
        ma_ratio = (
            f"{view.volume.ma_ratio:.2f}x"
            if view.volume.ma_ratio == view.volume.ma_ratio else "n/a"
        )
        lines.append(
            f"{interval:<5}{view.ema_state:<8}{view.structure:<9}"
            f"{view.position_in_range:>5.2f}{view.rsi_value:>6.1f}"
            f"{view.atr_pct:>6.2f}%{volume:>8}{ma_ratio:>10}"
            f"{view.volume.taker_buy_mean:>8.2f}  {squeeze}"
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
        "MA20/100 — объём последних 20 свечей к последним 100 на ЭТОМ ТФ: "
        "расхождение между строками (длинный мёртв, короткий оживает) и есть "
        "самый ранний признак набора"
    )
    lines.append(
        "узк N — свечей подряд с шириной диапазона(20) ниже 20-го перцентиля "
        "СВОЕЙ истории; ШОК — бар внутри этого окна размахом от 3 ATR при "
        "объёме от 3x, то есть сжатие держит внутри себя событие"
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
        if funding.structurally_negative:
            # Иначе глубокая ставка читается как событие, хотя это норма монеты.
            lines.append(
                f"  СТРУКТУРНО отрицательный: медиана за 30 сут. "
                f"{funding.median_annual_pct:+.0f}% годовых — для этой монеты "
                f"это норма, а не разовый перекос"
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
        annual = funding.annualized_pct if funding else None
        for window in open_interest.change:
            reading = side_of_flow(open_interest.quadrant(window), annual)
            lines.append(
                f"  {window:>4}: OI {open_interest.change[window] * 100:+6.2f}% · "
                f"цена {open_interest.price_change[window] * 100:+6.2f}% "
                f"→ {reading}"
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
        # База здесь ДРУГАЯ, чем строкой выше: там сезонная, здесь скользящая
        # двадцатка. Разница вещественная — у ASTER 18.08 одна и та же свеча
        # давала 7.14x и 4.40x, — поэтому база подписана у каждого числа.
        f"   бары набора      {volume.anomalous_bars} за "
        f"{candles(volume.bars_window)} "
        f"(объём ≥3x к MA20 при тихом теле: <0.5 ATR или <0.3%)",
        f"   taker buy доля   {volume.taker_buy_mean:.2f} средняя за 30 (нейтраль 0.50)",
        "",
        "3. Диапазон",
        f"   ширина(20)       {view.range_width * 100:.2f}% = "
        f"{view.range_width_atr:.1f} ATR",
        f"   перцентиль       {render_metric(view.range_metric, precision=2)}",
        f"   ниже порога      {view.narrow_bars} свечей подряд "
        f"= {view.narrow_days:.1f} сут"
        + (
            f"  (порог {view.range_threshold * 100:.1f}% — 20-й перцентиль "
            f"своей истории)"
            if view.range_threshold == view.range_threshold else ""
        ),
        # Длительность сама по себе несопоставима между парами: у монеты,
        # которая никогда не стояла дольше двух суток, полтора дня — рекорд,
        # а у BTC — шум. Поэтому рядом её перцентиль среди ЗАВЕРШЁННЫХ серий.
        f"   длительность     {render_metric(view.narrow_metric, precision=0)}"
        + (
            f", завершённых серий {view.narrow_metric.n_obs}"
            if view.narrow_metric.has_context else ""
        ),
        "",
    ]
    # Длительность сжатия меряет не то, что кажется, если внутри окна стоит
    # бар размахом почти во весь диапазон (§4.26). Вывод не делается: событие
    # в узком диапазоне бывает и истощением, и перезарядкой.
    if view.shock is not None:
        shock = view.shock
        share = f"{shock.range_share * 100:.0f}% ширины"
        if shock.loud:
            note = (
                f"   шок внутри       бар {shock.range_atr:.1f} ATR = {share}, "
                f"объём {shock.volume_ratio:.1f}x, {candles(shock.bars_ago)} назад "
                f"— СОБЫТИЕ внутри сжатия"
            )
        else:
            note = (
                f"   шок внутри       нет: максимум {shock.range_atr:.1f} ATR "
                f"({share}) при объёме {shock.volume_ratio:.1f}x"
            )
        lines.insert(len(lines) - 1, note)

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


#: Короткие имена групп индекса для табличной выдачи. Порядок — как в формуле.
#: Группы свёртки и их подписи в таблицах. Профиль и дивергенция убраны
#: вместе с их весами (§4.33): в выдаче они остались справкой, но колонки в
#: журнале держать незачем — вклада в индекс у них больше нет.
INDEX_GROUPS: tuple[tuple[str, str], ...] = (
    ("volatility", "vola"),
    ("range", "rang"),
    ("volume", "volu"),
    ("duration", "длит"),
)


def _cell(value: Any, spec: str = "") -> str:
    """Число или прочерк. Пусто значит «не измерено», а не «ноль»."""
    if value is None:
        return "—"
    return f"{value:{spec}}" if spec else f"{value}"


def _shock(row: Mapping[str, Any]) -> str:
    """Размах события внутри сжатия — или прочерк.

    Печатается только событие: размах от 3 ATR при объёме от 3x. Самый широкий
    бар тихого диапазона в колонке не нужен — в узком диапазоне он есть всегда
    и ни о чём не говорит (§4.26). Пустая ячейка здесь значит «сжатия нет или
    оно тихое», и это разные вещи, но обе — не событие.
    """
    atr_value, volume = row.get("shock_atr"), row.get("shock_volume")
    if atr_value is None or volume is None:
        return "—"
    if atr_value < SHOCK_RANGE_ATR or volume < SHOCK_VOLUME_MULTIPLE:
        return "—"
    return f"{atr_value:.1f}"


def _days(bars: Any, tf: str) -> str:
    """Свечи в сутки. «Узк 19» на 4h — это 3.2 суток, а «узк 7» на 1d — семь."""
    if bars is None:
        return "—"
    from .series import interval_ms

    return f"{int(bars) * interval_ms(tf) / 86_400_000:.1f}"


def _twin(row: Mapping[str, Any]) -> str:
    """Индекс, ширина и длительность на соседнем рынке — три колонки.

    Пусто у пар, которым соседний рынок не считался (он считается по верхушке
    списка) и у которых его нет вовсе — четверть перпетуалов не имеет
    спотовой пары. Прочерк тут значит «не мерили», а не «совпало».
    """
    index = row.get("twin_index")
    width = row.get("twin_range_width_pct")
    bars = row.get("twin_narrow_bars")
    return (
        f"{f'{index:.2f}' if index is not None else '—':>7}"
        f"{f'{width:.2f}%' if width is not None else '—':>8}"
        f"{bars if bars is not None else '—':>6}"
    )


def _price(value: Any) -> str:
    """Цена без потери разряда и без выдуманной точности.

    Восьми значащих хватает обоим краям рынка: 90123.45 не обрезается до
    90123.4, а 0.00581042 не превращается в 0.0058. Настоящий шаг цены живёт в
    SymbolInfo и приходит с биржи, но здесь выдача строится только из архива —
    ходить за ним ради колонки было бы запросом на ровном месте.
    """
    return f"{value:.8g}" if value else "n/a"


def _pair(entry: Any, current: Any) -> str:
    """Значение «на входе → сейчас».

    Прочерк там, где значения нет: у ручной записи ранга при входе не было, а
    выпавшая из универсума монета потеряла текущий. Это разные вещи, и
    показывать их одинаковым нулём нельзя.
    """
    left = f"{entry}" if entry else "—"
    right = f"{current}" if current else "—"
    return f"{left}→{right}"


def render_watchlist(
    episodes: Sequence[dict[str, Any]],
    scans: dict[tuple[str, str], dict[str, Any]],
    *,
    now_ms: int,
    prices: dict[str, float] | None = None,
) -> str:
    """Список наблюдения столбиком.

    Ширина диапазона и длительность сжатия печатаются отдельными колонками, а
    не сворачиваются в индекс: ранг сам по себе пропускает и широкие диапазоны
    (BULLA с 40% попала в топ на первой же выдаче), и решать, что с этим
    делать, должен человек.

    Колонка накопления печатается всегда, даже пока метрики нет: пустое место
    в ней — это «не измерено», а не «признака нет». Тот же принцип, по
    которому метрика без базы печатает причину, а не ноль.

    Колонка «кем» — источник записи: scanner, manual или scanner+manual у
    монеты, которую сканер отобрал сам, а человек до того заметил глазами
    (§4.28). Заметки печатаются под таблицей: в строке им места нет, а
    выбрасывать их нельзя — ради них ручную запись и заводят.
    """
    if not episodes:
        return "список наблюдения пуст"

    closed = any(row.get("exited_at") for row in episodes)
    lines = [
        f"эпизодов: {len(episodes)}",
        f"{'символ':<14}{'ТФ':>4}{'рынок':>7}{'статус':>11}{'ранг':>10}"
        f"{'инд':>7}{'Δинд':>7}"
        f"{'накопл':>8}{'узк':>5}{'сут':>6}{'узк²':>6}{'вход':>13}{'сейчас':>9}{'дней':>6}  кем"
        + ("  ·  чем кончилось" if closed else ""),
    ]

    for row in episodes:
        scan = scans.get((row["symbol"], row["tf"])) or {}
        index_now = row.get("last_index")
        index_in = row.get("squeeze_index")
        delta = (
            f"{index_now - index_in:+.2f}"
            if index_now is not None and index_in is not None else "—"
        )
        price_in = row.get("price_at_entry")
        # У сканерной записи «сейчас» — цена последней закрытой свечи скана, у
        # ручной её взять неоткуда: монеты может не быть в архиве вовсе, ради
        # таких её и заводят руками. Тогда берётся живая цена, и это честно:
        # цена входа у ручной записи тоже живая (§4.28).
        price_now = scan.get("price") or (prices or {}).get(row["symbol"])
        move = (
            f"{(price_now / price_in - 1) * 100:+.1f}%"
            if price_now and price_in else "n/a"
        )
        # У закрытого эпизода возраст считается до выхода, а не до сейчас:
        # иначе закрытые вчера продолжали бы «стареть» в выдаче.
        until = row.get("exited_at") or now_ms
        accumulation = row.get("accumulation_score")
        narrow = scan.get("narrow_bars")

        line = (
            f"{row['symbol']:<14}{row['tf']:>4}"
            f"{market_short(row.get('market') or scan.get('source')):>7}"
            f"{row['status']:>11}"
            f"{_pair(row.get('rank_at_entry'), row.get('last_rank')):>10}"
            f"{f'{index_now:.2f}' if index_now is not None else '—':>7}{delta:>7}"
            f"{f'{accumulation:.2f}' if accumulation is not None else 'n/a':>8}"
            f"{narrow if narrow is not None else '—':>5}"
            f"{_days(narrow, row['tf']):>6}"
            f"{_cell(scan.get('twin_narrow_bars')):>6}"
            f"{_price(price_in):>13}{move:>9}"
            f"{(until - row['entered_at']) / 86_400_000:>6.1f}  {row['entered_by']}"
        )
        if row.get("exited_at"):
            line += f"  ·  {row.get('exit_reason') or row['status']}"
        lines.append(line)

    notes = [
        f"  {row['symbol']} {row['tf']} — {row['note']}"
        for row in episodes if row.get("note")
    ]
    if notes:
        lines += ["", "заметки к ручным записям:"] + notes

    lines += [
        "",
        "ранг и Δинд — «при входе → сейчас»; инд — squeeze_index последнего скана",
        "узк — свечей подряд с шириной диапазона(20) ниже 20-го перцентиля "
        "своей истории; сут — та же длительность календарём, потому что "
        "свечи 4h и 1d в одном списке несопоставимы",
        "рынок — ряд, по которому монета отобрана: архив предпочитает спот "
        "(история глубже), и числа эпизода относятся именно к нему; "
        "сверять их через get_squeeze_metrics нужно с тем же market",
        "узк² — та же длительность на соседнем рынке (у спотовой записи это "
        "перп, у фьючерсной — спот): на HOMEUSDT 4h вышло 17 против 36 на "
        "одной свече. Справка; ни в ранг, ни в отбор не входит",
        "накопл — метрика накопления; колонка заведена, метрика ещё не считается",
        "кем — источник: scanner отбирает рангом, manual заводится руками и "
        "рангом не снимается (только руками или по сроку в 30 суток)",
        "«сейчас» у сканерных записей — цена последней закрытой свечи скана, "
        "у ручных — живая: их вход тоже отмечен по живой",
    ]
    return "\n".join(lines)


def render_scan_history(
    symbol: str, tf: str, rows: Sequence[dict[str, Any]], version: str
) -> str:
    """История индекса по паре с разложением на группы, свежие сверху.

    Разложение печатается всегда: вопрос «сжимается третью неделю или вошёл
    вчера» решается не индексом, а тем, какая из групп его держит и растёт ли
    она. Группа, исключённая из-за нехватки базы, показывается прочерком, а не
    нулём — ноль означал бы «признака нет».
    """
    if not rows:
        return (
            f"{symbol} {tf}: записей скана нет. Сканируются только "
            f"таймфреймы из списка наблюдения, и только по монетам архива."
        )

    header = "".join(f"{short:>7}" for _, short in INDEX_GROUPS)
    # Колонки соседнего рынка печатаются, только если он считался: считается
    # он по верхушке списка, и у остальных пар прочерк означал бы не «совпало»,
    # а «не мерили» — лишний столбец прочерков читателю ничего не говорит.
    twin = any(row.get("twin_index") is not None for row in rows)
    twin_head = f"{'инд²':>7}{'диап²':>8}{'узк²':>6}" if twin else ""
    markets = " / ".join(
        dict.fromkeys(market_short(row.get("source")) for row in rows)
    )
    lines = [
        f"{symbol} · {tf} · рынок: {markets} · формула {version} · "
        f"записей {len(rows)}, свежие сверху",
        f"{'закрыта':<17}{'индекс':>7}{header}{'BBW':>6}{'диап':>8}{'узк':>5}"
        f"{'сут':>6}{twin_head}{'объём':>8}{'МА':>7}{'шок':>6}{'погл':>6}{'клст':>6}{'tkМакс':>8}"
        f"{'лид':>6}{'фанд%':>9}{'цена':>13}",
    ]

    for row in rows:
        components = json.loads(row.get("components") or "{}")
        groups = "".join(
            f"{components[key]:>7.2f}" if key in components else f"{'—':>7}"
            for key, _ in INDEX_GROUPS
        )
        index = row.get("squeeze_index")
        bbw = row.get("bbw_pct_rank")
        width = row.get("range_width_pct")
        volume = row.get("volume_ratio")
        # closed_through_ms — последняя миллисекунда свечи, поэтому печатается
        # граница: «04:00», а не «03:59:59.999». Тот же приём, что в
        # closed_through(). Запасной ts_ms — просто момент прогона, ему +1 не
        # нужен, и путать эти две величины нельзя.
        closed = row.get("closed_through_ms")
        stamp = int(closed) + 1 if closed else int(row["ts_ms"])
        lines.append(
            f"{utc(stamp)[:16]:<17}"
            f"{index if index is not None else 0:>7.2f}{groups}"
            f"{f'{bbw:.0f}' if bbw is not None else 'n/a':>6}"
            f"{f'{width:.2f}%' if width is not None else 'n/a':>8}"
            f"{row.get('narrow_bars') if row.get('narrow_bars') is not None else '—':>5}"
            f"{_days(row.get('narrow_bars'), tf):>6}"
            f"{_twin(row) if twin else ''}"
            f"{f'{volume:.2f}x' if volume is not None else 'n/a':>8}"
            f"{_cell(row.get('ma_ratio'), '.2f'):>7}"
            f"{_shock(row):>6}"
            f"{_cell(row.get('absorption_bars')):>6}"
            f"{_cell(row.get('absorption_clusters')):>6}"
            f"{_cell(row.get('taker_max'), '.2f'):>8}"
            f"{_cell(row.get('volume_lead'), '+d'):>6}"
            f"{_cell(row.get('funding_annual'), '+.0f'):>9}"
            f"{_price(row.get('price')):>13}"
        )

    if twin:
        markets_twin = " / ".join(
            dict.fromkeys(
                market_short(row.get("twin_market"))
                for row in rows if row.get("twin_index") is not None
            )
        )
        lines.append(
            f"\n² — то же на соседнем рынке ({markets_twin}); справка, "
            "в индекс и в отбор не входит"
        )

    reading = next(
        (row["oi_reading"] for row in rows if row.get("oi_reading")), None
    )
    if reading:
        lines.append(f"\nпоток по OI за сутки (последняя запись): {reading}")
    lines += [
        "",
        "МА — объём MA20/MA100 на этом ТФ · шок — размах в ATR самого "
        "широкого бара ВНУТРИ окна сжатия; печатается только событие: от "
        "3 ATR при объёме от 3x",
        "погл — бары набора на младшем ряду · клст — кластеры набора "
        "(серия свечей, за которую цена никуда не ушла) · tkМакс — максимум "
        "доли тейкер-покупок · лид — на сколько свечей объём опередил цену",
        "группы — вклад в индекс до взвешивания; «—» значит базы не хватило "
        "и группа исключена из формулы с перенормировкой весов",
        "запись одна на закрытую свечу, поэтому шаг строк равен таймфрейму",
        "рынок — ряд, по которому считалась строка; чтобы сверить её с "
        "get_squeeze_metrics, вызывать его с тем же market",
    ]
    return "\n".join(lines)


def render_absorption(data: Absorption | None, *, skipped: str | None = None) -> str:
    """Блок поглощения — по младшему ряду, отдельной секцией.

    Печатается отдельно от группы 2 сознательно: там объём СВОЕГО таймфрейма и
    его затухание, здесь — набор позиции внутри дня. Слить их в одну секцию
    значило бы предложить сравнить числа, посчитанные по разным рядам.
    """
    if data is None or skipped is not None:
        return f"\n\n6. Поглощение (младший ТФ)\n   n/a — {skipped}"

    # «3+ подряд» — та граница, ниже которой серия неотличима от выброса.
    streak = f"{data.taker_streak} подряд" if data.taker_streak else "нет"
    # Кластер важнее одиночного бара: один бар с большим объёмом почти всегда
    # новость или вынос стопов, набор — это серия (§4.25).
    clusters = (
        f"{data.clusters}, длиннейший {candles(data.cluster_longest)}"
        if data.clusters else "нет"
    )
    return "\n".join([
        f"\n\n6. Поглощение (ряд {data.interval}, окно {candles(data.window)})",
        f"   бары набора      {data.bars} "
        f"(объём ≥3x к MA20 при тихом теле: <0.5 ATR или <0.3%)",
        f"   кластеры         {clusters} "
        f"(3+ свечей: объём ≥1.5x, тело <40% диапазона, цена на месте)",
        f"   нижние фитили    {candles(data.wick_streak)} подряд "
        f"больше половины диапазона",
        f"   takerB           средняя {data.taker_mean:.2f} · "
        f"максимум {data.taker_max:.2f} · "
        f"выше {TAKER_PRESSURE:.2f}: {data.taker_above} свечей",
        f"   серия выше 0.50  {streak}",
        f"   объём/цена       {data.lead_state}",
        f"   объём последней  {data.volume_ratio:.2f}x"
        + ("~ слабая база" if data.weak_basis else ""),
        "   считается на младшем ряду: дневное разрешение стирает поглощение "
        "внутри свечи",
    ])


def render_screen(
    tf: str,
    rows: Sequence[dict[str, Any]],
    *,
    version: str,
    sort_by: str,
    filtered: int,
    logged: int,
    matched: int,
    earlier: Sequence[str] = (),
) -> str:
    """Отбор по журналу: строка на монету, с признаками накопления.

    Печатается «закрыта по», потому что строки берутся из журнала, а он
    пишется на закрытие свечи: без штампа выдача выглядела бы живой, а
    отставать может почти на целый таймфрейм.
    """
    head = [
        f"{tf} · формула {version} · сортировка: {sort_by}",
        f"фильтр универсума пропустил {filtered} монет · "
        f"в журнале по {tf}: {logged} · совпало: {matched} · показано: {len(rows)}",
    ]
    if not rows:
        # Причины у пустой выдачи разные, и подсказка обязана их различать:
        # «журнала нет» лечится не тем же, чем «фильтры отсеяли всех».
        reason = (
            f"формула поднята до {version}, а записи в журнале пока только "
            f"прежних поколений ({', '.join(earlier)}). Ранги по разным "
            "поколениям не сравниваются, поэтому журнал наполняется заново: "
            "первая строка появится с ближайшим часовым прогоном сканера."
            if logged == 0 and earlier else
            f"по таймфрейму {tf} записей скана нет вовсе. Сканер ведёт журнал "
            "только по 4h, 1d и 1h и только по монетам архива; для остальных "
            "передайте symbols явно — они будут посчитаны по бирже."
            if logged == 0 else
            "записи в журнале есть, но ни одна не прошла фильтры. Ослабьте "
            "min_narrow_bars, границы оборота или max_abs_change_24h."
        )
        return "\n".join(head + ["", reason])

    head.append(
        f"{'символ':<14}{'индекс':>7}{'BBW':>6}{'диап':>8}{'узк':>5}{'объём':>8}"
        f"{'МА':>7}{'шок':>6}{'погл':>6}{'клст':>6}{'tkМакс':>8}{'лид':>6}"
        f"{'фанд%':>9}  поток по OI за сутки"
    )
    lines = list(head)
    for row in rows:
        index = row.get("squeeze_index")
        bbw = row.get("bbw_pct_rank")
        width = row.get("range_width_pct")
        volume = row.get("volume_ratio")
        lines.append(
            f"{row['symbol']:<14}"
            f"{index if index is not None else 0:>7.2f}"
            f"{f'{bbw:.0f}' if bbw is not None else 'n/a':>6}"
            f"{f'{width:.2f}%' if width is not None else 'n/a':>8}"
            f"{_cell(row.get('narrow_bars')):>5}"
            f"{f'{volume:.2f}x' if volume is not None else 'n/a':>8}"
            f"{_cell(row.get('ma_ratio'), '.2f'):>7}"
            f"{_shock(row):>6}"
            f"{_cell(row.get('absorption_bars')):>6}"
            f"{_cell(row.get('absorption_clusters')):>6}"
            f"{_cell(row.get('taker_max'), '.2f'):>8}"
            f"{_cell(row.get('volume_lead'), '+d'):>6}"
            f"{_cell(row.get('funding_annual'), '+.0f'):>9}"
            f"  {row.get('oi_reading') or '—'}"
        )

    stamps = {row.get("closed_through_ms") for row in rows if row.get("closed_through_ms")}
    if stamps:
        lines.append("")
        lines.append("закрыты по (UTC): " + " · ".join(
            sorted({utc(int(ms) + 1)[:16] for ms in stamps})
        ))
    lines += [
        "МА — объём MA20/MA100 на этом ТФ · шок — размах в ATR самого "
        "широкого бара ВНУТРИ окна сжатия; печатается только событие: от "
        "3 ATR при объёме от 3x",
        "погл — бары набора на младшем ряду · клст — кластеры набора "
        "(серия свечей, за которую цена никуда не ушла) · tkМакс — максимум "
        "доли тейкер-покупок · лид — на сколько свечей объём опередил цену",
        "выдача из журнала сканера, а не пересчёт: строка пишется на закрытие "
        "свечи, поэтому она может отставать почти на таймфрейм",
    ]
    if sort_by == "accumulation":
        lines.append(
            "сортировка «накопление» — порядок по составляющим (кластеры "
            "набора, затем одиночные бары, тейкеры и лид), а не по сводному "
            "числу: сводного пока нет"
        )
    return "\n".join(lines)
