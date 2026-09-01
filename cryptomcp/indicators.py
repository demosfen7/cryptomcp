"""Индикаторы и перцентили (PLAN §4.2, §6.8).

Все функции принимают и возвращают numpy-массивы длины входа: начальные позиции,
где индикатор ещё не определён, заполнены NaN. Это позволяет индексировать
результат теми же индексами, что и исходный ряд, без сдвигов и офф-бай-ванов.

Сглаживание RSI и ATR — рекуррентное по Уайлдеру (seed средним за period, далее
(prev*(n-1) + x)/n), а не простая скользящая средняя. Это то же сглаживание, что
стоит за RMA в терминалах; расхождение с ними возможно на первых значениях, если
там иначе засеян ряд.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Минимумы базы для перцентиля (PLAN §4.2). Нарушение любого из двух означает,
#: что перцентиль не печатается: «8-й перцентиль из 14 наблюдений» выглядит так
#: же уверенно, как «8-й перцентиль из 360», и вводит в заблуждение.
MIN_PERCENTILE_OBS = 60
MIN_PERCENTILE_SPAN_DAYS = 60.0


def _empty_like(values: np.ndarray) -> np.ndarray:
    return np.full(len(values), np.nan, dtype="float64")


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Простая скользящая средняя."""
    values = np.asarray(values, dtype="float64")
    out = _empty_like(values)
    if period <= 0 or len(values) < period:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """Скользящее стандартное отклонение, ddof=0 — как в классических полосах Боллинджера."""
    values = np.asarray(values, dtype="float64")
    out = _empty_like(values)
    if period <= 0 or len(values) < period:
        return out
    mean = sma(values, period)
    sq_mean = sma(values**2, period)
    variance = np.clip(sq_mean - mean**2, 0.0, None)
    out[period - 1 :] = np.sqrt(variance[period - 1 :])
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Экспоненциальная скользящая средняя, засеянная SMA первых period значений."""
    values = np.asarray(values, dtype="float64")
    out = _empty_like(values)
    if period <= 0 or len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder(values: np.ndarray, period: int, first_index: int) -> np.ndarray:
    """Сглаживание Уайлдера: seed средним, дальше рекуррентно."""
    out = np.full(len(values), np.nan, dtype="float64")
    seed_end = first_index + period
    if seed_end > len(values):
        return out
    out[seed_end - 1] = values[first_index:seed_end].mean()
    for i in range(seed_end, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI по Уайлдеру."""
    close = np.asarray(close, dtype="float64")
    out = _empty_like(close)
    if len(close) <= period:
        return out

    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = loss[0] = np.nan

    avg_gain = _wilder(gain, period, first_index=1)
    avg_loss = _wilder(loss, period, first_index=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss)
        out = 100.0 - 100.0 / (1.0 + rs)
    # Серия без единого убытка даёт бесконечный RS — это RSI 100, не NaN.
    out = np.where((avg_loss == 0) & ~np.isnan(avg_gain), 100.0, out)
    out = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, out)
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    close = np.asarray(close, dtype="float64")
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR по Уайлдеру."""
    tr = true_range(high, low, close)
    if len(tr) < period:
        return _empty_like(tr)
    return _wilder(tr, period, first_index=0)


def bollinger_width(close: np.ndarray, period: int = 20, k: float = 2.0) -> np.ndarray:
    """BBW = (верхняя − нижняя) / средняя. Основной индикатор сжатия (ТЗ §4.2)."""
    close = np.asarray(close, dtype="float64")
    middle = sma(close, period)
    std = rolling_std(close, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(2.0 * k * std, middle, out=_empty_like(close), where=middle != 0)


def donchian_width(high: np.ndarray, low: np.ndarray, period: int, close: np.ndarray) -> np.ndarray:
    """Ширина канала за period свечей, нормированная к цене (ТЗ §4.2, группа 3)."""
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    close = np.asarray(close, dtype="float64")
    out = _empty_like(close)
    if len(close) < period or period <= 0:
        return out
    for i in range(period - 1, len(close)):
        window_high = high[i - period + 1 : i + 1].max()
        window_low = low[i - period + 1 : i + 1].min()
        if close[i]:
            out[i] = (window_high - window_low) / close[i]
    return out


def consecutive_below(values: np.ndarray, threshold: float) -> int:
    """Сколько последних значений подряд ниже порога (длительность сжатия)."""
    count = 0
    for value in reversed(values):
        if np.isnan(value) or value >= threshold:
            break
        count += 1
    return count


def consecutive_declining(values: np.ndarray) -> int:
    """Сколько последних значений подряд снижаются — тренд ATR из ТЗ §4.2."""
    clean = values[~np.isnan(values)]
    count = 0
    for i in range(len(clean) - 1, 0, -1):
        if clean[i] < clean[i - 1]:
            count += 1
        else:
            break
    return count


def percentile_rank(history: np.ndarray, value: float) -> float:
    """Положение значения в распределении, 0..100.

    Равные значения считаются половиной — стандартное определение перцентильного
    ранга, устойчивое к повторам (у BBW они бывают на плоских участках).
    """
    clean = np.asarray(history, dtype="float64")
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0 or np.isnan(value):
        return float("nan")
    below = np.count_nonzero(clean < value)
    equal = np.count_nonzero(clean == value)
    return 100.0 * (below + 0.5 * equal) / len(clean)


@dataclass(frozen=True)
class Metric:
    """Значение вместе с базой сравнения.

    Голое число модель оценить не может: 0.0413 — это много или мало? Поэтому
    рендерер не умеет печатать значение без контекста, а контекст обязан нести
    размер и охват базы (PLAN §4.2).
    """

    name: str
    value: float
    unit: str = ""
    pct_rank: float | None = None
    n_obs: int = 0
    span_days: float = 0.0
    #: Пояснение к базе сравнения. Либо причина, по которой перцентиля нет,
    #: либо — при снятом требовании к охвату — оговорка о том, чем база
    #: ограничена (история OI обрезана биржей 30 сутками).
    base_note: str | None = None
    threshold: float | None = None
    #: "below" — признак срабатывает ниже порога, "above" — выше.
    threshold_side: str | None = None

    @property
    def flagged(self) -> bool:
        """Сработал ли порог по перцентилю."""
        if self.threshold is None or self.pct_rank is None:
            return False
        if self.threshold_side == "above":
            return self.pct_rank >= self.threshold
        return self.pct_rank <= self.threshold

    @property
    def has_context(self) -> bool:
        return self.pct_rank is not None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "value": self.value}
        if self.unit:
            payload["unit"] = self.unit
        if self.pct_rank is not None:
            payload["pct_rank"] = round(self.pct_rank, 1)
            payload["n_obs"] = self.n_obs
            payload["span_days"] = round(self.span_days, 1)
        else:
            payload["context"] = self.base_note or "нет базы для сравнения"
        if self.threshold is not None:
            payload["threshold"] = self.threshold
            payload["flagged"] = self.flagged
        return payload


def with_percentile(
    name: str,
    value: float,
    history: np.ndarray,
    span_days: float,
    *,
    unit: str = "",
    threshold: float | None = None,
    threshold_side: str = "below",
    min_obs: int = MIN_PERCENTILE_OBS,
    min_span_days: float = MIN_PERCENTILE_SPAN_DAYS,
    span_exemption: str | None = None,
) -> Metric:
    """Собрать метрику, применив правило достаточности базы (PLAN §4.2).

    ``span_exemption`` — текст причины, по которой требование к календарному
    охвату снимается. Используется только там, где данных глубже не существует
    у самой биржи: история открытого интереса обрезана 30 сутками (PLAN §4.12).
    """
    clean = np.asarray(history, dtype="float64")
    clean = clean[~np.isnan(clean)]
    n_obs = len(clean)

    if n_obs < min_obs:
        reason = f"n/a (наблюдений {n_obs}, нужно {min_obs})"
        return Metric(name, value, unit, None, n_obs, span_days, reason, threshold, threshold_side)

    if span_days < min_span_days and span_exemption is None:
        reason = f"n/a (охват {span_days:.1f} сут., нужно {min_span_days:.0f})"
        return Metric(name, value, unit, None, n_obs, span_days, reason, threshold, threshold_side)

    return Metric(
        name=name,
        value=value,
        unit=unit,
        pct_rank=percentile_rank(clean, value),
        n_obs=n_obs,
        span_days=span_days,
        base_note=span_exemption,
        threshold=threshold,
        threshold_side=threshold_side,
    )
