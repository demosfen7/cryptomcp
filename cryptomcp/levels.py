"""Уровни: swing-экстремумы, структура, кластеры S/R, пивоты, объёмный профиль.

Реализует группы 3 и 4 признаков ТЗ §4.2 плюс пивоты (PLAN §4.7).

Про объёмный профиль: настоящий строится по тиковым сделкам, а из OHLCV объём
можно только размазать по диапазону свечи. POC получается смещённым, поэтому во
всей выдаче он помечается тильдой как приближённый (PLAN §4.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SwingKind = Literal["high", "low"]
Structure = Literal["HH/HL", "LH/LL", "mixed", "unknown"]


@dataclass(frozen=True)
class Swing:
    index: int
    price: float
    kind: SwingKind


@dataclass(frozen=True)
class Level:
    """Кластер близких экстремумов — уровень поддержки или сопротивления."""

    price: float
    touches: int
    #: Индекс последнего касания: свежий уровень весомее давно забытого.
    last_touch_index: int
    kind: SwingKind

    def distance_pct(self, price: float) -> float:
        return (self.price - price) / price * 100.0

    def distance_atr(self, price: float, atr_value: float) -> float:
        if not atr_value:
            return float("nan")
        return (self.price - price) / atr_value


@dataclass(frozen=True)
class Pivots:
    """Классические пивоты (Traditional) от предыдущего закрытого периода."""

    p: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    def as_pairs(self) -> list[tuple[str, float]]:
        return [
            ("S3", self.s3), ("S2", self.s2), ("S1", self.s1),
            ("P", self.p),
            ("R1", self.r1), ("R2", self.r2), ("R3", self.r3),
        ]

    def nearest(self, price: float) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
        """Ближайший пивот снизу и сверху от цены."""
        below = [x for x in self.as_pairs() if x[1] <= price]
        above = [x for x in self.as_pairs() if x[1] > price]
        return (max(below, key=lambda x: x[1]) if below else None,
                min(above, key=lambda x: x[1]) if above else None)


@dataclass(frozen=True)
class VolumeProfile:
    """Приближённый профиль: объём размазан по диапазону каждой свечи."""

    poc: float
    value_area_low: float
    value_area_high: float
    total_quote_volume: float
    bins: int

    def contains(self, price: float) -> bool:
        return self.value_area_low <= price <= self.value_area_high


def pivots(high: float, low: float, close: float) -> Pivots:
    """Пивоты от H/L/C предыдущего ЗАКРЫТОГО периода (PLAN §4.7)."""
    p = (high + low + close) / 3.0
    span = high - low
    return Pivots(
        p=p,
        r1=2 * p - low,
        s1=2 * p - high,
        r2=p + span,
        s2=p - span,
        r3=high + 2 * (p - low),
        s3=low - 2 * (high - p),
    )


def swing_points(
    high: np.ndarray,
    low: np.ndarray,
    left: int = 2,
    right: int = 2,
) -> list[Swing]:
    """Фрактальные экстремумы: точка выше (ниже) соседей слева и справа.

    ``right`` свечей справа обязательны для подтверждения, поэтому последние
    ``right`` баров экстремумом стать не могут — это не задержка реализации,
    а свойство определения: экстремум неизвестен, пока цена не отошла.
    """
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    out: list[Swing] = []
    if len(high) < left + right + 1:
        return out

    for i in range(left, len(high) - right):
        window_h = high[i - left : i + right + 1]
        if high[i] == window_h.max() and np.count_nonzero(window_h == high[i]) == 1:
            out.append(Swing(i, float(high[i]), "high"))
            continue
        window_l = low[i - left : i + right + 1]
        if low[i] == window_l.min() and np.count_nonzero(window_l == low[i]) == 1:
            out.append(Swing(i, float(low[i]), "low"))
    return out


def market_structure(swings: list[Swing]) -> Structure:
    """Структура по двум последним максимумам и двум последним минимумам.

    Вторая колонка снапшота из PLAN §4.4. Расхождение со столбцом EMA само по
    себе информативно, поэтому вердикты не смешиваются в один.
    """
    highs = [s.price for s in swings if s.kind == "high"]
    lows = [s.price for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "unknown"
    higher_high = highs[-1] > highs[-2]
    higher_low = lows[-1] > lows[-2]
    if higher_high and higher_low:
        return "HH/HL"
    if not higher_high and not higher_low:
        return "LH/LL"
    return "mixed"


def cluster_levels(
    swings: list[Swing],
    tolerance: float,
    *,
    min_touches: int = 2,
) -> list[Level]:
    """Слить близкие экстремумы в уровни.

    ``tolerance`` — предельная ШИРИНА кластера в цене; вызывающий код обычно
    берёт долю ATR, чтобы допуск был сопоставим с текущей волатильностью, а не
    с процентом, одинаковым для BTC и для альткоина.

    Новая точка сравнивается с началом кластера, а не с последней добавленной.
    Сравнение с последней даёт эффект сцепления: цепочка экстремумов, каждый из
    которых отстоит от соседа на пол-допуска, слипается в один «уровень»
    шириной в десяток ATR и с семьюдесятью касаниями. Такой уровень описывает
    весь диапазон и не значит ничего.
    """
    if not swings or tolerance <= 0:
        return []

    levels: list[Level] = []
    for kind in ("high", "low"):
        group = sorted(
            (s for s in swings if s.kind == kind), key=lambda s: s.price
        )
        if not group:
            continue
        bucket = [group[0]]
        for swing in group[1:]:
            if swing.price - bucket[0].price <= tolerance:
                bucket.append(swing)
            else:
                levels.append(_level_from(bucket, kind))
                bucket = [swing]
        levels.append(_level_from(bucket, kind))

    return sorted(
        (lv for lv in levels if lv.touches >= min_touches),
        key=lambda lv: lv.price,
    )


def _level_from(bucket: list[Swing], kind: SwingKind) -> Level:
    return Level(
        price=float(np.mean([s.price for s in bucket])),
        touches=len(bucket),
        last_touch_index=max(s.index for s in bucket),
        kind=kind,
    )


def nearest_levels(
    levels: list[Level], price: float
) -> tuple[Level | None, Level | None]:
    """Ближайшие уровни снизу (поддержка) и сверху (сопротивление)."""
    below = [lv for lv in levels if lv.price <= price]
    above = [lv for lv in levels if lv.price > price]
    return (max(below, key=lambda lv: lv.price) if below else None,
            min(above, key=lambda lv: lv.price) if above else None)


def volume_profile(
    high: np.ndarray,
    low: np.ndarray,
    quote_volume: np.ndarray,
    *,
    bins: int = 60,
    value_area_pct: float = 0.70,
) -> VolumeProfile | None:
    """Приближённый объёмный профиль в котируемой валюте (PLAN §4.1, §4.8).

    Объём каждой свечи распределяется равномерно по ценовым корзинам, которые
    она перекрывает. Value Area набирается жадно от POC в сторону соседа с
    большим объёмом — упрощение относительно метода пар, допустимое для метрики,
    которая и так помечена приближённой.
    """
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    vol = np.asarray(quote_volume, dtype="float64")

    valid = ~(np.isnan(high) | np.isnan(low) | np.isnan(vol))
    high, low, vol = high[valid], low[valid], vol[valid]
    if len(high) == 0 or bins < 2:
        return None

    lo, hi = float(low.min()), float(high.max())
    if hi <= lo:
        return None

    edges = np.linspace(lo, hi, bins + 1)
    histogram = np.zeros(bins, dtype="float64")
    bin_width = (hi - lo) / bins

    for candle_low, candle_high, candle_vol in zip(low, high, vol, strict=True):
        if candle_vol <= 0:
            continue
        first = min(int((candle_low - lo) / bin_width), bins - 1)
        last = min(int((candle_high - lo) / bin_width), bins - 1)
        histogram[first : last + 1] += candle_vol / (last - first + 1)

    total = histogram.sum()
    if total <= 0:
        return None

    poc_index = int(np.argmax(histogram))
    lower = upper = poc_index
    covered = histogram[poc_index]
    target = total * value_area_pct

    while covered < target and (lower > 0 or upper < bins - 1):
        below = histogram[lower - 1] if lower > 0 else -1.0
        above = histogram[upper + 1] if upper < bins - 1 else -1.0
        if above >= below:
            upper += 1
            covered += histogram[upper]
        else:
            lower -= 1
            covered += histogram[lower]

    centre = (edges[:-1] + edges[1:]) / 2.0
    return VolumeProfile(
        poc=float(centre[poc_index]),
        value_area_low=float(edges[lower]),
        value_area_high=float(edges[upper + 1]),
        total_quote_volume=float(total),
        bins=bins,
    )
