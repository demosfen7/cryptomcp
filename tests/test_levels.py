"""Тесты уровней, пивотов и объёмного профиля (PLAN §4.7, §4.8, §6.8)."""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.levels import (
    Level,
    cluster_levels,
    market_structure,
    nearest_levels,
    pivots,
    swing_points,
    volume_profile,
)


class TestPivots:
    def test_manual_calculation(self):
        """Сверка с ручным расчётом по формулам Traditional (PLAN §4.7)."""
        high, low, close = 110.0, 90.0, 100.0
        p = pivots(high, low, close)

        expected_p = (110 + 90 + 100) / 3  # 100.0
        assert p.p == pytest.approx(expected_p)
        assert p.r1 == pytest.approx(2 * expected_p - low)      # 110
        assert p.s1 == pytest.approx(2 * expected_p - high)     # 90
        assert p.r2 == pytest.approx(expected_p + (high - low)) # 120
        assert p.s2 == pytest.approx(expected_p - (high - low)) # 80
        assert p.r3 == pytest.approx(high + 2 * (expected_p - low))  # 130
        assert p.s3 == pytest.approx(low - 2 * (high - expected_p))  # 70

    def test_ordering_is_monotonic(self):
        p = pivots(115.0, 95.0, 108.0)
        values = [v for _, v in p.as_pairs()]
        assert values == sorted(values)

    def test_nearest_brackets_price(self):
        p = pivots(110.0, 90.0, 100.0)
        below, above = p.nearest(105.0)
        assert below[0] == "P" and below[1] == pytest.approx(100.0)
        assert above[0] == "R1" and above[1] == pytest.approx(110.0)

    def test_nearest_above_none_beyond_r3(self):
        p = pivots(110.0, 90.0, 100.0)
        below, above = p.nearest(999.0)
        assert above is None
        assert below[0] == "R3"

    def test_doji_period_collapses_levels(self):
        p = pivots(100.0, 100.0, 100.0)
        assert p.r1 == pytest.approx(100.0)
        assert p.s3 == pytest.approx(100.0)


class TestSwingPoints:
    def test_finds_obvious_peak(self):
        high = np.array([1.0, 2, 3, 9, 3, 2, 1])
        low = np.array([1.0, 2, 3, 9, 3, 2, 1])
        swings = swing_points(high, low, 2, 2)
        peaks = [s for s in swings if s.kind == "high"]
        assert len(peaks) == 1
        assert peaks[0].index == 3 and peaks[0].price == 9.0

    def test_finds_obvious_trough(self):
        high = np.array([9.0, 8, 7, 1, 7, 8, 9])
        low = np.array([9.0, 8, 7, 1, 7, 8, 9])
        troughs = [s for s in swing_points(high, low, 2, 2) if s.kind == "low"]
        assert len(troughs) == 1
        assert troughs[0].price == 1.0

    def test_last_bars_cannot_be_swings(self):
        """Экстремум не подтверждён, пока справа нет `right` свечей."""
        high = np.array([1.0, 2, 3, 4, 5, 6, 9])
        low = high.copy()
        swings = swing_points(high, low, 2, 2)
        assert all(s.index <= len(high) - 3 for s in swings)

    def test_flat_series_has_no_swings(self):
        flat = np.full(20, 5.0)
        assert swing_points(flat, flat, 2, 2) == []

    def test_too_short_series(self):
        assert swing_points(np.array([1.0, 2]), np.array([1.0, 2])) == []


class TestMarketStructure:
    def _swings(self, highs, lows):
        from cryptomcp.levels import Swing
        out = []
        for i, h in enumerate(highs):
            out.append(Swing(i * 2, h, "high"))
        for i, low in enumerate(lows):
            out.append(Swing(i * 2 + 1, low, "low"))
        return out

    def test_uptrend(self):
        assert market_structure(self._swings([10, 12], [8, 9])) == "HH/HL"

    def test_downtrend(self):
        assert market_structure(self._swings([12, 10], [9, 8])) == "LH/LL"

    def test_mixed_when_signals_disagree(self):
        # Более высокий максимум, но более низкий минимум — расширение.
        assert market_structure(self._swings([10, 12], [9, 8])) == "mixed"

    def test_unknown_without_enough_swings(self):
        assert market_structure(self._swings([10], [8])) == "unknown"
        assert market_structure([]) == "unknown"


class TestClusterLevels:
    def _highs(self, prices):
        from cryptomcp.levels import Swing
        return [Swing(i, p, "high") for i, p in enumerate(prices)]

    def test_merges_close_touches(self):
        levels = cluster_levels(self._highs([100.0, 100.4, 100.8]), tolerance=1.0)
        assert len(levels) == 1
        assert levels[0].touches == 3
        assert levels[0].price == pytest.approx(100.4)

    def test_separates_distant_touches(self):
        levels = cluster_levels(self._highs([100.0, 100.2, 130.0, 130.1]), tolerance=1.0)
        assert len(levels) == 2
        assert all(lv.touches == 2 for lv in levels)

    def test_single_touch_filtered_by_default(self):
        assert cluster_levels(self._highs([100.0, 130.0]), tolerance=1.0) == []

    def test_min_touches_can_be_relaxed(self):
        levels = cluster_levels(self._highs([100.0, 130.0]), tolerance=1.0, min_touches=1)
        assert len(levels) == 2

    def test_last_touch_index_is_most_recent(self):
        levels = cluster_levels(self._highs([100.0, 100.3]), tolerance=1.0)
        assert levels[0].last_touch_index == 1

    def test_zero_tolerance_returns_nothing(self):
        assert cluster_levels(self._highs([100.0, 100.0]), tolerance=0.0) == []

    def test_no_chaining_across_wide_span(self):
        """Цепочка близких точек не должна слипаться в один широкий уровень.

        Регрессия по живым данным: на SUIUSDT сравнение с последней точкой
        кластера, а не с первой, дало «уровень» с 70 касаниями, покрывающий весь
        диапазон. Ширина кластера обязана быть ограничена допуском.
        """
        prices = [100.0 + i * 0.6 for i in range(20)]  # шаг меньше допуска
        levels = cluster_levels(self._highs(prices), tolerance=1.0)

        # Шаг 0.6 при допуске 1.0 даёт пары: 100.0+100.6, затем 101.2+101.8, ...
        assert len(levels) == 10, "цепочка слиплась вместо разбиения на пары"
        assert all(lv.touches == 2 for lv in levels)
        # Ни один уровень не покрывает больше допуска — то, что ломалось на SUI.
        assert max(lv.price for lv in levels) - min(lv.price for lv in levels) > 1.0

    def test_cluster_width_never_exceeds_tolerance(self):
        prices = [100.0, 100.9, 101.7, 102.4]
        levels = cluster_levels(self._highs(prices), tolerance=1.0, min_touches=1)
        # 100.0+100.9 (ширина 0.9), затем 101.7+102.4 (ширина 0.7).
        assert len(levels) == 2
        assert [lv.touches for lv in levels] == [2, 2]


class TestNearestLevels:
    def test_brackets_price(self):
        levels = [
            Level(90.0, 3, 5, "low"),
            Level(110.0, 2, 7, "high"),
        ]
        support, resistance = nearest_levels(levels, 100.0)
        assert support.price == 90.0
        assert resistance.price == 110.0

    def test_distance_helpers(self):
        level = Level(110.0, 2, 7, "high")
        assert level.distance_pct(100.0) == pytest.approx(10.0)
        assert level.distance_atr(100.0, atr_value=5.0) == pytest.approx(2.0)

    def test_distance_atr_guards_zero(self):
        assert np.isnan(Level(110.0, 1, 0, "high").distance_atr(100.0, 0.0))


class TestVolumeProfile:
    def test_poc_at_concentration(self):
        # Весь объём сосредоточен в узкой полосе около 100.
        high = np.array([101.0] * 10 + [150.0] * 2)
        low = np.array([99.0] * 10 + [149.0] * 2)
        vol = np.array([1000.0] * 10 + [1.0] * 2)

        vp = volume_profile(high, low, vol, bins=50)
        assert vp is not None
        assert vp.poc == pytest.approx(100.0, abs=2.0)

    def test_value_area_contains_poc(self):
        rng = np.random.default_rng(42)
        centre = rng.normal(100, 2, 300)
        high, low = centre + 0.5, centre - 0.5
        vol = np.full(300, 100.0)

        vp = volume_profile(high, low, vol, bins=40)
        assert vp.contains(vp.poc)
        assert vp.value_area_low < vp.poc < vp.value_area_high

    def test_value_area_narrower_than_full_range(self):
        rng = np.random.default_rng(7)
        centre = rng.normal(100, 3, 500)
        vp = volume_profile(centre + 0.5, centre - 0.5, np.full(500, 10.0), bins=50)
        full_range = centre.max() - centre.min()
        assert (vp.value_area_high - vp.value_area_low) < full_range

    def test_total_volume_preserved(self):
        high = np.array([102.0, 103.0])
        low = np.array([100.0, 101.0])
        vol = np.array([500.0, 700.0])
        vp = volume_profile(high, low, vol, bins=20)
        assert vp.total_quote_volume == pytest.approx(1200.0)

    def test_returns_none_on_degenerate_input(self):
        assert volume_profile(np.array([]), np.array([]), np.array([])) is None
        flat = np.full(5, 100.0)
        assert volume_profile(flat, flat, np.full(5, 10.0)) is None

    def test_nan_rows_ignored(self):
        high = np.array([101.0, np.nan, 102.0])
        low = np.array([99.0, np.nan, 100.0])
        vol = np.array([10.0, np.nan, 20.0])
        vp = volume_profile(high, low, vol, bins=10)
        assert vp.total_quote_volume == pytest.approx(30.0)
