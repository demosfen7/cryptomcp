"""Тесты сезонного baseline объёма (PLAN §4.10)."""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.series import INTERVAL_MS, build_series
from cryptomcp.volume import (
    MIN_SAMPLES_PER_SLOT,
    MIN_SAMPLES_TO_SCORE,
    seasonal_baseline,
    slot_of_day,
    volume_context,
)

H4 = INTERVAL_MS["4h"]
D1 = INTERVAL_MS["1d"]
T0 = 0  # полночь UTC, слот 0


def kline(open_time: int, step: int, quote_vol: float, *, close=100.0, taker=None):
    taker = quote_vol * 0.5 if taker is None else taker
    return [
        open_time, f"{close:.8f}", f"{close + 1:.8f}", f"{close - 1:.8f}",
        f"{close:.8f}", "1.0", open_time + step - 1, f"{quote_vol:.8f}", 10,
        "0.5", f"{taker:.8f}", "0",
    ]


def seasonal_series(days: int, *, day_volumes: list[float], step: int = H4):
    """Ряд с заложенной суточной сезонностью: объём зависит от слота суток."""
    per_day = 86_400_000 // step
    assert len(day_volumes) == per_day
    raw = []
    for day in range(days):
        for slot in range(per_day):
            raw.append(kline(T0 + (day * per_day + slot) * step, step, day_volumes[slot]))
    return raw


class TestSlotOfDay:
    def test_four_hour_slots(self):
        times = np.array([0, H4, 2 * H4, 5 * H4, 6 * H4], dtype="int64")
        assert list(slot_of_day(times, "4h")) == [0, 1, 2, 5, 0]

    def test_daily_has_single_slot(self):
        times = np.array([0, D1, 2 * D1], dtype="int64")
        assert list(slot_of_day(times, "1d")) == [0, 0, 0]


class TestSeasonalBaseline:
    """Ночная свеча не должна читаться как затухание объёма."""

    def test_quiet_slot_compared_with_its_own_history(self):
        # Тихий слот — последний в сутках: 30*6 свечей дают индекс 179, 179%6=5.
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]
        raw = seasonal_series(30, day_volumes=volumes)
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        baseline, basis, samples = seasonal_baseline(s)

        assert baseline == pytest.approx(100.0)
        assert "слот" in basis
        assert samples == 29  # 30 суток минус текущая свеча

    def test_level_drift_does_not_fake_a_spike(self):
        """Рост общего уровня объёма не должен читаться как всплеск.

        Регрессия по живым данным: у SUIUSDT медиана слота в первой половине
        60-суточного окна была 1.97M, во второй 3.16M, и обычная свеча
        показывала 4x. База обязана следовать за уровнем, а не за его прошлым.
        """
        quiet = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]
        loud = [v * 3 for v in quiet]  # режим втрое активнее

        raw = seasonal_series(30, day_volumes=quiet)
        base_time = raw[-1][0] + H4
        for day in range(10):
            for slot in range(6):
                raw.append(kline(base_time + (day * 6 + slot) * H4, H4, loud[slot]))

        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)
        ctx = volume_context(s, np.full(len(s), 1.0))

        # Свеча типична для нового режима и своего слота — значит около 1.0x.
        assert ctx.ratio == pytest.approx(1.0, abs=0.15)

    def test_ratio_is_one_for_typical_night_candle(self):
        """Главная проверка: тихая ночь даёт 1.0x, а не 0.1x."""
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]
        raw = seasonal_series(30, day_volumes=volumes)
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        ctx = volume_context(s, np.full(len(s), 1.0))
        assert ctx.ratio == pytest.approx(1.0, abs=0.01)

    def test_flat_ma_would_have_been_misleading(self):
        """Контрольный расчёт: плоская средняя показала бы ложное затухание."""
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]
        raw = seasonal_series(30, day_volumes=volumes)
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        flat_ratio = s.quote_volume[-1] / np.mean(s.quote_volume[-21:-1])
        assert flat_ratio < 0.2  # «затухание в пять раз», которого нет

    def test_real_volume_spike_still_detected(self):
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 100.0]
        raw = seasonal_series(30, day_volumes=volumes)
        raw[-1] = kline(raw[-1][0], H4, 500.0)  # ночь, но объём впятеро выше нормы
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        ctx = volume_context(s, np.full(len(s), 1.0))
        assert ctx.ratio == pytest.approx(5.0, abs=0.1)

    def test_daily_uses_rolling_median_not_slots(self):
        raw = [kline(T0 + i * D1, D1, 500.0) for i in range(40)]
        s = build_series(raw, "T", "1d", raw[-1][6] + 10_000, grace_ms=0)

        _, basis, _ = seasonal_baseline(s)
        assert "слот" not in basis
        assert "медиана" in basis

    def test_empty_series(self):
        s = build_series([], "T", "4h", 0)
        baseline, basis, samples = seasonal_baseline(s)
        assert np.isnan(baseline) and samples == 0


class TestBasisStrength:
    def test_weak_basis_flagged(self):
        volumes = [100.0] * 6
        raw = seasonal_series(6, day_volumes=volumes)  # 5 наблюдений на слот
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        ctx = volume_context(s, np.full(len(s), 1.0))
        assert ctx.weak_basis
        assert ctx.scorable  # 5 наблюдений — ещё можно скорить

    def test_not_scorable_below_minimum(self):
        volumes = [100.0] * 6
        raw = seasonal_series(3, day_volumes=volumes)  # 2 наблюдения на слот
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        ctx = volume_context(s, np.full(len(s), 1.0))
        assert not ctx.scorable

    def test_thresholds_ordered(self):
        assert MIN_SAMPLES_TO_SCORE < MIN_SAMPLES_PER_SLOT


class TestAnomalousBars:
    def test_high_volume_small_move_counted(self):
        step = H4
        raw = [kline(T0 + i * step, step, 100.0) for i in range(60)]
        # Бар с объёмом в пять раз выше при том же нулевом движении цены.
        raw[-5] = kline(raw[-5][0], step, 500.0)
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        ctx = volume_context(s, np.full(len(s), 10.0))
        assert ctx.anomalous_bars == 1

    def test_high_volume_with_big_move_not_counted(self):
        step = H4
        raw = [kline(T0 + i * step, step, 100.0) for i in range(60)]
        big = kline(raw[-5][0], step, 500.0)
        big[4] = "120.00000000"  # close сильно выше open
        raw[-5] = big
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

        # ATR = 10, движение 20 — это больше половины ATR, не накопление.
        ctx = volume_context(s, np.full(len(s), 10.0))
        assert ctx.anomalous_bars == 0

    def test_short_series_returns_zero(self):
        raw = [kline(T0 + i * H4, H4, 100.0) for i in range(10)]
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)
        ctx = volume_context(s, np.full(len(s), 1.0))
        assert ctx.anomalous_bars == 0


class TestTakerBuy:
    def test_mean_ratio(self):
        raw = [kline(T0 + i * H4, H4, 100.0, taker=70.0) for i in range(40)]
        s = build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)
        ctx = volume_context(s, np.full(len(s), 1.0))
        assert ctx.taker_buy_mean == pytest.approx(0.7)


class TestAbsorption:
    """Разрешение по тейкерам: средняя прячет то, ради чего блок и заведён.

    У ASTER 19.08 дневная средняя 0.48 при часах по 0.57, 0.60 и 0.72.
    """

    def series(self, taker):
        import numpy as np

        from cryptomcp.series import INTERVAL_MS, build_series

        step = INTERVAL_MS["1h"]
        raw = []
        for i, value in enumerate(taker):
            quote = 1000.0
            raw.append([
                i * step, "100.0", "101.0", "99.0", "100.0", "10.0",
                (i + 1) * step - 1, f"{quote}", 10,
                f"{5.0}", f"{quote * value}", "0",
            ])
        return build_series(raw, "TESTUSDT", "1h", len(taker) * step + 10_000,
                            grace_ms=0), np.zeros(len(taker))

    def test_mean_hides_the_peaks(self):
        from cryptomcp.volume import absorption

        taker = [0.40] * 26 + [0.57, 0.60, 0.72, 0.45]
        series, atr_values = self.series(taker)
        data = absorption(series, atr_values)

        assert data.taker_mean < 0.50, "средняя действительно ничего не показывает"
        assert data.taker_max == pytest.approx(0.72, abs=0.01)
        assert data.taker_above == 3

    def test_streak_counts_consecutive_only(self):
        from cryptomcp.volume import absorption

        taker = [0.60, 0.60, 0.40] + [0.55, 0.56, 0.57, 0.58] + [0.40] * 23
        series, atr_values = self.series(taker)

        assert absorption(series, atr_values).taker_streak == 4

    def test_interval_is_reported(self):
        from cryptomcp.volume import absorption

        series, atr_values = self.series([0.5] * 30)
        assert absorption(series, atr_values).interval == "1h"
