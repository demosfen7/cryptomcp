"""Тесты индикаторов и правила достаточности базы перцентиля (PLAN §4.2, §6.8).

Ожидаемые значения выводятся независимой арифметикой прямо в тестах, а не
берутся из внешних таблиц. Причина конкретная: широко цитируемое эталонное
значение RSI(14) = 70.53 для контрольного ряда Уайлдера к этим данным не
сходится — ручной пересчёт даёт 70.46. Самопроверяемый тест надёжнее ссылки
на источник, который не удаётся воспроизвести.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.indicators import (
    MIN_PERCENTILE_OBS,
    Metric,
    atr,
    bollinger_width,
    consecutive_below,
    consecutive_declining,
    donchian_width,
    ema,
    percentile_rank,
    rolling_std,
    rsi,
    sma,
    true_range,
    with_percentile,
)

# Классический контрольный ряд Уайлдера для RSI(14).
WILDER_CLOSES = np.array([
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
])


class TestSMA:
    def test_known_values(self):
        out = sma(np.array([1.0, 2, 3, 4, 5]), 3)
        assert np.isnan(out[:2]).all()
        assert out[2] == pytest.approx(2.0)
        assert out[4] == pytest.approx(4.0)

    def test_too_short(self):
        assert np.isnan(sma(np.array([1.0, 2]), 5)).all()


class TestRollingStd:
    def test_matches_numpy_population_std(self):
        values = np.array([2.0, 4, 4, 4, 5, 5, 7, 9])
        out = rolling_std(values, 4)
        assert out[3] == pytest.approx(np.std(values[:4]))
        assert out[7] == pytest.approx(np.std(values[4:8]))

    def test_constant_series_has_zero_std(self):
        out = rolling_std(np.full(10, 5.0), 5)
        assert out[9] == pytest.approx(0.0, abs=1e-9)


class TestEMA:
    def test_seeded_with_sma(self):
        values = np.arange(1.0, 11.0)
        out = ema(values, 5)
        assert out[4] == pytest.approx(3.0)  # SMA(1..5)

    def test_converges_on_constant_series(self):
        out = ema(np.full(50, 7.0), 10)
        assert out[-1] == pytest.approx(7.0)


class TestRSI:
    def test_wilder_reference_value(self):
        """Первое значение RSI(14) — seed по Уайлдеру, выведен здесь же.

        Ожидаемое число считается из тех же данных независимой арифметикой, а не
        берётся из книжной таблицы: ходящая по сети цифра 70.53 к этому ряду не
        сходится, а самопроверяемый тест надёжнее ссылки на источник.
        """
        deltas = np.diff(WILDER_CLOSES[:15])
        avg_gain = np.where(deltas > 0, deltas, 0.0).sum() / 14
        avg_loss = np.where(deltas < 0, -deltas, 0.0).sum() / 14
        expected = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

        assert expected == pytest.approx(70.46, abs=0.01)
        assert rsi(WILDER_CLOSES, 14)[14] == pytest.approx(expected, abs=1e-9)

    def test_undefined_before_seed(self):
        out = rsi(WILDER_CLOSES, 14)
        assert np.isnan(out[:14]).all()

    def test_monotonic_rise_gives_100(self):
        out = rsi(np.arange(1.0, 40.0), 14)
        assert out[-1] == pytest.approx(100.0)

    def test_monotonic_fall_approaches_zero(self):
        out = rsi(np.arange(40.0, 1.0, -1.0), 14)
        assert out[-1] == pytest.approx(0.0, abs=1e-6)

    def test_flat_series_is_neutral(self):
        out = rsi(np.full(40, 100.0), 14)
        assert out[-1] == pytest.approx(50.0)

    def test_too_short_is_all_nan(self):
        assert np.isnan(rsi(np.arange(5.0), 14)).all()


class TestTrueRangeATR:
    def test_true_range_uses_prev_close(self):
        high = np.array([10.0, 12.0])
        low = np.array([9.0, 11.5])
        close = np.array([9.5, 12.0])
        tr = true_range(high, low, close)
        # Вторая свеча: гэп вверх, TR = high - prev_close = 12 - 9.5 = 2.5
        assert tr[1] == pytest.approx(2.5)

    def test_atr_constant_range(self):
        n = 40
        high = np.full(n, 11.0)
        low = np.full(n, 10.0)
        close = np.full(n, 10.5)
        out = atr(high, low, close, 14)
        assert out[-1] == pytest.approx(1.0, abs=1e-9)

    def test_atr_undefined_before_seed(self):
        out = atr(np.arange(20.0) + 2, np.arange(20.0), np.arange(20.0) + 1, 14)
        assert np.isnan(out[:13]).all()
        assert not np.isnan(out[13])


class TestBollingerWidth:
    def test_zero_on_flat_series(self):
        out = bollinger_width(np.full(30, 100.0), 20, 2.0)
        assert out[-1] == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        values = np.array([2.0, 4, 4, 4, 5, 5, 7, 9] * 3, dtype="float64")
        out = bollinger_width(values, 8, 2.0)
        window = values[16:24]
        expected = 4.0 * np.std(window) / window.mean()
        assert out[23] == pytest.approx(expected)

    def test_widens_with_volatility(self):
        calm = np.concatenate([np.full(25, 100.0)])
        wild = np.array([100.0 + (10 if i % 2 else -10) for i in range(25)])
        assert bollinger_width(wild, 20)[-1] > bollinger_width(calm, 20)[-1]


class TestDonchianWidth:
    def test_range_normalised_to_price(self):
        high = np.array([105.0] * 20)
        low = np.array([95.0] * 20)
        close = np.array([100.0] * 20)
        out = donchian_width(high, low, 20, close)
        assert out[-1] == pytest.approx(0.10)


class TestDurations:
    def test_consecutive_below(self):
        values = np.array([5.0, 5, 1, 1, 1])
        assert consecutive_below(values, threshold=2.0) == 3

    def test_consecutive_below_stops_at_nan(self):
        values = np.array([np.nan, 1.0, 1.0])
        assert consecutive_below(values, threshold=2.0) == 2

    def test_consecutive_below_zero_when_last_above(self):
        assert consecutive_below(np.array([1.0, 1.0, 9.0]), 2.0) == 0

    def test_consecutive_declining(self):
        assert consecutive_declining(np.array([1.0, 5, 4, 3, 2])) == 3

    def test_consecutive_declining_ignores_nan_prefix(self):
        assert consecutive_declining(np.array([np.nan, np.nan, 9.0, 5.0])) == 1


class TestPercentileRank:
    def test_uniform_distribution(self):
        history = np.arange(0.0, 100.0)
        assert percentile_rank(history, 49.5) == pytest.approx(50.0, abs=1.0)

    def test_minimum_and_maximum(self):
        history = np.arange(0.0, 100.0)
        assert percentile_rank(history, -1) == pytest.approx(0.0)
        assert percentile_rank(history, 1000) == pytest.approx(100.0)

    def test_ties_count_as_half(self):
        history = np.array([1.0, 1.0, 1.0, 1.0])
        assert percentile_rank(history, 1.0) == pytest.approx(50.0)

    def test_nan_history_ignored(self):
        history = np.array([1.0, np.nan, 2.0, 3.0])
        assert percentile_rank(history, 2.0) == pytest.approx(50.0, abs=0.1)


class TestPercentileGate:
    """Правило §4.2: перцентиль требует и количества, и календарного охвата."""

    def test_ok_when_base_is_sufficient(self):
        history = np.arange(0.0, 360.0)
        m = with_percentile("BBW", 30.0, history, span_days=90.0)
        assert m.has_context
        assert m.pct_rank == pytest.approx(8.3, abs=0.2)
        assert m.n_obs == 360

    def test_rejected_when_too_few_observations(self):
        history = np.arange(0.0, 14.0)
        m = with_percentile("BBW", 3.0, history, span_days=365.0)
        assert not m.has_context
        assert "наблюдений 14" in m.base_note

    def test_rejected_when_span_too_short(self):
        # Наблюдений достаточно, но это 3.7 суток пятнадцатиминуток.
        history = np.arange(0.0, 360.0)
        m = with_percentile("BBW", 30.0, history, span_days=3.7)
        assert not m.has_context
        assert "охват 3.7" in m.base_note

    def test_span_exemption_allows_short_window(self):
        """Исключение только там, где данных глубже нет у биржи (OI, 30 суток)."""
        history = np.arange(0.0, 180.0)
        m = with_percentile(
            "OI", 30.0, history, span_days=30.0,
            span_exemption="история OI ограничена биржей 30 сутками",
        )
        assert m.has_context
        assert m.base_note == "история OI ограничена биржей 30 сутками"

    def test_min_obs_constant_is_sixty(self):
        assert MIN_PERCENTILE_OBS == 60


class TestMetricRendering:
    def test_flag_below_threshold(self):
        m = Metric("BBW", 0.04, pct_rank=8.0, n_obs=360, span_days=60,
                   threshold=20, threshold_side="below")
        assert m.flagged

    def test_no_flag_above_threshold(self):
        m = Metric("BBW", 0.04, pct_rank=31.0, n_obs=360, span_days=60,
                   threshold=20, threshold_side="below")
        assert not m.flagged

    def test_flag_above_side(self):
        m = Metric("funding", 0.01, pct_rank=95.0, n_obs=900, span_days=900,
                   threshold=90, threshold_side="above")
        assert m.flagged

    def test_never_flags_without_context(self):
        """Без базы сравнения порог не срабатывает — иначе ложная уверенность."""
        m = Metric("BBW", 0.04, pct_rank=None, threshold=20, threshold_side="below")
        assert not m.flagged

    def test_dict_carries_base_size(self):
        m = with_percentile("BBW", 30.0, np.arange(0.0, 360.0), span_days=90.0)
        payload = m.to_dict()
        assert payload["n_obs"] == 360
        assert "span_days" in payload

    def test_dict_explains_missing_context(self):
        m = with_percentile("BBW", 30.0, np.arange(0.0, 10.0), span_days=1.0)
        payload = m.to_dict()
        assert "pct_rank" not in payload
        assert "context" in payload
