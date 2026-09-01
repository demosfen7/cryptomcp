"""Тесты нормализации ряда: отсечение незакрытой свечи и разрывы (PLAN §6.1–6.2)."""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.series import (
    CLOSE_GRACE_MS,
    INTERVAL_MS,
    build_series,
    count_missing,
    interval_ms,
)

H1 = INTERVAL_MS["1h"]
T0 = 1_700_000_000_000 - 1_700_000_000_000 % H1  # ровная граница часа


def kline(open_time: int, step: int = H1, *, close=100.0, quote_vol=1000.0, taker=600.0):
    """Одна свеча в формате ответа /fapi/v1/klines."""
    return [
        open_time,
        f"{close - 1:.8f}",
        f"{close + 2:.8f}",
        f"{close - 3:.8f}",
        f"{close:.8f}",
        "10.0",
        open_time + step - 1,
        f"{quote_vol:.8f}",
        42,
        "6.0",
        f"{taker:.8f}",
        "0",
    ]


def series_of(n: int, *, start: int = T0, step: int = H1):
    return [kline(start + i * step, step) for i in range(n)]


def bs(raw, symbol, interval, now_ms, *, grace_ms: int = 0):
    """build_series без запаса по умолчанию.

    Запас проверяется отдельным классом ниже; в остальных тестах он только
    смазывал бы границу закрытости, которую они как раз и проверяют.
    """
    return build_series(raw, symbol, interval, now_ms, grace_ms=grace_ms)


class TestUnclosedCandle:
    """Незакрытая свеча не должна доходить до индикаторов ни при каких условиях."""

    def test_last_candle_dropped_while_open(self):
        raw = series_of(5)
        # Время биржи — середина последней свечи: она ещё открыта.
        now = T0 + 4 * H1 + H1 // 2
        s = bs(raw, "BTCUSDT", "1h", now)

        assert len(s) == 4
        assert s.dropped_unclosed == 1
        assert s.closed_through_ms == T0 + 4 * H1 - 1

    def test_candle_kept_once_closed(self):
        raw = series_of(5)
        # Ровно на миллисекунду позже close_time последней свечи.
        now = T0 + 5 * H1
        s = bs(raw, "BTCUSDT", "1h", now)

        assert len(s) == 5
        assert s.dropped_unclosed == 0

    def test_boundary_exactly_at_close_time_is_not_closed_yet(self):
        raw = series_of(1)
        close_time = T0 + H1 - 1
        s = bs(raw, "BTCUSDT", "1h", close_time)

        # close_time включительный: пока now == close_time, свеча ещё идёт.
        assert len(s) == 0
        assert s.dropped_unclosed == 1

    def test_all_candles_open_gives_empty_series(self):
        raw = series_of(3)
        s = bs(raw, "BTCUSDT", "1h", T0)

        assert len(s) == 0
        assert s.dropped_unclosed == 3
        assert s.closed_through_ms == 0


class TestCloseGrace:
    """Запас после закрытия: страховка от дрейфа часов и доправки свечи биржей."""

    def test_candle_rejected_inside_grace_window(self):
        raw = series_of(5)
        # Свеча закрылась 1 мс назад — формально закрыта, но запас не истёк.
        s = bs(raw, "BTCUSDT", "1h", T0 + 5 * H1, grace_ms=CLOSE_GRACE_MS)

        assert len(s) == 4
        assert s.dropped_unclosed == 1

    def test_candle_accepted_after_grace(self):
        raw = series_of(5)
        s = bs(raw, "BTCUSDT", "1h", T0 + 5 * H1 + CLOSE_GRACE_MS,
               grace_ms=CLOSE_GRACE_MS)

        assert len(s) == 5
        assert s.dropped_unclosed == 0

    def test_grace_is_five_seconds_by_default(self):
        assert CLOSE_GRACE_MS == 5_000

    def test_grace_smaller_than_shortest_interval(self):
        """Запас не должен съедать целую свечу даже на минутках."""
        assert CLOSE_GRACE_MS < INTERVAL_MS["1m"]


class TestGaps:
    """Дыры от приостановки торгов ломают индикаторы молча (PLAN §6.2)."""

    def test_continuous_series_has_no_gaps(self):
        s = bs(series_of(10), "BTCUSDT", "1h", T0 + 10 * H1)
        assert s.missing == 0
        assert not s.has_gaps

    def test_single_missing_candle_counted(self):
        raw = series_of(10)
        del raw[4]
        s = bs(raw, "BTCUSDT", "1h", T0 + 10 * H1)

        assert s.missing == 1
        assert s.has_gaps

    def test_multi_candle_hole_counted_by_length(self):
        raw = series_of(10)
        del raw[3:7]  # выпали четыре подряд
        s = bs(raw, "BTCUSDT", "1h", T0 + 10 * H1)

        assert s.missing == 4

    def test_short_series_has_no_gaps(self):
        assert count_missing(bs(series_of(1), "B", "1h", T0 + H1).df, "1h") == 0


class TestParsing:
    def test_numeric_columns_are_floats(self):
        s = bs(series_of(3), "btcusdt", "1h", T0 + 3 * H1)

        assert s.symbol == "BTCUSDT"
        assert s.df["close"].dtype == np.dtype("float64")
        assert s.df["quote_volume"].dtype == np.dtype("float64")
        assert s.df["open_time"].dtype == np.dtype("int64")

    def test_taker_buy_ratio(self):
        raw = [kline(T0, quote_vol=1000.0, taker=600.0)]
        s = bs(raw, "BTCUSDT", "1h", T0 + H1)

        assert s.taker_buy_ratio[0] == pytest.approx(0.6)

    def test_taker_buy_ratio_nan_on_zero_volume(self):
        raw = [kline(T0, quote_vol=0.0, taker=0.0)]
        s = bs(raw, "BTCUSDT", "1h", T0 + H1)

        assert np.isnan(s.taker_buy_ratio[0])

    def test_duplicates_dropped_and_order_restored(self):
        raw = series_of(5)
        shuffled = [raw[3], raw[0], raw[1], raw[3], raw[2], raw[4]]
        s = bs(shuffled, "BTCUSDT", "1h", T0 + 5 * H1)

        assert len(s) == 5
        assert list(s.df["open_time"]) == sorted(s.df["open_time"])

    def test_empty_response(self):
        s = bs([], "BTCUSDT", "1h", T0)
        assert len(s) == 0
        assert s.span_days == 0.0

    def test_span_days(self):
        # Охват считается от открытия первой свечи до закрытия последней,
        # поэтому 24 часовые свечи дают ровно сутки.
        s = bs(series_of(24), "BTCUSDT", "1h", T0 + 24 * H1)
        assert s.span_days == pytest.approx(1.0, abs=0.001)

    def test_tail_preserves_metadata(self):
        s = bs(series_of(50), "BTCUSDT", "1h", T0 + 50 * H1)
        t = s.tail(10)

        assert len(t) == 10
        assert t.symbol == s.symbol
        assert t.closed_through_ms == s.closed_through_ms


class TestIntervals:
    def test_known_interval(self):
        assert interval_ms("4h") == 4 * 3600 * 1000

    def test_unknown_interval_rejected(self):
        with pytest.raises(ValueError, match="Неподдерживаемый интервал"):
            interval_ms("7h")

    def test_month_not_supported_variable_length(self):
        # 1M имеет непостоянную длину, проверка непрерывности по шагу неприменима.
        assert "1M" not in INTERVAL_MS
