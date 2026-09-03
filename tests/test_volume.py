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


class TestAbsorptionWindow:
    """Окно набора меряется сутками, а не свечами (PLAN §4.25).

    Постоянные тридцать свечей означали тридцать часов на 1h и тридцать суток
    на 1d — разные вопросы под одним именем. Кейс, на котором это поймано:
    ASTER 1h на 19.08.2026, фаза набора началась в 51 свече назад и в окно не
    попадала вовсе, из-за чего в выдаче стояло «баров набора 0».
    """

    def test_window_follows_calendar_not_candle_count(self):
        from cryptomcp.volume import absorption_window

        assert absorption_window("1h") == 72       # трое суток
        assert absorption_window("4h") == 30       # трое суток мало, пол снизу
        assert absorption_window("1d") == 30       # то же
        assert absorption_window("5m") == 120      # потолок сверху

    def test_bar_beyond_thirty_candles_is_found_on_1h(self):
        """Тот самый бар, который старое окно не доставало."""
        from cryptomcp.volume import anomalous_bars

        step = INTERVAL_MS["1h"]
        raw = [kline(T0 + i * step, step, 100.0) for i in range(120)]
        # Пятидесятая свеча с конца: внутри трёх суток, вне тридцати часов.
        raw[-50] = kline(raw[-50][0], step, 500.0)
        s = build_series(raw, "T", "1h", raw[-1][6] + 10_000, grace_ms=0)
        atr_values = np.full(len(s), 10.0)

        assert anomalous_bars(s, atr_values) == 1
        assert anomalous_bars(s, atr_values, window=30) == 0


class TestQuietBarFloor:
    """Тело сравнивается с максимумом из 0.5 ATR и 0.3% цены (PLAN §4.25).

    Порог, привязанный только к ATR, в сжатии падает вместе с ним и признак
    исчезает ровно там, где он нужен.
    """

    def series(self, body: float, volume: float = 500.0):
        step = H4
        raw = [kline(T0 + i * step, step, 100.0) for i in range(60)]
        row = kline(raw[-5][0], step, volume)
        row[4] = f"{100.0 + body:.8f}"
        row[2] = f"{100.0 + body + 1:.8f}"
        raw[-5] = row
        return build_series(raw, "T", "4h", raw[-1][6] + 10_000, grace_ms=0)

    def test_small_body_counted_when_atr_is_compressed(self):
        from cryptomcp.volume import anomalous_bars

        # ATR 0.2 при цене 100 — половина ATR это 0.1, тело 0.2 её превышает.
        # Абсолютный потолок 0.3% = 0.3 и делает бар набором, как и должно.
        s = self.series(body=0.2)
        assert anomalous_bars(s, np.full(len(s), 0.2)) == 1

    def test_body_above_both_thresholds_not_counted(self):
        from cryptomcp.volume import anomalous_bars

        s = self.series(body=0.4)
        assert anomalous_bars(s, np.full(len(s), 0.2)) == 0

    def test_floor_changes_nothing_when_atr_is_normal(self):
        """У ликвидной монеты 0.5 ATR и так больше 0.3% — потолок молчит."""
        from cryptomcp.volume import anomalous_bars

        s = self.series(body=0.4)
        assert anomalous_bars(s, np.full(len(s), 10.0)) == 1


class TestVolumeLeadsPrice:
    """Признак, отличающий набор позиции от реакции на событие.

    Замерено на живых данных: ASTER 19.08 даёт +29 свечей (объём пришёл
    заранее), UAI 02.09 — +4 (объём догонял цену). Без требования «свеча
    всплеска тихая» оба давали близкие положительные числа, и признак не
    различал случаи вовсе.

    Ряды в тестах длиннее окна: окно с §4.25 задаётся сутками, и на 1h это 72
    свечи, а не 30.
    """

    def series(self, bars):
        """bars: список (объём, тело в долях ATR). ATR фиксирован."""
        import numpy as np

        from cryptomcp.series import INTERVAL_MS, build_series

        step = INTERVAL_MS["1h"]
        atr_value, raw = 1.0, []
        for i, (volume, body) in enumerate(bars):
            close = 100.0 + body * atr_value
            raw.append([
                i * step, "100.0", f"{max(100.0, close) + 0.5}",
                f"{min(100.0, close) - 0.5}", f"{close}", "10.0",
                (i + 1) * step - 1, f"{volume}", 10, "5.0", f"{volume / 2}", "0",
            ])
        series = build_series(raw, "TESTUSDT", "1h", len(bars) * step + 10_000,
                              grace_ms=0)
        return series, np.full(len(bars), atr_value)

    def test_quiet_spike_before_move_is_accumulation(self):
        from cryptomcp.volume import volume_leads_price

        bars = [(1000.0, 0.0)] * 87 + [(5000.0, 0.1)] + [(1000.0, 0.0)] * 5
        bars += [(1200.0, 2.0)] + [(1000.0, 0.0)] * 3
        series, atr_values = self.series(bars)

        lead, state = volume_leads_price(series, atr_values)
        assert lead == 6
        assert "опередил" in state

    def test_loud_spike_is_not_accumulation(self):
        """Всплеск объёма СВОИМ телом — это реакция, а не набор. Кейс UAI."""
        from cryptomcp.volume import volume_leads_price

        bars = [(1000.0, 0.0)] * 87 + [(5000.0, 2.5)] + [(1000.0, 0.0)] * 9
        series, atr_values = self.series(bars)

        lead, state = volume_leads_price(series, atr_values)
        assert lead is None
        assert "был движением цены" in state

    def test_spike_without_resolution_reports_waiting(self):
        """Самое интересное состояние: набор был, развязки ещё нет."""
        from cryptomcp.volume import volume_leads_price

        bars = [(1000.0, 0.0)] * 87 + [(5000.0, 0.1)] + [(1000.0, 0.0)] * 9
        series, atr_values = self.series(bars)

        lead, state = volume_leads_price(series, atr_values)
        assert lead == 9
        assert "движения ещё не было" in state

    def test_nothing_happened(self):
        from cryptomcp.volume import volume_leads_price

        series, atr_values = self.series([(1000.0, 0.0)] * 97)
        lead, state = volume_leads_price(series, atr_values)
        assert lead is None
        assert "ни всплеска" in state

    def test_plural_forms_are_readable(self):
        from cryptomcp.volume import candles

        assert candles(1) == "1 свечу"
        assert candles(3) == "3 свечи"
        assert candles(8) == "8 свечей"
        assert candles(11) == "11 свечей"
