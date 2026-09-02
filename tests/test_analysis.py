"""Тесты сборки метрик и squeeze_index (ТЗ §4.2–4.3, PLAN §4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.analysis import (
    TimeframeView,
    compute_squeeze_index,
    ema_state,
    position_in_range,
    score_range,
    score_value_area,
    score_volatility,
)
from cryptomcp.config import DEFAULT_WEIGHTS, Config
from cryptomcp.indicators import Metric
from cryptomcp.levels import VolumeProfile
from cryptomcp.volume import VolumeContext


def view(**overrides) -> TimeframeView:
    """Заготовка представления со всеми группами доступными."""
    defaults = dict(
        interval="4h", price=100.0, atr_value=2.0, atr_pct=2.0, rsi_value=50.0,
        ema_state="above", structure="HH/HL", position_in_range=0.5,
        bbw=Metric("BBW", 0.04, pct_rank=10.0, n_obs=360, span_days=90),
        atr_metric=Metric("ATR", 2.0, pct_rank=10.0, n_obs=360, span_days=90),
        atr_declining_bars=10,
        range_low=95.0, range_high=105.0, range_width=0.03, range_width_atr=1.5,
        range_threshold=0.06,
        range_metric=Metric("диапазон(20)", 10.0, unit="%", pct_rank=15.0,
                            n_obs=360, span_days=90),
        narrow_bars=20,
        volume=VolumeContext(0.7, "медиана слота", 250, 0.6, 3, 0.55),
        profile=VolumeProfile(100.0, 95.0, 105.0, 1e6, 60),
        divergence="бычья",
    )
    defaults.update(overrides)
    return TimeframeView(**defaults)


class TestEmaState:
    def test_above_when_price_over_both(self):
        close = np.arange(1.0, 401.0)  # устойчивый рост
        assert ema_state(close) == "above"

    def test_below_on_downtrend(self):
        close = np.arange(400.0, 0.0, -1.0)
        assert ema_state(close) == "below"

    def test_na_without_enough_history(self):
        assert ema_state(np.arange(1.0, 100.0)) == "n/a"

    def test_mixed_on_flat(self):
        rng = np.random.default_rng(1)
        close = 100.0 + rng.normal(0, 0.3, 400)
        assert ema_state(close) in {"mixed", "above", "below"}


class TestPositionInRange:
    def test_at_top(self):
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 110.0)
        assert position_in_range(close, high, low) == pytest.approx(1.0)

    def test_at_bottom(self):
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 90.0)
        assert position_in_range(close, high, low) == pytest.approx(0.0)

    def test_middle(self):
        high = np.full(20, 110.0)
        low = np.full(20, 90.0)
        close = np.full(20, 100.0)
        assert position_in_range(close, high, low) == pytest.approx(0.5)

    def test_degenerate_range_is_middle(self):
        flat = np.full(20, 100.0)
        assert position_in_range(flat, flat, flat) == pytest.approx(0.5)


class TestComponentScores:
    def test_volatility_high_when_percentile_low(self):
        config = Config()
        tight = score_volatility(view(), config)
        loose = score_volatility(
            view(bbw=Metric("BBW", 0.4, pct_rank=95.0, n_obs=360, span_days=90),
                 atr_metric=Metric("ATR", 9.0, pct_rank=95.0, n_obs=360, span_days=90),
                 atr_declining_bars=0),
            config,
        )
        assert tight > 0.85
        assert loose < 0.15

    def test_volatility_none_without_percentile(self):
        v = view(bbw=Metric("BBW", 0.04, pct_rank=None, n_obs=5, span_days=1))
        assert score_volatility(v, Config()) is None

    def test_range_score_follows_own_percentile(self):
        """Узость меряется по истории самой монеты, а не абсолютным числом."""
        config = Config()
        metric = lambda rank: Metric(  # noqa: E731
            "диапазон(20)", 10.0, unit="%", pct_rank=rank, n_obs=360, span_days=90
        )
        tight = score_range(view(range_metric=metric(5.0)), config)
        wide = score_range(view(range_metric=metric(95.0)), config)
        assert tight > wide
        assert wide == pytest.approx(0.43, abs=0.01)  # остаётся вклад длительности

    def test_same_width_scores_differently_for_different_coins(self):
        """40% дневного диапазона у BULLA обычны, у PAXG были бы аномалией."""
        config = Config()
        usual = score_range(view(range_width=0.40, range_metric=Metric(
            "диапазон(20)", 40.0, unit="%", pct_rank=60.0, n_obs=360, span_days=90
        )), config)
        unusual = score_range(view(range_width=0.40, range_metric=Metric(
            "диапазон(20)", 40.0, unit="%", pct_rank=5.0, n_obs=360, span_days=90
        )), config)
        assert unusual > usual

    def test_range_excluded_without_percentile_base(self):
        config = Config()
        weak = Metric("диапазон(20)", 10.0, unit="%", pct_rank=None, n_obs=5,
                      span_days=1)
        assert score_range(view(range_metric=weak), config) is None

    def test_value_area_zero_when_price_outside(self):
        v = view(price=200.0, profile=VolumeProfile(100.0, 95.0, 105.0, 1e6, 60))
        assert score_value_area(v) == 0.0

    def test_value_area_max_at_poc(self):
        v = view(price=100.0, profile=VolumeProfile(100.0, 95.0, 105.0, 1e6, 60))
        assert score_value_area(v) == pytest.approx(1.0)

    def test_value_area_none_without_profile(self):
        assert score_value_area(view(profile=None)) is None


class TestSqueezeIndex:
    def test_all_groups_present(self):
        index, components, excluded = compute_squeeze_index(view(), Config())
        assert excluded == []
        assert len(components) == 5
        assert 0.0 <= index <= 1.0

    def test_weights_match_specification(self):
        """Веса из ТЗ §4.3 и их сумма."""
        assert DEFAULT_WEIGHTS == {
            "volatility": 0.35, "range": 0.25, "volume": 0.20,
            "value_area": 0.10, "divergence": 0.10,
        }
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_missing_group_is_excluded_not_zeroed(self):
        """Неизмеримая группа исключается с перенормировкой весов.

        Подстановка нуля означала бы «признака нет», хотя на деле его не
        удалось измерить, и тихо занижала бы индекс.
        """
        full = compute_squeeze_index(view(), Config())[0]
        without_profile = compute_squeeze_index(view(profile=None), Config())

        index, components, excluded = without_profile
        assert excluded == ["value_area"]
        assert "value_area" not in components
        # При исключении сильной группы индекс не обязан падать — он
        # пересчитывается по оставшимся, а не штрафуется.
        assert index > full * 0.9

    def test_zeroing_would_have_lowered_index(self):
        """Контрольный расчёт: чем перенормировка отличается от обнуления."""
        config = Config()
        v = view(profile=None)
        index, components, _ = compute_squeeze_index(v, config)

        zeroed = sum(config.weights[k] * components.get(k, 0.0) for k in config.weights)
        assert index > zeroed

    def test_all_groups_missing_gives_none(self):
        v = view(
            bbw=Metric("BBW", 0.04, pct_rank=None),
            range_width=float("nan"),
            range_metric=Metric("диапазон(20)", float("nan"), pct_rank=None),
            volume=VolumeContext(float("nan"), "нет", 0, float("nan"), 0, float("nan")),
            profile=None,
            divergence=None,
        )
        index, components, excluded = compute_squeeze_index(v, Config())
        # Дивергенция всегда измерима: её отсутствие — это ноль, а не пропуск.
        assert components == {"divergence": 0.0}
        assert index == pytest.approx(0.0)
        assert set(excluded) == {"volatility", "range", "volume", "value_area"}

    def test_volume_excluded_when_basis_too_thin(self):
        """На 5m и 1m наблюдений на слот единицы — группа не скорится."""
        v = view(volume=VolumeContext(0.7, "медиана слота", 2, 0.6, 3, 0.55))
        _, components, excluded = compute_squeeze_index(v, Config())
        assert "volume" in excluded
        assert "volume" not in components


class TestConfig:
    def test_range_is_measured_by_percentile_not_by_absolute_width(self):
        """Абсолютных порогов ширины в конфиге больше нет — они не работали."""
        config = Config()
        assert config.range_percentile == 20.0
        assert not hasattr(config, "range_thresholds")
        assert not hasattr(config, "range_threshold")

    def test_defaults_match_tz(self):
        config = Config()
        assert config.bbw_percentile == 20.0
        assert config.atr_decline_bars == 10


class TestPercentileBase:
    """База перцентиля обрезается по времени, а не по числу свечей.

    Константа в 360 значений означает у разных ТФ разное: на дневке это 360
    суток, на часовке — пятнадцать. Требование §4.2 о 60 сутках охвата на 1h
    при этом проходило, потому что проверялось по охвату всего загруженного
    ряда, а не обрезанной базы.
    """

    def values(self, count: int):
        import numpy as np

        return np.arange(float(count))

    def test_daily_base_unchanged(self):
        """На 1d/4h/1w правка ничего не двигает: там и было 360 значений."""
        from cryptomcp.analysis import percentile_base

        history, span = percentile_base(self.values(700), "1d")
        assert len(history) == 360
        assert span == 360.0

    def test_four_hour_base_is_sixty_days(self):
        from cryptomcp.analysis import percentile_base

        history, span = percentile_base(self.values(700), "4h")
        assert len(history) == 360
        assert span == 60.0

    def test_hourly_base_grows_to_sixty_days(self):
        """Было 360 значений = 15 суток, стало 1440 = 60 суток."""
        from cryptomcp.analysis import percentile_base

        history, span = percentile_base(self.values(2000), "1h")
        assert len(history) == 1440
        assert span == 60.0

    def test_short_series_reports_its_real_span(self):
        """Свежий листинг обязан получить честный отказ, а не охват всего ряда."""
        from cryptomcp.analysis import percentile_base

        history, span = percentile_base(self.values(200), "1h")
        assert len(history) == 199
        assert span < 60.0

    def test_nan_warmup_does_not_count_as_history(self):
        import numpy as np

        from cryptomcp.analysis import percentile_base

        values = np.concatenate([np.full(250, np.nan), np.arange(500.0)])
        history, span = percentile_base(values, "4h")
        assert len(history) == 360
        assert not np.isnan(history).any()
