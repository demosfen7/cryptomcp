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
        range_duration=20,
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

    def test_range_score_falls_as_width_grows(self):
        config = Config()
        narrow = score_range(view(range_width=0.01), config)
        wide = score_range(view(range_width=0.20), config)
        assert narrow > wide
        assert wide == pytest.approx(0.4, abs=0.01)  # остаётся вклад длительности

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
    def test_range_threshold_per_timeframe(self):
        config = Config()
        assert config.range_threshold("1w") > config.range_threshold("4h")
        assert config.range_threshold("4h") > config.range_threshold("15m")

    def test_unknown_timeframe_falls_back(self):
        assert Config().range_threshold("3h") == Config().range_threshold("4h")

    def test_defaults_match_tz(self):
        config = Config()
        assert config.range_threshold("4h") == pytest.approx(0.06)  # ТЗ §4.2
        assert config.bbw_percentile == 20.0
        assert config.atr_decline_bars == 10
