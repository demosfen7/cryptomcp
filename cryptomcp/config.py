"""Пороги и веса (ТЗ §8, PLAN §9).

Всё, что подлежит калибровке, живёт здесь и переопределяется YAML-файлом.
Веса взяты из ТЗ §4.3 и объявлены там же стартовыми: пересматривать их следует
по накопленной статистике из журнала (PLAN §4.11), а не по впечатлению.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

#: Лестница таймфреймов по умолчанию. 5m и 1m в неё не входят и запрашиваются
#: явно — см. PLAN §5.
DEFAULT_TIMEFRAMES = ("1w", "1d", "4h", "1h", "15m")

#: Веса групп признаков из ТЗ §4.3. Сумма — единица.
DEFAULT_WEIGHTS: dict[str, float] = {
    "volatility": 0.35,
    "range": 0.25,
    "volume": 0.20,
    "value_area": 0.10,
    "divergence": 0.10,
}

#: Порог ширины диапазона по таймфреймам, доля от цены.
#: ТЗ §4.2 даёт «менее 6–8%, настраивается по таймфреймам»; одно значение для
#: недели и для минутки заведомо не подходит.
DEFAULT_RANGE_THRESHOLDS: dict[str, float] = {
    "1w": 0.15,
    "1d": 0.10,
    "4h": 0.06,
    "1h": 0.04,
    "15m": 0.025,
    "5m": 0.015,
    "1m": 0.008,
}

#: Перцентиль BBW, ниже которого признак сжатия считается сработавшим (ТЗ §4.2).
DEFAULT_BBW_PERCENTILE = 20.0

#: Сколько свечей подряд ATR должен снижаться, чтобы признак засчитался.
DEFAULT_ATR_DECLINE_BARS = 10


@dataclass
class Config:
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    range_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RANGE_THRESHOLDS)
    )
    bbw_percentile: float = DEFAULT_BBW_PERCENTILE
    atr_decline_bars: int = DEFAULT_ATR_DECLINE_BARS
    #: Порог значимости изменения OI для классификации (PLAN §4.12).
    oi_change_threshold: float = 0.01
    #: Окно объёмного профиля в свечах. ТЗ §4.2 предписывает 100–200.
    volume_profile_window: int = 200
    #: Окно поиска дивергенций RSI. ТЗ §4.2: 30–50.
    divergence_window: int = 40
    #: Куда писать журнал расчётов (PLAN §4.11).
    journal_path: str = "journal/squeeze.jsonl"

    def range_threshold(self, interval: str) -> float:
        return self.range_thresholds.get(interval, DEFAULT_RANGE_THRESHOLDS["4h"])

    @classmethod
    def load(cls, path: str | None = None) -> Config:
        path = path or os.environ.get("CRYPTOMCP_CONFIG")
        config = cls()
        if not path or not os.path.exists(path):
            return config

        with open(path, encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}

        if "timeframes" in data:
            config.timeframes = tuple(data["timeframes"])
        # Веса и пороги сливаются, а не заменяются: частичное переопределение
        # не должно молча обнулять остальные ключи.
        config.weights.update(data.get("weights", {}))
        config.range_thresholds.update(data.get("range_thresholds", {}))
        for key in ("bbw_percentile", "atr_decline_bars", "oi_change_threshold",
                    "volume_profile_window", "divergence_window", "journal_path"):
            if key in data:
                setattr(config, key, data[key])
        return config
