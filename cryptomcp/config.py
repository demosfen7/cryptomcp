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

#: Веса групп признаков. Сумма — единица.
#:
#: Против ТЗ §4.3 изменены дважды. `value_area` (0.10) и `divergence` (0.10)
#: убраны из свёртки в §4.33: первая обнулялась ровно тогда, когда цена
#: подходила к границе диапазона, и давала максимум за широкий профиль;
#: вторая была плоской единицей у пяти монет подряд, потому что RSI выходит
#: из перепроданности механически. Обе остались в выдаче справкой.
#:
#: Освободившиеся 0.20 целиком ушли в `duration`: длительность сжатия была
#: единственной метрикой, меряющей время, и на индекс не влияла вовсе — HOME
#: с 33 свечами подряд и ZAMA с нулём стояли в списке рядом.
#:
#: Веса стартовые и не откалиброваны, как и прежние. Менять их следует по
#: `outcomes`, а не по впечатлению.
DEFAULT_WEIGHTS: dict[str, float] = {
    "volatility": 0.35,
    "range": 0.25,
    "volume": 0.20,
    "duration": 0.20,
}

#: Сколько ЗАВЕРШЁННЫХ серий сжатия нужно, чтобы сравнивать с ними текущую.
#: Правило «60 наблюдений» здесь не годится: единица наблюдения не свеча, а
#: серия, их на порядок меньше. Замер 04.09.2026 на каноническом окне: на 4h
#: у пяти проверенных монет 4–12 завершённых серий, на 1d — 5–14.
DEFAULT_DURATION_MIN_STREAKS = 10

#: Календарный охват ряда, ниже которого длительность не с чем сравнивать.
DEFAULT_DURATION_MIN_SPAN_DAYS = 180.0

#: Перцентиль ширины диапазона, ниже которого он считается узким.
#:
#: Раньше здесь стояли абсолютные пороги из ТЗ §4.2 («менее 6–8%») по
#: таймфреймам. Измерение 02.09.2026 показало, что они не работают: медианный
#: дневной диапазон рынка 65% при пороге 10%, и группа «диапазон» с весом 0.25
#: давала ровно ноль всем ста двадцати монетам, кроме золотых токенов. Четверть
#: формулы работала как премия за то, что инструмент не криптовалюта.
#:
#: Абсолютный порог и не мог сработать: единый для BTC и для свежего листинга
#: он бессмыслен так же, как единый множитель объёма. Теперь узость меряется
#: перцентилем по собственной истории монеты — тем же способом, что BBW.
DEFAULT_RANGE_PERCENTILE = 20.0

#: Перцентиль BBW, ниже которого признак сжатия считается сработавшим (ТЗ §4.2).
DEFAULT_BBW_PERCENTILE = 20.0

#: Сколько свечей подряд ATR должен снижаться, чтобы признак засчитался.
DEFAULT_ATR_DECLINE_BARS = 10


@dataclass
class Config:
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    range_percentile: float = DEFAULT_RANGE_PERCENTILE
    bbw_percentile: float = DEFAULT_BBW_PERCENTILE
    atr_decline_bars: int = DEFAULT_ATR_DECLINE_BARS
    #: База группы «длительность»: сколько завершённых серий и какой охват
    #: нужен, чтобы текущую серию было с чем сравнивать (§4.33).
    duration_min_streaks: int = DEFAULT_DURATION_MIN_STREAKS
    duration_min_span_days: float = DEFAULT_DURATION_MIN_SPAN_DAYS
    #: Порог значимости изменения OI для классификации (PLAN §4.12).
    oi_change_threshold: float = 0.01
    #: Окно объёмного профиля в свечах. ТЗ §4.2 предписывает 100–200.
    volume_profile_window: int = 200
    #: Окно поиска дивергенций RSI. ТЗ §4.2: 30–50.
    divergence_window: int = 40
    #: Куда писать журнал расчётов (PLAN §4.11).
    journal_path: str = "journal/squeeze.jsonl"

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
        for key in ("bbw_percentile", "range_percentile", "atr_decline_bars",
                    "oi_change_threshold",
                    "duration_min_streaks", "duration_min_span_days",
                    "volume_profile_window", "divergence_window", "journal_path"):
            if key in data:
                setattr(config, key, data[key])
        return config
