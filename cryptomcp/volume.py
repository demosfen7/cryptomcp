"""Объёмные метрики с поправкой на суточную сезонность (PLAN §4.10).

У крипты сильная суточная сезонность: свеча в 04:00 UTC систематически тише,
чем в 14:00. Сравнение с плоской MA20 на внутридневных таймфреймах показывало бы
«затухание объёма 0.6x» там, где на самом деле просто ночь — то есть врало бы
системно, а не изредка.

Это касается и 4h: в сутках шесть четырёхчасовых свечей, и ночная из них тише
дневных. Ограничиться «только 4h и 1d» проблему не решает.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .series import Series, interval_ms

DAY_MS = 86_400_000

#: Таймфреймы, на которых суточной сезонности нет по построению.
_NO_SEASONALITY = {"1d", "3d", "1w"}

#: Минимум наблюдений на слот, ниже которого baseline объявляется слабым.
MIN_SAMPLES_PER_SLOT = 10

#: Минимум, ниже которого объёмные метрики не участвуют в скоринге вовсе.
MIN_SAMPLES_TO_SCORE = 5


@dataclass(frozen=True)
class VolumeContext:
    """Текущий объём относительно сопоставимой базы."""

    #: Отношение объёма последней свечи к базе.
    ratio: float
    #: Как считалась база: "сезонный слот" или "скользящая средняя".
    basis: str
    #: Наблюдений в базе.
    samples: int
    #: Отношение MA20 к MA100 — затухание объёма из ТЗ §4.2, группа 2.
    ma_ratio: float
    #: Число аномальных баров за последние 30 свечей.
    anomalous_bars: int
    #: Средняя доля тейкер-покупок за последние 30 свечей.
    taker_buy_mean: float

    @property
    def weak_basis(self) -> bool:
        return self.samples < MIN_SAMPLES_PER_SLOT

    @property
    def scorable(self) -> bool:
        """Участвует ли объёмная группа в squeeze_index.

        На 5m и 1m наблюдений на слот суток набирается единицы, и база
        перестаёт что-либо значить. Лучше исключить группу из скоринга, чем
        подмешать в индекс шум (PLAN §4.10).
        """
        return self.samples >= MIN_SAMPLES_TO_SCORE

    def to_dict(self) -> dict[str, object]:
        return {
            "ratio": round(self.ratio, 3),
            "basis": self.basis,
            "samples": self.samples,
            "weak_basis": self.weak_basis,
            "ma20_over_ma100": round(self.ma_ratio, 3),
            "anomalous_bars_30": self.anomalous_bars,
            "taker_buy_mean_30": round(self.taker_buy_mean, 3),
        }


def slot_of_day(open_time_ms: np.ndarray, interval: str) -> np.ndarray:
    """Номер слота внутри суток для каждой свечи."""
    step = interval_ms(interval)
    return ((open_time_ms % DAY_MS) // step).astype("int64")


#: Окно для оценки текущего уровня объёма, свечей.
LEVEL_WINDOW = 20


def seasonal_baseline(series: Series) -> tuple[float, str, int]:
    """База для сравнения объёма: свежий уровень, поправленный на форму суток.

    Возвращает (база, описание, число наблюдений на слот).

    Наивная медиана того же слота за всё окно смешивает два разных эффекта:
    форму суток и общий дрейф активности. Измерено на живых данных: у SUIUSDT
    медиана слота в первой половине шестидесятидневного окна 1.97M, во второй
    3.16M. При такой базе обычная свеча показывала бы 4x, и метрика читалась бы
    как всплеск, хотя измеряла бы всего лишь «сегодня оживлённее, чем два
    месяца назад».

    Поэтому эффекты разделены. Форма суток — устойчивое отношение медианы слота
    к общей медиане, её имеет смысл брать с длинного окна. Уровень — медиана
    последних свечей. База есть произведение одного на другое.

    Дрейф уровня при этом не теряется: он измеряется отдельной метрикой
    MA20/MA100, где он и является предметом (ТЗ §4.2, группа 2).
    """
    volumes = series.quote_volume
    if len(volumes) < 2:
        return float("nan"), "недостаточно данных", 0

    recent = volumes[-(LEVEL_WINDOW + 1):-1]
    level = float(np.median(recent)) if len(recent) else float("nan")

    if series.interval in _NO_SEASONALITY:
        return level, f"медиана последних {len(recent)}", len(recent)

    slots = slot_of_day(series.df["open_time"].to_numpy(dtype="int64"), series.interval)
    current_slot = int(slots[-1])
    # Текущая свеча из базы исключается: сравнивать её с самой собой нельзя.
    history = volumes[:-1]
    same_slot = history[slots[:-1] == current_slot]

    overall = float(np.median(history)) if len(history) else float("nan")
    if len(same_slot) == 0 or not overall or np.isnan(overall):
        return level, f"медиана последних {len(recent)} (слот не набран)", len(recent)

    factor = float(np.median(same_slot)) / overall
    hours = current_slot * interval_ms(series.interval) / 3_600_000
    return (
        level * factor,
        f"уровень последних {len(recent)} × сезонность слота {hours:04.1f} UTC ({factor:.2f})",
        len(same_slot),
    )


def anomalous_bars(series: Series, atr_values: np.ndarray, *, window: int = 30,
                   volume_multiple: float = 3.0, move_atr_fraction: float = 0.5) -> int:
    """Бары с большим объёмом и малым движением цены (ТЗ §4.2, группа 2).

    Интерпретация из ТЗ: набор позиции без движения цены. Объём сравнивается
    со скользящей средней, движение — с ATR, чтобы порог был сопоставим с
    волатильностью инструмента, а не задан в процентах на все случаи.
    """
    volumes = series.quote_volume
    if len(volumes) < window + 20:
        return 0

    rolling_mean = pd.Series(volumes).rolling(20).mean().to_numpy()
    opens = series.col("open")
    closes = series.close

    count = 0
    for i in range(len(volumes) - window, len(volumes)):
        mean = rolling_mean[i]
        atr_value = atr_values[i] if i < len(atr_values) else np.nan
        if np.isnan(mean) or np.isnan(atr_value) or mean <= 0 or atr_value <= 0:
            continue
        big_volume = volumes[i] >= volume_multiple * mean
        small_move = abs(closes[i] - opens[i]) < move_atr_fraction * atr_value
        if big_volume and small_move:
            count += 1
    return count


def volume_context(series: Series, atr_values: np.ndarray) -> VolumeContext:
    """Полная объёмная картина по ряду."""
    volumes = series.quote_volume
    baseline, basis, samples = seasonal_baseline(series)

    ratio = float(volumes[-1] / baseline) if baseline and not np.isnan(baseline) else float("nan")

    ma20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float("nan")
    ma100 = float(np.mean(volumes[-100:])) if len(volumes) >= 100 else float("nan")
    ma_ratio = ma20 / ma100 if ma100 else float("nan")

    taker = series.taker_buy_ratio[-30:]
    taker_mean = float(np.nanmean(taker)) if len(taker) else float("nan")

    return VolumeContext(
        ratio=ratio,
        basis=basis,
        samples=samples,
        ma_ratio=ma_ratio,
        anomalous_bars=anomalous_bars(series, atr_values),
        taker_buy_mean=taker_mean,
    )
