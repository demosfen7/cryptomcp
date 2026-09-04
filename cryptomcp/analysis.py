"""Сборка пяти групп признаков и squeeze_index (модуль A2/A3 из ТЗ).

Ключевое свойство: индекс считается только по тем группам, база которых
достаточна. Недостающие группы исключаются, а веса оставшихся перенормируются,
и список исключённого едет вместе с результатом. Альтернатива — подставлять
ноль за отсутствующую группу — тихо занижала бы индекс и делала бы значения
несопоставимыми между таймфреймами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .indicators import (
    Metric,
    atr,
    bollinger_width,
    completed_streaks,
    consecutive_below,
    consecutive_declining,
    donchian_width,
    ema,
    rsi,
    with_percentile,
)
from .levels import (
    Level,
    Pivots,
    VolumeProfile,
    cluster_levels,
    market_structure,
    pivots,
    swing_points,
    volume_profile,
)
from .series import Series
from .volume import VolumeContext, volume_context

#: Сколько свечей нужно, чтобы вообще что-то считать.
MIN_CANDLES = 60

#: Допуск кластеризации уровней в долях ATR (PLAN §4.3).
LEVEL_TOLERANCE_ATR = 0.5

#: Окно перцентилей: до 360 предыдущих значений (ТЗ §4.2).
PERCENTILE_WINDOW = 360


def required_candles(interval: str) -> int:
    """Сколько свечей нужно метрикам — и ровно столько им и даётся.

    Окно каноническое: один и тот же запрос обязан давать один и тот же ряд,
    откуда бы он ни пришёл — из архива или из биржи. Иначе повторяется история
    с объёмной базой, где одна свеча получала разные числа в разных
    инструментах: сезонный множитель и перцентили считаются по всему ряду, и
    лишняя сотня свечей в одном из путей тихо меняет результат.

    Величина определяется двумя требованиями §4.2: 360 предыдущих значений для
    перцентиля и 60 суток календарного охвата. Что больше, то и берётся.
    """
    from .indicators import MIN_PERCENTILE_SPAN_DAYS
    from .series import interval_ms

    by_span = int(MIN_PERCENTILE_SPAN_DAYS * 86_400_000 / interval_ms(interval))
    return max(PERCENTILE_WINDOW + 1, by_span + 1)


#: Крупнейший таймфрейм, на котором ещё видно поглощение внутри дня.
ACCUMULATION_CEILING = "1h"

#: Лестница вниз для выбора ряда под накопление.
_LADDER_DOWN = ("1w", "1d", "4h", "1h", "15m", "5m", "1m")


def accumulation_interval(interval: str) -> str:
    """На каком ряду считать накопление для сжатия, найденного на ``interval``.

    Правило: сжатие меряется на своём таймфрейме, накопление — всегда на
    часовом или мельче. Причина в §4.19: дневное разрешение стирает поглощение
    целиком, потому что оно происходит внутри одной свечи.

    Для запросов от 1h и ниже берётся ступень вниз: на своём же ряду
    «поглощение» было бы просто повтором объёмной группы.
    """
    from .series import interval_ms

    if interval_ms(interval) > interval_ms(ACCUMULATION_CEILING):
        return ACCUMULATION_CEILING
    if interval not in _LADDER_DOWN:
        return interval
    index = _LADDER_DOWN.index(interval)
    return _LADDER_DOWN[min(index + 1, len(_LADDER_DOWN) - 1)]


def percentile_base(values: np.ndarray, interval: str) -> tuple[np.ndarray, float]:
    """База для перцентиля и её КАЛЕНДАРНЫЙ охват.

    Обрезка по времени, а не по числу свечей. Константа в 360 значений
    означает у разных таймфреймов разное: на дневке это 360 суток, на
    четырёхчасовке 60, а на часовке — пятнадцать. Требование §4.2 «не менее
    60 суток охвата» на часовке при этом проходило, потому что проверялось по
    охвату всего загруженного ряда (1691 свеча, 70 суток), а не обрезанной
    базы. Правило, которое должно ловить короткие базы, на 1h не срабатывало,
    и в выдаче стоял охват 70 суток вместо честных 15.

    Длина берётся из required_candles — того же места, откуда берётся длина
    самого ряда. Иначе ряд грузится под одно окно, а перцентиль считается по
    другому: данные уже лежат в памяти и просто выбрасываются.

    Охват считается по числу удержанных наблюдений: ряд регулярный, свеча
    закрывается строго по расписанию, и произведение на шаг таймфрейма — это
    и есть календарный охват базы.
    """
    from .series import interval_ms

    clean = values[~np.isnan(values)]
    keep = required_candles(interval) - 1
    history = clean[-(keep + 1):-1]
    span_days = len(history) * interval_ms(interval) / 86_400_000
    return history, span_days


#: Ниже этой длительности вопрос «чем сжатие началось» не стоит: диапазон ещё
#: не сложился, и любой размашистый бар внутри — это просто соседний бар.
SHOCK_MIN_NARROW_BARS = 5

#: Событием бар считается по размаху И по объёму сразу. Замер, из которого
#: взяты пороги (226 монет, 4h, 03.09.2026): среди одиннадцати монет со
#: сжатием от пяти свечей доля ширины диапазона не отбирает ничего — «бар
#: шире 40% диапазона» верно для десяти из одиннадцати, потому что в узком
#: диапазоне широчайший бар и обязан занимать заметную его часть. Размах от
#: 3 ATR вместе с объёмом от 3x оставляет три монеты из одиннадцати, и HOME —
#: кейс, ради которого признак заводился, — среди них.
#:
#: Меряется РАЗМАХ, а не тело, хотя в присланной формуле стояло тело: у HOME
#: событийный бар имел размах 9.18% при теле +0.34%, то есть по телу не
#: сработало бы вовсе. Событие в узком диапазоне обычно выглядит как выброс с
#: возвратом, и тело у него мало по построению.
SHOCK_RANGE_ATR = 3.0
SHOCK_VOLUME_MULTIPLE = 3.0


@dataclass(frozen=True)
class Shock:
    """Самый размашистый бар ВНУТРИ окна сжатия.

    Проверять надо именно внутри, а не перед началом. Кейс HOME 03.09.2026:
    заказчик прочитал сжатие как «истощение после обвала», но обвал 02.09
    случился на 33-й свече из 33, то есть В СЕРЕДИНЕ сжатия, а двенадцать
    свечей ПЕРЕД его началом были тихими (объём 0.37–1.11x, тела до 0.8 ATR).
    Правило «шок перед началом» по универсуму не сработало ни разу из девяти
    монет со сжатием; правило «шок внутри» на HOME срабатывает.

    Смысл признака: «узк 33» означает 33 свечи узкого диапазона, но если одна
    из них размахом почти во весь этот диапазон, то длительность меряет не то,
    что кажется. Вывода код не делает — печатает и пишет в журнал (ТЗ §1.1).
    """

    #: Сколько свечей назад стоит этот бар.
    bars_ago: int
    #: Размах бара в ATR, посчитанном ДО него: иначе сам бар раздувает базу.
    range_atr: float
    #: Доля текущей ширины диапазона(20), которую занимает размах бара.
    range_share: float
    volume_ratio: float

    @property
    def loud(self) -> bool:
        """Событие, а не просто самый широкий бар тихого диапазона."""
        return (
            self.range_atr >= SHOCK_RANGE_ATR
            and self.volume_ratio >= SHOCK_VOLUME_MULTIPLE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bars_ago": self.bars_ago,
            "range_atr": round(self.range_atr, 2),
            "range_share": round(self.range_share, 3),
            "volume_ratio": round(self.volume_ratio, 2),
            "loud": self.loud,
        }


def shock_inside(
    series: Series,
    atr_values: np.ndarray,
    *,
    narrow_bars: int,
    range_width: float,
) -> Shock | None:
    """Самый размашистый бар внутри окна сжатия, или None.

    None означает «вопрос не стоит»: сжатия нет или оно короче
    SHOCK_MIN_NARROW_BARS. Тихое сжатие возвращает бар с малым размахом —
    отсутствие события видно по его же числам, отдельного состояния для этого
    не нужно.
    """
    if narrow_bars < SHOCK_MIN_NARROW_BARS:
        return None
    high, low = series.high, series.low
    volumes = series.quote_volume
    rolling = pd.Series(volumes).rolling(20).mean().to_numpy()
    start = max(1, len(high) - narrow_bars)
    price = float(series.close[-1])
    width_price = range_width * price if price else float("nan")

    best: Shock | None = None
    for i in range(start, len(high)):
        # ATR берётся ДО бара: ATR(14) на самом баре уже включает его размах и
        # занижает отношение — тем сильнее, чем крупнее событие.
        base = atr_values[i - 1]
        if np.isnan(base) or base <= 0:
            continue
        span = high[i] - low[i]
        mean = rolling[i]
        candidate = Shock(
            bars_ago=len(high) - 1 - i,
            range_atr=span / base,
            range_share=span / width_price if width_price else float("nan"),
            volume_ratio=(
                volumes[i] / mean if not np.isnan(mean) and mean > 0 else float("nan")
            ),
        )
        if best is None or candidate.range_atr > best.range_atr:
            best = candidate
    return best


@dataclass
class TimeframeView:
    """Полная картина по одному таймфрейму."""

    interval: str
    price: float
    atr_value: float
    atr_pct: float
    rsi_value: float
    ema_state: str
    structure: str
    position_in_range: float

    bbw: Metric
    atr_metric: Metric
    atr_declining_bars: int

    range_low: float
    range_high: float
    range_width: float
    range_width_atr: float
    #: Ширина, соответствующая заданному перцентилю истории ЭТОЙ монеты.
    range_threshold: float
    #: Ширина диапазона с перцентилем по собственной истории.
    range_metric: Metric
    #: Сколько последних свечей ПОДРЯД ширина диапазона(20) держалась ниже
    #: порога. Это длительность СЖАТИЯ, а не возраст текущего диапазона:
    #: у широкого диапазона здесь ноль, и это верное значение, а не сбой.
    narrow_bars: int
    #: Та же длительность в сутках. Свечи разных ТФ несопоставимы: «узк 19» на
    #: 4h выглядит внушительнее, чем «узк 7» на 1d, хотя это 3.2 суток против
    #: семи. В общем списке, где строки обоих ТФ стоят рядом, счётчик без
    #: пересчёта в календарь читается неверно.
    narrow_days: float
    #: Длительность текущей серии с перцентилем среди ЗАВЕРШЁННЫХ серий этой
    #: пары. Отдельная метрика, а не число: у монеты, которая никогда не
    #: стояла дольше двух суток, полтора дня — это её максимум, а у BTC — шум.
    narrow_metric: Metric

    volume: VolumeContext
    profile: VolumeProfile | None
    divergence: str | None
    #: Самый размашистый бар внутри окна сжатия; None — сжатия нет.
    shock: Shock | None = None

    levels: list[Level] = field(default_factory=list)
    pivots_weekly: Pivots | None = None

    squeeze_index: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    excluded: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "price": self.price,
            "ema": self.ema_state,
            "structure": self.structure,
            "position_in_range": round(self.position_in_range, 3),
            "rsi": round(self.rsi_value, 1),
            "atr_pct": round(self.atr_pct, 3),
            "bbw": self.bbw.to_dict(),
            "atr_percentile": self.atr_metric.to_dict(),
            "atr_declining_bars": self.atr_declining_bars,
            "range": {
                "low": self.range_low,
                "high": self.range_high,
                "width_pct": round(self.range_width * 100, 2),
                "width_atr": round(self.range_width_atr, 2),
                "threshold_pct": (
                    round(self.range_threshold * 100, 2)
                    if self.range_threshold == self.range_threshold else None
                ),
                "width_percentile": self.range_metric.to_dict(),
                "bars_below_threshold": self.narrow_bars,
                "days_below_threshold": round(self.narrow_days, 2),
                "duration_percentile": self.narrow_metric.to_dict(),
                "shock_inside": self.shock.to_dict() if self.shock else None,
            },
            "volume": self.volume.to_dict(),
            "divergence": self.divergence,
            "squeeze_index": (
                round(self.squeeze_index, 3) if self.squeeze_index is not None else None
            ),
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "excluded_from_index": self.excluded,
            "series": self.meta,
        }


def ema_state(close: np.ndarray) -> str:
    """Положение цены относительно EMA50 и EMA200 (PLAN §4.4).

    По положению, а не по наклону: наклон дёргается на флэте, давая
    противоположные вердикты на соседних свечах, а флэт — основной объект.
    """
    if len(close) < 200:
        return "n/a"
    e50 = ema(close, 50)[-1]
    e200 = ema(close, 200)[-1]
    if np.isnan(e50) or np.isnan(e200):
        return "n/a"
    price = close[-1]
    if price > e50 and e50 > e200:
        return "above"
    if price < e50 and e50 < e200:
        return "below"
    return "mixed"


def position_in_range(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                      window: int = 20) -> float:
    """Где цена внутри диапазона окна: 0 — у низа, 1 — у верха."""
    if len(close) < window:
        return float("nan")
    top = float(high[-window:].max())
    bottom = float(low[-window:].min())
    if top <= bottom:
        return 0.5
    return float((close[-1] - bottom) / (top - bottom))


def rsi_divergence(series: Series, rsi_values: np.ndarray, window: int = 40) -> str | None:
    """Расхождение экстремумов цены и RSI (ТЗ §4.2, группа 5).

    Признак объявлен в ТЗ информативным, но слабым, поэтому его вес в индексе
    самый низкий наравне с объёмным профилем.
    """
    if len(series) < window + 5:
        return None

    tail = series.tail(window)
    offset = len(series) - len(tail)
    swings = swing_points(tail.high, tail.low, 2, 2)

    lows = [s for s in swings if s.kind == "low"]
    if len(lows) >= 2:
        first, second = lows[-2], lows[-1]
        r1, r2 = rsi_values[offset + first.index], rsi_values[offset + second.index]
        if not (np.isnan(r1) or np.isnan(r2)):
            if second.price < first.price and r2 > r1:
                return "бычья"

    highs = [s for s in swings if s.kind == "high"]
    if len(highs) >= 2:
        first, second = highs[-2], highs[-1]
        r1, r2 = rsi_values[offset + first.index], rsi_values[offset + second.index]
        if not (np.isnan(r1) or np.isnan(r2)):
            if second.price > first.price and r2 < r1:
                return "медвежья"

    return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_volatility(view: TimeframeView, config: Config) -> float | None:
    """Группа 1. Чем ниже перцентиль BBW и ATR, тем сильнее сжатие."""
    if not view.bbw.has_context:
        return None
    score = 0.6 * (1.0 - view.bbw.pct_rank / 100.0)
    if view.atr_metric.has_context:
        score += 0.2 * (1.0 - view.atr_metric.pct_rank / 100.0)
    else:
        score += 0.2 * 0.5  # нейтрально, если перцентиля ATR нет
    score += 0.2 * _clamp(view.atr_declining_bars / config.atr_decline_bars)
    return _clamp(score)


def score_range(view: TimeframeView, config: Config) -> float | None:
    """Группа 3. Узкий диапазон и его длительность.

    Узость меряется перцентилем по собственной истории монеты, а не абсолютным
    порогом: у BULLA дневной диапазон 40% — это её обычное состояние, у PAXG
    8.6% — необычно широко. Один и тот же процент означает у них
    противоположное, и группа с абсолютным порогом давала ноль всем, кроме
    золота.

    Группа исключается из индекса, если базы для перцентиля не хватило, — как
    и все остальные (см. compute_squeeze_index).
    """
    if not view.range_metric.has_context:
        return None
    tightness = 1.0 - view.range_metric.pct_rank / 100.0
    duration = _clamp(view.narrow_bars / 20.0)
    return _clamp(0.6 * tightness + 0.4 * duration)


def score_volume(view: TimeframeView) -> float | None:
    """Группа 2. Затухание объёма и бары набора позиции."""
    if not view.volume.scorable or np.isnan(view.volume.ma_ratio):
        return None
    decline = _clamp((1.0 - view.volume.ma_ratio) / 0.5)
    bars = _clamp(view.volume.anomalous_bars / 3.0)
    return _clamp(0.7 * decline + 0.3 * bars)


def score_duration(view: TimeframeView) -> float | None:
    """Группа 4. Насколько длинна текущая серия сжатия ДЛЯ ЭТОЙ пары.

    Перцентилем, а не отношением к константе: длина типичного сжатия у BTC и
    у свежего листинга различается кратно — ровно как ширина диапазона, где
    абсолютный порог уже один раз не сработал ни на одной монете.

    Группа исключается, если завершённых серий или охвата не хватило. Ноль
    означал бы «сжатие короткое», хотя его просто не с чем сравнить, — а
    свежие листинги это как раз целевая категория сканера.
    """
    if not view.narrow_metric.has_context:
        return None
    return _clamp(view.narrow_metric.pct_rank / 100.0)


def compute_squeeze_index(
    view: TimeframeView, config: Config
) -> tuple[float | None, dict[str, float], list[str]]:
    """Взвешенная сумма групп с перенормировкой по доступным.

    Группа без достаточной базы исключается, а не обнуляется: ноль означал бы
    «признака нет», хотя на деле его не удалось измерить.

    Объёмный профиль и дивергенция RSI в свёртку не входят (§4.33). Обе
    остались в выдаче справкой: профиль — разделом 4, дивергенция — разделом
    5. Причина у них разная, а следствие было общее — «бесплатный пол»
    индекса, с которого до топ-15 оставалось добрать 0.22 из 0.60.
    """
    raw: dict[str, float | None] = {
        "volatility": score_volatility(view, config),
        "range": score_range(view, config),
        "volume": score_volume(view),
        "duration": score_duration(view),
    }

    available = {k: v for k, v in raw.items() if v is not None}
    excluded = [k for k, v in raw.items() if v is None]
    if not available:
        return None, {}, excluded

    total_weight = sum(config.weights.get(k, 0.0) for k in available)
    if total_weight <= 0:
        return None, {}, excluded

    index = sum(config.weights.get(k, 0.0) * v for k, v in available.items()) / total_weight
    return round(index, 4), available, excluded


def analyse_timeframe(
    series: Series,
    config: Config,
    *,
    weekly_pivots: Pivots | None = None,
) -> TimeframeView:
    """Собрать полную картину по одному таймфрейму."""
    from .series import interval_ms

    close, high, low = series.close, series.high, series.low

    atr_values = atr(high, low, close, 14)
    atr_value = float(atr_values[-1])
    price = float(close[-1])
    atr_pct = atr_value / price * 100.0 if price else float("nan")

    bbw_values = bollinger_width(close, 20, 2.0)
    bbw_history, bbw_span = percentile_base(bbw_values, series.interval)
    bbw_metric = with_percentile(
        "BBW(20,2)", float(bbw_values[-1]), bbw_history, bbw_span,
        threshold=config.bbw_percentile, threshold_side="below",
    )

    atr_pct_series = np.divide(
        atr_values, close, out=np.full_like(atr_values, np.nan), where=close > 0
    ) * 100.0
    atr_history, atr_span = percentile_base(atr_pct_series, series.interval)
    atr_metric = with_percentile(
        "ATR%/price", atr_pct, atr_history, atr_span,
        unit="%", threshold=config.bbw_percentile, threshold_side="below",
    )

    width_series = donchian_width(high, low, 20, close)
    width_history, width_span = percentile_base(width_series, series.interval)
    range_metric = with_percentile(
        "диапазон(20)", float(width_series[-1]) * 100, width_history * 100,
        width_span, unit="%",
        threshold=config.range_percentile, threshold_side="below",
    )
    # Порог — не число из конфига, а та ширина, которая у ЭТОЙ монеты
    # соответствует заданному перцентилю её собственной истории.
    threshold = (
        float(np.percentile(width_history, config.range_percentile))
        if len(width_history) else float("nan")
    )
    window_high = float(high[-20:].max()) if len(high) >= 20 else float("nan")
    window_low = float(low[-20:].min()) if len(low) >= 20 else float("nan")
    range_width = float(width_series[-1])

    profile_window = series.tail(config.volume_profile_window)
    profile = volume_profile(
        profile_window.high, profile_window.low, profile_window.quote_volume
    )

    rsi_values = rsi(close, 14)
    swings = swing_points(high, low, 2, 2)

    narrow_bars = (
        consecutive_below(width_series, threshold) if threshold == threshold else 0
    )
    step_days = interval_ms(series.interval) / 86_400_000
    narrow_days = narrow_bars * step_days
    # База длительности — завершённые серии по всему ряду, который дан
    # анализу. Ряд один и тот же для всех метрик (§4.15): взять для этой
    # группы более глубокий значило бы завести второй вход и второе
    # расхождение между сканером и сервером — то самое, что чинилось в §4.31.
    streaks = (
        completed_streaks(width_series, threshold) if threshold == threshold else []
    )
    narrow_metric = with_percentile(
        "длительность сжатия", float(narrow_bars),
        np.asarray(streaks, dtype="float64"), len(width_series) * step_days,
        unit=" св.", min_obs=config.duration_min_streaks,
        min_span_days=config.duration_min_span_days,
    )

    view = TimeframeView(
        interval=series.interval,
        price=price,
        atr_value=atr_value,
        atr_pct=atr_pct,
        rsi_value=float(rsi_values[-1]),
        ema_state=ema_state(close),
        structure=market_structure(swings),
        position_in_range=position_in_range(close, high, low),
        bbw=bbw_metric,
        atr_metric=atr_metric,
        atr_declining_bars=consecutive_declining(atr_values),
        range_low=window_low,
        range_high=window_high,
        range_width=range_width,
        range_width_atr=(range_width * price / atr_value) if atr_value else float("nan"),
        range_threshold=threshold,
        range_metric=range_metric,
        narrow_bars=narrow_bars,
        narrow_days=narrow_days,
        narrow_metric=narrow_metric,
        volume=volume_context(series, atr_values),
        profile=profile,
        divergence=rsi_divergence(series, rsi_values, config.divergence_window),
        shock=shock_inside(
            series, atr_values, narrow_bars=narrow_bars, range_width=range_width
        ),
        levels=cluster_levels(swings, tolerance=LEVEL_TOLERANCE_ATR * atr_value),
        pivots_weekly=weekly_pivots,
        meta=series.meta(),
    )

    view.squeeze_index, view.components, view.excluded = compute_squeeze_index(view, config)
    return view


def weekly_pivots_from(series: Series) -> Pivots | None:
    """Пивоты от предыдущей закрытой недели (PLAN §4.7).

    Берётся именно предпоследняя свеча: последняя закрытая неделя — это текущий
    расчётный период, а пивоты строятся от того, что было до него.
    """
    if len(series) < 2:
        return None
    return pivots(
        float(series.high[-2]), float(series.low[-2]), float(series.close[-2])
    )
