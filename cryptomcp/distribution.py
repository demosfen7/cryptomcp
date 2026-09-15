"""Детектор распределения и его зеркало — накопление (SPEC-distribution-detector).

Протокол умеет отвечать «накопления нет». Это пассивный ответ: монета
опускается в ранге и остаётся под наблюдением. У противоположного состояния
есть собственная сигнатура, такая же чёткая, как накопление у ASTER, и она
читается за три события. Детектор нужен не для сортировки, а для **снятия с
наблюдения**: OAX за шесть месяцев двадцать раз подряд дал один и тот же
ответ, а сканер держал бы монету в списке.

Границы. Детектор не предсказывает направление и не даёт рекомендаций. Он
отвечает на один вопрос: «продолжает ли эта монета систематически отдавать
предложение в попытки роста».

**Одна процедура с параметром стороны, а не две.** Требование не
косметическое: четыре задокументированных случая расхождения баз объёма
возникли ровно потому, что одна и та же величина считалась в двух местах. Две
независимые реализации зеркальных метрик разойдутся так же — это вопрос
времени.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .series import Series
from .volume import absorption_window, baseline_series

#: Окно расчёта. ТЗ §2 задаёт «30 закрытых свечей», и на дневке это ровно то
#: же, что даёт `absorption_window` (трое суток, но не меньше тридцати свечей).
#: На часовом ряду числа расходятся, и правило проекта оказывается верным:
#:
#: приёмочный тест 6 ждёт по зеркалу кластеры накопления ASTER 16.08 21:00 —
#: это 51 свеча назад от 19.08 00:00, и окно в 30 свечей до них физически не
#: дотягивается. Замер: окно 30 → событий 0, окно 72 → событий 4, среди них
#: 16.08 21:00 (6.92x, три бара) и 17.08 03:00. Ровно эта ошибка уже
#: чинилась в `absorption_window`, и заводить рядом второе окно значило бы
#: завести её заново.
#:
#: Поэтому окно берётся оттуда же, а не константой. На 1d обе величины равны
#: 30, поэтому приёмочные тесты 1–5 от этого не меняются.
WINDOW = 30

#: Множитель объёма к базе, с которого свеча может стать баром распределения.
VOLUME_MULTIPLE = 2.0

#: Доля фитиля в диапазоне СВОЕЙ свечи.
#:
#: Порог не привязан ни к ATR, ни к абсолютному проценту цены — это прямое
#: следствие двух задокументированных багов: у ASTER порог самоуничтожился,
#: потому что ATR был слишком мал, у ORN — потому что слишком велик. Доля
#: диапазона свечи не вырождается ни на одном конце распределения
#: волатильности.
WICK_FRACTION = 0.5

#: Сколько свечей между барами ещё схлопываются в одно событие.
#:
#: Без этого правила один новостной спайк и его отдача засчитываются как два
#: независимых наблюдения: у OAX так стоят пары 25–26.07 и 09–10.09.
EVENT_GAP = 1

#: Событий, ниже которого вердикт всегда `none`. Три — вместе с требованием
#: отрицательного наклона это то, что отсекает откат после пробоя.
MIN_EVENTS = 3

#: Смещение в ATR(14), отделяющее подтверждённое распределение от
#: формирующегося.
SHIFT_ATR = 1.0

#: Оборот, ниже которого фитиль может быть артефактом одной сделки. Ниже этого
#: порога детектор отвечает `n/a` с причиной, а не `none`: `none` означает
#: «проверено, признака нет», `n/a` — «не проверялось».
MIN_TURNOVER = 2_000_000.0

DISTRIBUTION = "distribution"
ACCUMULATION = "accumulation"

#: Направление стороны: у распределения максимумы должны снижаться, у
#: накопления — расти. Вся зеркальность сведена в этот знак.
_DIRECTION = {DISTRIBUTION: -1.0, ACCUMULATION: 1.0}

VERDICT_NONE = "none"
VERDICT_FORMING = "forming"
VERDICT_CONFIRMED = "confirmed"
VERDICT_NA = "n/a"


@dataclass(frozen=True)
class Event:
    """Событие: один бар или схлопнутая группа баров.

    Максимум события — наибольший экстремум входящих баров, дата — дата
    ПЕРВОГО бара.
    """

    ts_ms: int
    extreme: float
    volume_ratio: float
    wick: float
    bars: int


@dataclass(frozen=True)
class Sided:
    """Одна сторона детектора."""

    side: str
    window: int
    events: tuple[Event, ...]
    #: Наклон МНК по экстремумам событий; None, когда событий меньше двух.
    slope: float | None
    #: Смещение от первого события к последнему в ATR(14). Знак — по стороне:
    #: у распределения положительное значит падение, у накопления — рост.
    shift_atr: float | None
    verdict: str
    #: Почему `n/a`. Пусто у остальных вердиктов.
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.events)


def _wick(
    side: str, high: float, low: float, open_: float, close: float
) -> float:
    """Доля фитиля нужной стороны в диапазоне свечи.

    Свеча с нулевым диапазоном баром не является: деление не выполняется.
    """
    span = high - low
    if span <= 0:
        return float("nan")
    if side == DISTRIBUTION:
        return (high - max(open_, close)) / span
    return (min(open_, close) - low) / span


def _turnover_per_day(series: Series, window: int) -> float:
    """Оборот за сутки по окну — медианой, а не последней свечой.

    Считается из самого ряда, а не из тикера биржи, по одной причине: все
    приёмочные тесты исторические и проходят через `as_of_ms`, а живого
    оборота на дату в прошлом не существует. Медиана, а не среднее и не
    последняя свеча: один тихий день не должен объявлять пару тонкой, а один
    новостной — ликвидной.
    """
    from .series import interval_ms

    tail = series.quote_volume[-window:]
    if len(tail) == 0:
        return float("nan")
    per_candle = float(np.median(tail))
    candles_per_day = 86_400_000 / interval_ms(series.interval)
    return per_candle * candles_per_day


def _bars(
    series: Series, side: str, window: int
) -> list[tuple[int, float, float, float]]:
    """Бары нужной стороны в окне: (индекс, экстремум, множитель, фитиль).

    База объёма — та же, что во всех остальных метриках: уровень 20
    предыдущих свечей × сезонность слота. Своей базы детектор не заводит,
    иначе множитель бара разошёлся бы с колонкой «объём» в `get_klines` на той
    же свече (приёмочный тест 8).
    """
    baseline = baseline_series(series, window)
    ratios = baseline.ratio(series.quote_volume)
    opens, closes = series.col("open"), series.close
    high, low = series.high, series.low

    found = []
    for i in range(max(0, len(series) - window), len(series)):
        ratio = ratios[i]
        if np.isnan(ratio) or ratio < VOLUME_MULTIPLE:
            continue
        wick = _wick(side, high[i], low[i], opens[i], closes[i])
        if np.isnan(wick) or wick < WICK_FRACTION:
            continue
        extreme = high[i] if side == DISTRIBUTION else low[i]
        found.append((i, float(extreme), float(ratio), float(wick)))
    return found


def _collapse(
    series: Series, side: str, bars: list[tuple[int, float, float, float]]
) -> tuple[Event, ...]:
    """Схлопнуть бары, стоящие подряд или через одну свечу, в события."""
    times = series.df["open_time"].to_numpy(dtype="int64")
    events: list[Event] = []
    group: list[tuple[int, float, float, float]] = []

    def flush() -> None:
        if not group:
            return
        extremes = [b[1] for b in group]
        extreme = max(extremes) if side == DISTRIBUTION else min(extremes)
        events.append(Event(
            ts_ms=int(times[group[0][0]]),
            extreme=extreme,
            volume_ratio=max(b[2] for b in group),
            wick=max(b[3] for b in group),
            bars=len(group),
        ))
        group.clear()

    for bar in bars:
        if group and bar[0] - group[-1][0] > EVENT_GAP + 1:
            flush()
        group.append(bar)
    flush()
    return tuple(events)


def detect(
    series: Series,
    atr_values: np.ndarray,
    *,
    side: str = DISTRIBUTION,
    window: int | None = None,
    min_turnover: float = MIN_TURNOVER,
) -> Sided:
    """Одна сторона детектора по закрытым свечам ряда.

    `takerB` в детектор не входит, и его нейтральность не является
    контраргументом к вердикту: у OAX на всех двадцати барах распределения он
    лежал в 0.46–0.53 и не отличил распределение ни от чего. Различил только
    фитиль.

    Открытый интерес и фандинг тоже не входят — они принадлежат шагу 6 и
    отвечают на другой вопрос, кто набирает. Детектор обязан работать на
    монетах без деривативов, включая спот и разборы старше 30 суток.
    """
    if side not in _DIRECTION:
        raise ValueError(f"Неизвестная сторона {side!r}")
    if window is None:
        window = absorption_window(series.interval)

    turnover = _turnover_per_day(series, window)
    if not np.isnan(turnover) and turnover < min_turnover:
        return Sided(
            side=side, window=window, events=(), slope=None, shift_atr=None,
            verdict=VERDICT_NA,
            reason=(
                f"оборот {turnover / 1e6:.2f}M USDT за сутки ниже "
                f"{min_turnover / 1e6:.0f}M — фитиль может быть артефактом "
                f"одной сделки"
            ),
        )

    events = _collapse(series, side, _bars(series, side, window))
    slope = None
    if len(events) >= 2:
        extremes = np.array([e.extreme for e in events], dtype="float64")
        slope = float(np.polyfit(np.arange(len(extremes)), extremes, 1)[0])

    atr_value = float(atr_values[-1]) if len(atr_values) else float("nan")
    shift = None
    if len(events) >= 2 and atr_value == atr_value and atr_value > 0:
        move = events[-1].extreme - events[0].extreme
        # Знак приводится к стороне: у распределения «смещение 1.7 ATR»
        # означает падение, у накопления — рост.
        shift = float(move / atr_value * _DIRECTION[side])

    return Sided(
        side=side, window=window, events=events, slope=slope, shift_atr=shift,
        verdict=_verdict(side, events, slope, shift),
    )


def _verdict(
    side: str,
    events: tuple[Event, ...],
    slope: float | None,
    shift_atr: float | None,
) -> str:
    """Три исхода по §2.1.

    Растущие максимумы дают `none` независимо от числа событий — это
    единственное, что отделяет распределение от отката после пробоя. Эталон:
    ORN 01–03.06.2024, два бара проходят по объёму и фитилю, но событий два, а
    максимумы растут 1.794 → 1.810; дальше цена прошла +19% за две недели.
    """
    if len(events) < MIN_EVENTS:
        return VERDICT_NONE
    if slope is None or slope * _DIRECTION[side] <= 0:
        return VERDICT_NONE
    if shift_atr is None or shift_atr < SHIFT_ATR:
        return VERDICT_FORMING
    return VERDICT_CONFIRMED


@dataclass(frozen=True)
class Detector:
    """Обе стороны разом: у выдачи вердикт одной, а счёт событий — обеих."""

    distribution: Sided
    accumulation: Sided


def analyse(
    series: Series,
    atr_values: np.ndarray,
    *,
    window: int | None = None,
    min_turnover: float = MIN_TURNOVER,
) -> Detector:
    """Обе стороны одной процедурой."""
    kwargs = {"window": window, "min_turnover": min_turnover}
    return Detector(
        distribution=detect(series, atr_values, side=DISTRIBUTION, **kwargs),
        accumulation=detect(series, atr_values, side=ACCUMULATION, **kwargs),
    )
