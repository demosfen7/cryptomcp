"""Поток тейкеров, поглощение и расширение объёма (SPEC-flow-and-absorption-v2).

`takerB` живёт в системе посвечно и средней за 30 свечей. Средняя растворяет
сигнал, но хуже другое: **пассивный набор регистрируется как продажа**.
Покупатель, стоящий лимитом, не тейкер; в него бьют рыночные продажи, и биржа
пишет их как sell. Поэтому `takerB` ниже нейтрали одинаково совместим и с
раздачей, и с тихим поглощением — различить можно только накопленным потоком,
сопоставленным с ходом цены.

Ни одна величина здесь не входит в `squeeze_index`: это отдельное измерение,
как и детектор распределения. Смешать их значило бы потерять возможность
сказать «сжатие сильное, поток нейтральный».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .series import Series
from .volume import baseline_series

#: Окна накопления потока в закрытых свечах. Печатаются все три: разворот
#: потока виден только на их сравнении.
WINDOWS = (30, 60, 90)

#: Ниже этой доли оборота поток неотличим от шума, и `absorption_ratio`
#: превратился бы в деление малого на малое.
MIN_DELTA_SHARE = 0.5

#: Ниже этого хода цены отношение печатается, но помечается: «цена стоит»
#: означает, что числитель сам по себе ничего не утверждает.
FLAT_PRICE = 1.0

#: Окна расширения объёма: последние N свечей к предыдущим M. Фиксированные и
#: непересекающиеся.
VOL_RECENT = 12
VOL_PREVIOUS = 30

#: Бар поглощения (§5): объём к сезонной базе, доля фитиля и потолок тела.
EVENT_VOLUME = 5.0
EVENT_WICK = 0.8
EVENT_BODY = 0.15

ABSORPTION = "поглощение"
DEMAND = "обычный спрос"
EXIT = "обычный выход"
DISTRIBUTION = "распределение"


@dataclass(frozen=True)
class Flow:
    """Накопленный поток за одно окно."""

    window: int
    bars: int
    turnover_sum: float
    buy_sum: float
    delta_sum: float
    delta_share: float
    price_change: float
    delta_slope: float | None
    quadrant: str
    #: `price_change / delta_share`; None, когда поток в пределах шума.
    absorption_ratio: float | None
    #: Почему отношение не посчитано или чем оговорено.
    ratio_note: str = ""

    @property
    def flagged(self) -> bool:
        """Первый квадрант — единственный, ради которого метрика заведена."""
        return self.quadrant == ABSORPTION


def _quadrant(delta_sum: float, price_change: float) -> str:
    """Четыре сочетания знаков и ничего больше (§1.3).

    Порогов на дельту намеренно нет: вердикт по знакам. AGLD и ANKR имеют
    одинаковую до десятой доли процента долю покупок (48.8%), и разводит их
    только ход цены — без `price_change` метрика бессмысленна.
    """
    if delta_sum < 0:
        return ABSORPTION if price_change > 0 else EXIT
    return DEMAND if price_change > 0 else DISTRIBUTION


def _ratio(delta_share: float, price_change: float) -> tuple[float | None, str]:
    """Во сколько обошёлся рынку этот поток.

    Знак дельты отвечает, в какую сторону шёл поток; он не отвечает, дорого
    ли поток обошёлся. Отрицательное отношение значит, что цена шла ПРОТИВ
    потока, — это и есть поглощение. Положительное и малое по модулю значит,
    что цену двигают дёшево: тонкий стакан, а не поглощение.
    """
    if abs(delta_share) < MIN_DELTA_SHARE:
        return None, (
            f"поток в пределах шума: |доля| {abs(delta_share):.2f}% ниже "
            f"{MIN_DELTA_SHARE}%"
        )
    ratio = price_change / delta_share
    if abs(price_change) < FLAT_PRICE:
        return ratio, f"цена стоит ({price_change:+.2f}% за окно)"
    return ratio, ""


def flow(series: Series, window: int) -> Flow | None:
    """Накопленный поток за окно закрытых свечей.

    Всё берётся из тех же полей, что печатает `get_klines`: оборот свечи и
    купленное по рынку. Своих источников метрика не заводит — иначе сумма по
    свечам выдачи не сошлась бы с `delta_sum` (приёмочный тест 11).

    Перпетуал и спот считаются раздельно и никогда не складываются: дельта на
    перпетуале может быть создана хеджем к спотовой позиции.
    """
    if len(series) < window:
        return None

    turnover = series.quote_volume[-window:]
    buy = series.col("taker_buy_quote")[-window:]
    delta_bar = 2 * buy - turnover

    turnover_sum = float(np.nansum(turnover))
    if turnover_sum <= 0:
        return None
    delta_sum = float(np.nansum(delta_bar))
    delta_share = delta_sum / turnover_sum * 100

    opens = series.col("open")[-window:]
    closes = series.close[-window:]
    price_change = (
        float(closes[-1] / opens[0] - 1) * 100 if opens[0] else float("nan")
    )

    # Наклон по КУМУЛЯТИВНОМУ ряду, нормированный на средний оборот свечи
    # окна: без нормировки наклон у монеты с оборотом 45M и у монеты с 3M
    # несопоставим, хотя вопрос к ним один и тот же.
    slope = None
    if window >= 2:
        mean_turnover = turnover_sum / window
        if mean_turnover > 0:
            cumulative = np.nancumsum(delta_bar) / mean_turnover
            slope = float(np.polyfit(np.arange(window), cumulative, 1)[0])

    ratio, note = _ratio(delta_share, price_change)
    return Flow(
        window=window,
        bars=window,
        turnover_sum=turnover_sum,
        buy_sum=float(np.nansum(buy)),
        delta_sum=delta_sum,
        delta_share=delta_share,
        price_change=price_change,
        delta_slope=slope,
        quadrant=_quadrant(delta_sum, price_change),
        absorption_ratio=ratio,
        ratio_note=note,
    )


def flows(series: Series, windows: tuple[int, ...] = WINDOWS) -> list[Flow]:
    """Поток по всем окнам, какие помещаются в ряд."""
    return [f for f in (flow(series, w) for w in windows) if f is not None]


@dataclass(frozen=True)
class VolRatio:
    """Расширение объёма: последние N свечей к предыдущим M."""

    value: float | None
    recent: int
    previous: int
    reason: str = ""


def vol_ratio(
    series: Series, recent: int = VOL_RECENT, previous: int = VOL_PREVIOUS
) -> VolRatio:
    """Во сколько оборот последних свечей отличается от предыдущих.

    **Почему не MA20/MA100.** Та колонка для этой задачи непригодна из-за
    неполного знаменателя у монет с короткой историей: замер по свечам дал
    ENSOUSDT 1d — 1.13x против 3.59 в колонке, ROBOUSDT 1d — 0.58x (оборот
    упал вдвое) против 1.92. Знак вывода противоположный.

    Окна фиксированные и непересекающиеся. Если предыдущих свечей нет целиком
    — `n/a` с причиной; считать по тому, что есть, запрещено: именно так и
    получается неполный знаменатель.
    """
    need = recent + previous
    volumes = series.quote_volume
    if len(volumes) < need:
        return VolRatio(
            None, recent, previous,
            f"нужно {need} закрытых свечей, есть {len(volumes)}",
        )
    tail = volumes[-recent:]
    base = volumes[-need:-recent]
    base_mean = float(np.nanmean(base))
    if not base_mean or np.isnan(base_mean):
        return VolRatio(None, recent, previous, "оборота в базовом окне нет")
    return VolRatio(float(np.nanmean(tail)) / base_mean, recent, previous)


@dataclass(frozen=True)
class AbsorptionEvent:
    """Одиночный бар поглощения или раздачи (§5)."""

    ts_ms: int
    volume_ratio: float
    wick: float
    body: float
    side: str


def absorption_events(
    series: Series, *, window: int, side: str = "buy"
) -> tuple[AbsorptionEvent, ...]:
    """Бары поглощения: исключение из правила «считать кластеры, а не бары».

    Эталон — ANKR 1h, 16.08.2026 18:00: объём 9.53x, тело −0.09% (4% диапазона
    свечи), нижний фитиль 88%. Цена внутри часа уходила на 0.003332 и
    вернулась к 0.003391. В дневном агрегате того же дня стояло 0.78x, и три
    часовых бара от 5x исчезли полностью.

    Порог по телу — то, что отличает поглощение от выноса стопов: у выноса
    тело большое и закрытие у края диапазона.

    `takerB` в определение НЕ входит: в эталонном баре он равен 0.38 и
    указывал бы в противоположную сторону — агрессором был продавец, а
    покупатель стоял лимитом.
    """
    baseline = baseline_series(series, window)
    ratios = baseline.ratio(series.quote_volume)
    opens, closes = series.col("open"), series.close
    high, low = series.high, series.low
    times = series.df["open_time"].to_numpy(dtype="int64")

    found: list[AbsorptionEvent] = []
    for i in range(max(0, len(series) - window), len(series)):
        span = high[i] - low[i]
        ratio = ratios[i]
        if span <= 0 or np.isnan(ratio) or ratio < EVENT_VOLUME:
            continue
        wick = (
            (min(opens[i], closes[i]) - low[i]) / span if side == "buy"
            else (high[i] - max(opens[i], closes[i])) / span
        )
        body = abs(closes[i] - opens[i]) / span
        if wick < EVENT_WICK or body > EVENT_BODY:
            continue
        found.append(AbsorptionEvent(
            ts_ms=int(times[i]), volume_ratio=float(ratio),
            wick=float(wick), body=float(body), side=side,
        ))
    return tuple(found)
