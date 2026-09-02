"""Фандинг и открытый интерес (PLAN §4.12, расширение ТЗ §12).

Единственный доступный без стакана ответ на вопрос «откуда движение»: рост цены
при растущем открытом интересе — приток новых денег, рост при падающем —
закрытие шортов. Это констатация факта о прошедшем окне, а не прогноз, поэтому
ограничение ТЗ §1.1 не нарушается.

В squeeze_index не входит: формула ТЗ §4.3 остаётся нетронутой (PLAN §4.12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .client import BinanceClient
from .indicators import Metric, with_percentile

HOURS_PER_YEAR = 24 * 365

#: Интервал начисления для символов, отсутствующих в /fapi/v1/fundingInfo.
DEFAULT_FUNDING_INTERVAL_H = 8

#: История открытого интереса обрезана биржей 30 сутками — проверено отказом
#: -1130 на запрос с startTime 60 суток назад. Единственная причина, по которой
#: правило §4.2 о календарном охвате для OI смягчается.
OI_HISTORY_DAYS = 30
OI_SPAN_EXEMPTION = "история OI ограничена биржей 30 сутками"

#: Порог значимости изменения OI для классификации, доля.
DEFAULT_OI_THRESHOLD = 0.01

#: Сколько точек ряда отдавать в выдачу. Три дельты показывают итог окна, ряд
#: показывает, КОГДА поток развернулся. Данные для него уже загружены ради
#: дельт, поэтому ряд не стоит ни одного дополнительного запроса.
OI_HISTORY_POINTS = 24
FUNDING_HISTORY_POINTS = 8

Quadrant = Literal[
    "приток новых денег",
    "закрытие шортов",
    "новые шорты",
    "закрытие лонгов",
    "набор позиций без движения цены",
    "разгрузка позиций без движения цены",
    "движение без притока (ротация)",
    "без выраженного потока",
]


@dataclass(frozen=True)
class Funding:
    """Ставка финансирования, приведённая к сопоставимому виду."""

    symbol: str
    rate: float
    interval_hours: int
    next_funding_ms: int
    mark_price: float
    index_price: float
    percentile: Metric | None = None
    #: Последние начисления, старые→новые: (время, ставка).
    history: tuple[tuple[int, float], ...] = ()

    @property
    def annualized_pct(self) -> float:
        """Годовая ставка в процентах.

        Без этой нормализации сравнение между монетами бессмысленно: интервал
        начисления различается — по данным биржи 4 часа у большинства символов,
        8 у трети, 1 час у трёх. Ставка 0.01% за час и 0.01% за восемь часов
        отличаются в восемь раз (PLAN §4.12).
        """
        return self.rate * (HOURS_PER_YEAR / self.interval_hours) * 100.0

    @property
    def basis_pct(self) -> float:
        """Отклонение маркировочной цены от индексной."""
        if not self.index_price:
            return float("nan")
        return (self.mark_price - self.index_price) / self.index_price * 100.0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rate_pct": round(self.rate * 100, 6),
            "interval_hours": self.interval_hours,
            "annualized_pct": round(self.annualized_pct, 2),
            "basis_pct": round(self.basis_pct, 4),
            "next_funding_ms": self.next_funding_ms,
        }
        if self.percentile is not None:
            payload["percentile"] = self.percentile.to_dict()
        return payload


@dataclass(frozen=True)
class OpenInterest:
    """Открытый интерес и его динамика."""

    symbol: str
    #: Число контрактов в базовом активе — по нему считается динамика.
    contracts: float
    #: Стоимость позиций в USDT — только для масштаба.
    notional_usdt: float
    #: Изменение числа контрактов по окнам, доля: {"1h": 0.012, ...}
    change: dict[str, float]
    #: Изменение цены за те же окна, доля.
    price_change: dict[str, float]
    percentile: Metric | None = None
    #: Почасовой ряд, старые→новые: (время, контракты, цена).
    history: tuple[tuple[int, float, float], ...] = ()

    def quadrant(
        self, window: str, threshold: float = DEFAULT_OI_THRESHOLD
    ) -> Quadrant:
        return classify_price_oi(
            self.price_change.get(window, 0.0),
            self.change.get(window, 0.0),
            threshold,
        )

    def to_dict(self, threshold: float = DEFAULT_OI_THRESHOLD) -> dict[str, object]:
        payload: dict[str, object] = {
            "contracts": self.contracts,
            "notional_usdt": round(self.notional_usdt, 2),
            "windows": {
                window: {
                    "oi_change_pct": round(value * 100, 2),
                    "price_change_pct": round(self.price_change.get(window, 0.0) * 100, 2),
                    "reading": self.quadrant(window, threshold),
                }
                for window, value in self.change.items()
            },
        }
        if self.percentile is not None:
            payload["percentile"] = self.percentile.to_dict()
        return payload


def classify_price_oi(
    price_change: float,
    oi_change: float,
    threshold: float = DEFAULT_OI_THRESHOLD,
) -> Quadrant:
    """Квадранты «цена × открытый интерес» (PLAN §4.12).

    Динамика считается по числу контрактов, а не по стоимости позиций в USDT:
    при росте цены на 5% стоимость OI вырастет на те же 5% даже если ни одна
    новая позиция не открылась. Классификация по стоимости показывала бы приток
    новых денег там, где его нет.

    Помимо четырёх основных квадрантов различаются вырожденные случаи, и они не
    менее содержательны. Заметный рост OI при стоящей цене — это то, что ТЗ §4.2
    называет «набор позиции без движения цены», главный признак накопления.
    Падение OI при стоящей цене — обратное по смыслу: позиции закрываются, и
    путать одно с другим нельзя. Движение цены без изменения OI означает переход
    позиций между участниками, а не приход новых денег.
    """
    oi_moved = abs(oi_change) >= threshold
    price_moved = abs(price_change) >= threshold

    if not oi_moved and not price_moved:
        return "без выраженного потока"
    if oi_moved and not price_moved:
        return (
            "набор позиций без движения цены" if oi_change > 0
            else "разгрузка позиций без движения цены"
        )
    if price_moved and not oi_moved:
        return "движение без притока (ротация)"

    if price_change > 0:
        return "приток новых денег" if oi_change > 0 else "закрытие шортов"
    return "новые шорты" if oi_change > 0 else "закрытие лонгов"


class DerivativesReader:
    """Собирает фандинг и открытый интерес по символу."""

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._intervals: dict[str, int] | None = None

    async def funding_intervals(self) -> dict[str, int]:
        if self._intervals is None:
            info = await self._client.funding_info()
            self._intervals = {
                row["symbol"]: int(row["fundingIntervalHours"]) for row in info
            }
        return self._intervals

    async def funding(self, symbol: str, *, history: bool = True) -> Funding:
        symbol = symbol.upper()
        premium = await self._client.premium_index(symbol)
        intervals = await self.funding_intervals()
        interval_h = intervals.get(symbol, DEFAULT_FUNDING_INTERVAL_H)
        rate = float(premium["lastFundingRate"])

        percentile: Metric | None = None
        settlements: tuple[tuple[int, float], ...] = ()
        if history:
            # История фандинга глубокая — проверено на 900 суток назад,
            # поэтому правило §4.2 выполняется без всяких послаблений.
            rows = await self._client.funding_rate(symbol, limit=1000)
            if rows:
                values = np.array([float(r["fundingRate"]) for r in rows])
                span_days = (
                    int(rows[-1]["fundingTime"]) - int(rows[0]["fundingTime"])
                ) / 86_400_000
                percentile = with_percentile(
                    "funding", rate, values[:-1], span_days,
                    unit="", threshold=90, threshold_side="above",
                )
                settlements = tuple(
                    (int(r["fundingTime"]), float(r["fundingRate"]))
                    for r in rows[-FUNDING_HISTORY_POINTS:]
                )

        return Funding(
            symbol=symbol,
            rate=rate,
            interval_hours=interval_h,
            next_funding_ms=int(premium["nextFundingTime"]),
            mark_price=float(premium["markPrice"]),
            index_price=float(premium["indexPrice"]),
            percentile=percentile,
            history=settlements,
        )

    async def open_interest(
        self,
        symbol: str,
        *,
        period: str = "5m",
        windows: tuple[str, ...] = ("1h", "4h", "24h"),
        history_period: str = "1h",
    ) -> OpenInterest:
        """Открытый интерес: дельты по окнам, почасовой ряд и перцентиль.

        Два запроса истории вместо одного, и вот почему. Пятиминутный шаг нужен
        дельтам: он даёт свежий отсчёт, отставший от «сейчас» не больше чем на
        пять минут. Но 500 пятиминутных точек — это всего 41 час, и перцентиль
        на такой базе означал бы «против позавчера», а не против месяца.
        Часовой шаг за те же 500 точек покрывает 20 суток — близко к тому
        максимуму в 30 суток, который вообще хранит биржа. Каждый запрос стоит
        единицу веса и кэшируется на пять минут.
        """
        symbol = symbol.upper()
        current = await self._client.open_interest(symbol)
        rows = await self._client.open_interest_hist(symbol, period=period, limit=500)
        long_rows = await self._client.open_interest_hist(
            symbol, period=history_period, limit=500
        )

        contracts = float(current["openInterest"])
        if not rows:
            return OpenInterest(symbol, contracts, 0.0, {}, {})

        oi_series = np.array([float(r["sumOpenInterest"]) for r in rows])
        value_series = np.array([float(r["sumOpenInterestValue"]) for r in rows])
        # Цена восстанавливается из пары «стоимость / контракты»: отдельный
        # запрос свечей ради этого не нужен, а моменты замеров совпадают точно.
        price_series = np.divide(
            value_series, oi_series,
            out=np.full_like(value_series, np.nan), where=oi_series > 0,
        )

        step_minutes = _period_minutes(period)
        change: dict[str, float] = {}
        price_change: dict[str, float] = {}
        for window in windows:
            back = _window_minutes(window) // step_minutes
            if back <= 0 or back >= len(oi_series):
                continue
            change[window] = _pct_change(oi_series[-1], oi_series[-1 - back])
            price_change[window] = _pct_change(price_series[-1], price_series[-1 - back])

        # Ряд и перцентиль — по длинной истории; дельты выше — по короткой.
        base_rows = long_rows or rows
        base_step = _period_minutes(history_period if long_rows else period)
        base_oi = np.array([float(r["sumOpenInterest"]) for r in base_rows])
        base_value = np.array([float(r["sumOpenInterestValue"]) for r in base_rows])
        base_price = np.divide(
            base_value, base_oi,
            out=np.full_like(base_value, np.nan), where=base_oi > 0,
        )
        tail = slice(max(0, len(base_rows) - OI_HISTORY_POINTS), len(base_rows))
        oi_history = tuple(
            (int(base_rows[i]["timestamp"]), float(base_oi[i]), float(base_price[i]))
            for i in range(tail.start, tail.stop)
        )

        span_days = len(base_oi) * base_step / (60 * 24)
        percentile = with_percentile(
            "open_interest", base_oi[-1], base_oi[:-1], span_days,
            span_exemption=OI_SPAN_EXEMPTION,
        )

        return OpenInterest(
            symbol=symbol,
            contracts=contracts,
            notional_usdt=float(value_series[-1]),
            change=change,
            price_change=price_change,
            percentile=percentile,
            history=oi_history,
        )


def _pct_change(current: float, previous: float) -> float:
    if not previous or np.isnan(previous) or np.isnan(current):
        return 0.0
    return (current - previous) / previous


def _period_minutes(period: str) -> int:
    table = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
             "4h": 240, "6h": 360, "12h": 720, "1d": 1440}
    if period not in table:
        raise ValueError(f"Неподдерживаемый период истории OI: {period!r}")
    return table[period]


def _window_minutes(window: str) -> int:
    unit = window[-1]
    amount = int(window[:-1])
    if unit == "m":
        return amount
    if unit == "h":
        return amount * 60
    if unit == "d":
        return amount * 1440
    raise ValueError(f"Неразборчивое окно: {window!r}")
