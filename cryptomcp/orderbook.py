"""Расчёты L2-стакана без форматирования (SPEC-order-book-v1 A.4--A.5).

REST-snapshot не имеет своей временной шкалы: он отвечает состоянием «сейчас».
Поэтому модуль намеренно не хранит данные и не выводит причин о стенах. База
размеров уровней потребовала бы сотен снимков, а временные сессии из части B
не являются таким архивом (PLAN §4.37).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .distribution import MIN_TURNOVER

#: Диапазоны из A.5.1. Свой набор допустим, но отсутствие параметра не должно
#: превращать запрос в выдачу без сравнительной базы.
DEFAULT_DEPTH_PCTS = (0.25, 0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class BookLevel:
    """Один уровень и накопительный notional от лучшей цены (A.4)."""

    price: float
    qty: float
    notional_usdt: float
    cum_notional_usdt: float


@dataclass(frozen=True)
class DepthRange:
    """Агрегаты одной симметричной дистанции от середины книги."""

    depth_pct: float
    bid_notional_usdt: float
    ask_notional_usdt: float
    imbalance: float | None
    imbalance_reason: str | None
    bid_coverage_pct: float
    ask_coverage_pct: float

    @property
    def fully_covered(self) -> bool:
        return (
            self.bid_coverage_pct + 1e-12 >= self.depth_pct
            and self.ask_coverage_pct + 1e-12 >= self.depth_pct
        )


@dataclass(frozen=True)
class OrderBook:
    """Нормализованный один снимок, готовый для рендера и сессии B."""

    timestamp_ms: int
    last_price: float
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    spread_pct: float
    limit: int
    turnover_24h_usdt: float
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    ranges: tuple[DepthRange, ...]


def normalise_depth_pcts(value: float | list[float] | None) -> tuple[float, ...]:
    """Проверить пользовательский набор диапазонов без округления.

    Параметр A.2 — процент, а не число уровней: ноль и отрицательные значения
    не имеют смысла. Верхнюю границу намеренно не придумываем: инструмент
    честно сообщит, что конечный REST-snapshot её не покрывает (A.2.1).
    """
    if value is None:
        return DEFAULT_DEPTH_PCTS
    values = value if isinstance(value, list) else [value]
    result: list[float] = []
    for item in values:
        try:
            pct = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"depth_pct содержит нечисловое значение {item!r}") from exc
        if not math.isfinite(pct) or pct <= 0:
            raise ValueError("depth_pct должен содержать конечные числа больше нуля")
        if pct not in result:
            result.append(pct)
    if not result:
        raise ValueError("depth_pct не должен быть пустым")
    return tuple(result)


def build_order_book(
    snapshot: dict[str, Any],
    *,
    timestamp_ms: int,
    last_price: float,
    limit: int,
    depth_pcts: tuple[float, ...],
    turnover_24h_usdt: float,
) -> OrderBook:
    """Собрать A.4 и A.5.1 из ответа Binance без сетевых запросов.

    Все проценты глубины считаются от середины ТОГО ЖЕ снимка, а не от
    ``last_price`` из отдельного запроса (PLAN §4.37). Так
    выбранный уровень и его дистанция проверяются по сырому блоку вручную.
    """
    bids = _levels(snapshot.get("bids", ()), reverse=True)
    asks = _levels(snapshot.get("asks", ()), reverse=False)
    if not bids or not asks:
        raise ValueError("Binance вернул стакан без bids или asks")

    best_bid = bids[0].price
    best_ask = asks[0].price
    mid_price = (best_bid + best_ask) / 2.0
    if mid_price <= 0:
        raise ValueError("середина стакана должна быть больше нуля")
    spread = best_ask - best_bid

    return OrderBook(
        timestamp_ms=timestamp_ms,
        last_price=last_price,
        mid_price=mid_price,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        spread_pct=spread / mid_price * 100.0,
        limit=limit,
        turnover_24h_usdt=turnover_24h_usdt,
        bids=bids,
        asks=asks,
        ranges=tuple(
            _range(bids, asks, mid_price, pct, turnover_24h_usdt)
            for pct in depth_pcts
        ),
    )


def _levels(rows: Any, *, reverse: bool) -> tuple[BookLevel, ...]:
    parsed = sorted(
        ((float(row[0]), float(row[1])) for row in rows), reverse=reverse
    )
    total = 0.0
    levels: list[BookLevel] = []
    for price, qty in parsed:
        notional = price * qty
        total += notional
        levels.append(BookLevel(price, qty, notional, total))
    return tuple(levels)


def _range(
    bids: tuple[BookLevel, ...],
    asks: tuple[BookLevel, ...],
    mid_price: float,
    depth_pct: float,
    turnover_24h_usdt: float,
) -> DepthRange:
    lower = mid_price * (1.0 - depth_pct / 100.0)
    upper = mid_price * (1.0 + depth_pct / 100.0)
    bid_notional = sum(level.notional_usdt for level in bids if level.price >= lower)
    ask_notional = sum(level.notional_usdt for level in asks if level.price <= upper)
    total = bid_notional + ask_notional

    if turnover_24h_usdt < MIN_TURNOVER:
        imbalance = None
        reason = (
            f"оборот 24ч {turnover_24h_usdt / 1e6:.2f}M USDT ниже "
            f"{MIN_TURNOVER / 1e6:.0f}M"
        )
    elif total <= 0:
        imbalance = None
        reason = "в выбранном диапазоне нет видимой ликвидности"
    else:
        imbalance = (bid_notional - ask_notional) / total
        reason = None

    return DepthRange(
        depth_pct=depth_pct,
        bid_notional_usdt=bid_notional,
        ask_notional_usdt=ask_notional,
        imbalance=imbalance,
        imbalance_reason=reason,
        bid_coverage_pct=(mid_price - bids[-1].price) / mid_price * 100.0,
        ask_coverage_pct=(asks[-1].price - mid_price) / mid_price * 100.0,
    )
