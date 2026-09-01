"""Учёт веса запросов ДО отправки (PLAN §6.4).

Реактивный backoff после 429 недостаточен: пакетный вызов вроде scan_pairs
отправляет полсотни запросов залпом и упирается в лимит раньше, чем придёт
первый отказ. Поэтому вес резервируется заранее, а фактическое значение из
заголовка X-MBX-USED-WEIGHT-1M используется для сверки — сервер публичный,
и лимит расходуется не только нами.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

#: Лимит веса на IP для Binance USDⓈ-M Futures, единиц в минуту.
FUTURES_IP_WEIGHT_LIMIT = 2400

#: Доля лимита, которую разрешаем себе занимать. Запас нужен на расхождение
#: между нашим скользящим окном и фиксированной минутой на стороне биржи.
DEFAULT_SAFETY_FACTOR = 0.8

_WINDOW_S = 60.0


def klines_weight(limit: int) -> int:
    """Вес запроса /fapi/v1/klines в зависимости от limit.

    Проверено эмпирически: limit=1 возвращает x-mbx-used-weight-1m: 1.
    """
    if limit <= 100:
        return 1
    if limit <= 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class WeightBudget:
    """Скользящее окно веса за минуту с резервированием до отправки."""

    def __init__(
        self,
        limit: int = FUTURES_IP_WEIGHT_LIMIT,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        *,
        clock=time.monotonic,
    ) -> None:
        if not 0 < safety_factor <= 1:
            raise ValueError("safety_factor должен быть в диапазоне (0, 1]")
        self.limit = limit
        self.usable = max(1, int(limit * safety_factor))
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        # Последнее значение, сообщённое биржей, и момент его получения.
        self._reported: int = 0
        self._reported_at: float = 0.0

    # -- внутреннее ---------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - _WINDOW_S
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    def _local_used(self, now: float) -> int:
        self._prune(now)
        return sum(w for _, w in self._events)

    def used(self, now: float | None = None) -> int:
        """Оценка занятого веса: максимум из своего счёта и отчёта биржи.

        Отчёт биржи учитывается только пока он свежий — он относится к текущей
        минуте на стороне сервера, а не к нашему скользящему окну.
        """
        now = self._clock() if now is None else now
        local = self._local_used(now)
        if now - self._reported_at <= _WINDOW_S:
            return max(local, self._reported)
        return local

    def _wait_time(self, weight: int, now: float) -> float:
        """Сколько ждать, чтобы вес поместился в окно. 0 — можно слать сразу."""
        if self.used(now) + weight <= self.usable:
            return 0.0
        if not self._events:
            # Окно пусто, но биржа считает нас занятыми — ждём истечения её отчёта.
            return max(0.0, self._reported_at + _WINDOW_S - now)
        # Ждём, пока из окна выпадет достаточно старых событий.
        need_to_free = self.used(now) + weight - self.usable
        freed = 0
        for ts, w in self._events:
            freed += w
            if freed >= need_to_free:
                return max(0.0, ts + _WINDOW_S - now)
        return max(0.0, self._events[-1][0] + _WINDOW_S - now)

    # -- публичное ----------------------------------------------------------

    async def reserve(self, weight: int) -> None:
        """Дождаться места в бюджете и записать вес как израсходованный."""
        if weight <= 0:
            return
        if weight > self.usable:
            raise ValueError(
                f"Вес {weight} превышает доступный бюджет {self.usable} — "
                "запрос не может быть выполнен ни при каких условиях."
            )
        async with self._lock:
            while True:
                now = self._clock()
                delay = self._wait_time(weight, now)
                if delay <= 0:
                    self._events.append((now, weight))
                    return
                await asyncio.sleep(delay)

    def observe_header(self, used_weight: int | None) -> None:
        """Принять фактическое значение X-MBX-USED-WEIGHT-1M от биржи."""
        if used_weight is None:
            return
        self._reported = used_weight
        self._reported_at = self._clock()

    def snapshot(self) -> dict[str, int]:
        return {
            "used": self.used(),
            "usable": self.usable,
            "limit": self.limit,
            "reported_by_exchange": self._reported,
        }
