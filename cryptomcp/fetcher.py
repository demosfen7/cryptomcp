"""Модуль A1 из ТЗ: загрузка свечей с кэшем и отсечением незакрытой свечи.

Кэш живёт ровно столько, сколько имеет смысл: свеча 4h не перезапрашивается
четыре часа. Благодаря этому цепочка вызовов L1 → L2 → L3 по одной паре стоит
один сетевой запрос на таймфрейм, а не три (PLAN §6.4).
"""

from __future__ import annotations

import time
from typing import Any

from .client import BinanceClient
from .errors import insufficient_history
from .series import Series, build_series, interval_ms

#: Максимум свечей в одном ответе по умолчанию. Фактический потолок берётся у
#: рынка: у фьючерсов 1500, у спота 1000 (спот молча обрезает запрос сверх).
MAX_LIMIT = 1500

#: Потолок числа страниц при догрузке истории. Ограничивает и вес (страница
#: на 1500 свечей стоит 10 единиц), и задержку ответа. При 12 страницах
#: 60 суток достижимы вплоть до 5m; на 1m — нет, и это отражено в выдаче.
MAX_PAGES = 12

#: Доля интервала, в течение которой ответ считается свежим. Свеча всё равно
#: не изменится до своего закрытия, а на границе лучше сходить лишний раз.
_TTL_FRACTION = 0.5

#: Потолок кэша: даже недельную свечу перечитываем раз в 10 минут, чтобы
#: подхватывать поздние правки биржи и не держать протухший хвост.
_TTL_CAP_S = 600.0

#: Пол кэша: на 1m это почти реальное время (PLAN §5, раздел про 5m и 1m).
_TTL_FLOOR_S = 15.0


def cache_ttl_for(interval: str) -> float:
    """Время жизни кэша свечей для таймфрейма."""
    seconds = interval_ms(interval) / 1000
    return max(_TTL_FLOOR_S, min(_TTL_CAP_S, seconds * _TTL_FRACTION))


def pages_needed(interval: str, span_days: float, max_limit: int = MAX_LIMIT) -> int:
    """Сколько страниц по max_limit свечей нужно, чтобы покрыть span_days."""
    candles = span_days * 86_400_000 / interval_ms(interval)
    return max(1, -(-int(candles) // max_limit))


class CandleFetcher:
    """Отдаёт ряды закрытых свечей, скрывая кэш, пагинацию и часы биржи."""

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._cache: dict[Any, tuple[float, Series]] = {}

    async def get(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        as_of_ms: int | None = None,
        min_candles: int | None = None,
        target_span_days: float | None = None,
        max_pages: int = MAX_PAGES,
    ) -> Series:
        """Ряд закрытых свечей.

        ``as_of_ms`` — состояние на исторический момент (PLAN §4.6). Ответ
        детерминирован, поэтому кэшируется по тому же ключу без риска устареть.

        ``target_span_days`` — требуемый календарный охват. Если одной страницы
        не хватает, история догружается назад постранично, пока охват не наберётся
        либо не кончится история символа. Это то, что делает перцентили доступными
        на 1h и 15m: одной страницы там хватает лишь на 62 и 15 суток
        соответственно, а правилу §4.2 нужно 60.

        Пагинация не бесплатна: у фьючерсов полная страница стоит 10 единиц
        веса. Поэтому пакетные инструменты вроде scan_pairs её не запрашивают.
        """
        interval_ms(interval)  # ранняя валидация с понятным сообщением
        symbol = symbol.upper()

        key = (symbol, interval, limit, as_of_ms, target_span_days)
        cached = self._cache_get(key)
        if cached is not None:
            series = cached
        else:
            series = await self._load(
                symbol, interval, limit, as_of_ms, target_span_days, max_pages
            )
            self._cache[key] = (time.monotonic() + cache_ttl_for(interval), series)

        if min_candles is not None and len(series) < min_candles:
            raise insufficient_history(symbol, interval, len(series), min_candles)
        return series

    def _cache_get(self, key: Any) -> Series | None:
        item = self._cache.get(key)
        if item is None:
            return None
        expires_at, series = item
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        return series

    async def _load(
        self,
        symbol: str,
        interval: str,
        limit: int,
        as_of_ms: int | None,
        target_span_days: float | None,
        max_pages: int,
    ) -> Series:
        now_ms = await self._client.now_ms()
        # Для ретроспективы «сейчас» — это запрошенный момент: всё, что после
        # него, не должно попадать в расчёт даже как незакрытая свеча.
        effective_now = min(as_of_ms, now_ms) if as_of_ms is not None else now_ms

        top = self._client.market.max_limit
        page_limit = top if target_span_days else min(limit, top)
        pages_left = (
            pages_needed(interval, target_span_days, top) if target_span_days else 1
        )
        pages_left = min(pages_left, max_pages)

        rows: list[list[Any]] = []
        end_time = as_of_ms

        while pages_left > 0:
            chunk = await self._client.klines(
                symbol,
                interval,
                limit=page_limit,
                end_time=end_time,
                cache_ttl_s=cache_ttl_for(interval),
            )
            if not chunk:
                break
            rows = list(chunk) + rows
            pages_left -= 1
            if len(chunk) < page_limit:
                break  # достигнуто начало истории символа
            # Следующая страница — строго до открытия самой ранней свечи.
            end_time = int(chunk[0][0]) - 1

        return build_series(rows, symbol, interval, effective_now)
