"""HTTP-клиент к публичным рыночным данным Binance (PLAN §6.4, §6.6).

Один класс на оба рынка: различия фьючерсов и спота — хост, префикс пути,
веса и потолок свечей в ответе — описаны данными в markets.py. Логика запроса,
бюджет веса, ретраи и часы биржи от рынка не зависят.

Только публичные рыночные данные. Ни одного API-ключа в проекте нет и не
предполагается — все подписанные эндпоинты требуют ключ, поэтому торговые
операции невозможны независимо от хоста (PLAN §6.6).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .errors import (
    ErrorKind,
    ToolError,
    bad_params,
    symbol_closed,
    unknown_symbol,
)
from .markets import FUTURES, Market
from .ratelimit import WeightBudget

#: Одновременных запросов к бирже. Последовательный scan_pairs занимал бы
#: минуты, неограниченно параллельный уводит в 418 (PLAN §6.4).
DEFAULT_CONCURRENCY = 5

#: Максимум суммарного ожидания по Retry-After, прежде чем сдаться и вернуть
#: ошибку модели. Держать её в неведении дольше смысла нет.
MAX_RETRY_WAIT_S = 20.0

#: Как часто пересчитывать смещение часов относительно биржи. Процесс живёт на
#: сервере неделями, системные часы за это время уползают.
CLOCK_RESYNC_S = 1800.0

_MAX_ATTEMPTS = 3

#: Коды Binance, которые означают ошибку запроса, а не сбой.
_UNKNOWN_SYMBOL_CODES = {-1121}
#: Пара существует, но торги по ней закрыты: делистинг, поставка, расчёт или
#: ещё не начавшийся пре-трейдинг. Повтор не поможет ни через секунду, ни
#: через сутки, поэтому вид ошибки отдельный и неретраебельный.
#:
#: Набор узкий сознательно: -4108 получен живым запросом
#: (STORJUSDT, market=futures, 14.09.2026), остальные «похожие по смыслу»
#: коды сюда не добавлены. Ошибочно попавший код объявил бы неретраебельным
#: то, что стоило повторить, — а это отказ там, где данные были доступны.
_CLOSED_SYMBOL_CODES = {-4108}
_BAD_PARAM_CODES = {-1100, -1101, -1102, -1104, -1105, -1106, -1130, -1131}


class _TTLCache:
    """Примитивный кэш с временем жизни на ключ."""

    def __init__(self) -> None:
        self._data: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() >= expires_at:
            self._data.pop(key, None)
            return None
        return value

    def put(self, key: Any, value: Any, ttl_s: float) -> None:
        if ttl_s > 0:
            self._data[key] = (time.monotonic() + ttl_s, value)

    def clear(self) -> None:
        self._data.clear()


class BinanceClient:
    """Асинхронный клиент с бюджетом веса, семафором и учётом Retry-After."""

    def __init__(
        self,
        market: Market = FUTURES,
        *,
        base_url: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        budget: WeightBudget | None = None,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.market = market
        self.base_url = (base_url or market.base_url).rstrip("/")
        # Пулы веса у рынков раздельные: 2400/мин у фьючерсов, 6000 у спота.
        self._budget = budget or WeightBudget(market.weight_limit)
        self._sem = asyncio.Semaphore(concurrency)
        self._cache = _TTLCache()
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json"},
        )
        # Смещение часов относительно биржи. Определять закрытость свечи по
        # локальным часам нельзя: расхождение в минуту меняет результат.
        self._clock_offset_ms: int | None = None
        self._clock_synced_at: float = 0.0
        self._offset_lock = asyncio.Lock()
        # Момент, до которого биржа нас забанила (418). Пока он в будущем,
        # запросы отклоняются локально — ретраи продлевают бан.
        self._banned_until: float = 0.0

    async def __aenter__(self) -> BinanceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- время биржи --------------------------------------------------------

    async def server_time_ms(self) -> int:
        data = await self._request(self.market.path("time"), weight=1, cache_ttl_s=0)
        return int(data["serverTime"])

    async def sync_clock(self) -> int:
        """Определить смещение локальных часов относительно биржи."""
        async with self._offset_lock:
            before = time.time() * 1000
            server = await self.server_time_ms()
            after = time.time() * 1000
            # Середина интервала как оценка момента ответа биржи.
            self._clock_offset_ms = int(server - (before + after) / 2)
            self._clock_synced_at = time.monotonic()
            return self._clock_offset_ms

    async def now_ms(self) -> int:
        """Текущее время по часам биржи.

        Смещение пересчитывается периодически: процесс на сервере живёт неделями,
        а системные часы за это время уползают. Одной синхронизации при старте
        хватило бы только короткоживущему CLI.
        """
        stale = time.monotonic() - self._clock_synced_at > CLOCK_RESYNC_S
        if self._clock_offset_ms is None or stale:
            await self.sync_clock()
        return int(time.time() * 1000) + (self._clock_offset_ms or 0)

    # -- транспорт ----------------------------------------------------------

    async def _request(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        weight: int = 1,
        cache_ttl_s: float = 0.0,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = (path, tuple(sorted(params.items())))

        if cache_ttl_s > 0:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        self._check_ban()

        waited = 0.0
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            await self._budget.reserve(weight)
            try:
                async with self._sem:
                    response = await self._http.get(path, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == _MAX_ATTEMPTS:
                    raise ToolError(
                        ErrorKind.UPSTREAM_ERROR,
                        f"Сеть недоступна при обращении к {path}: {exc}",
                    ) from exc
                await asyncio.sleep(2**attempt * 0.5)
                continue

            self._budget.observe_header(_int_or_none(
                response.headers.get("x-mbx-used-weight-1m")
            ))

            if response.status_code == 200:
                data = response.json()
                self._cache.put(cache_key, data, cache_ttl_s)
                return data

            if response.status_code == 418:
                retry_after = _float_or_none(response.headers.get("retry-after"))
                self._banned_until = time.monotonic() + (retry_after or 120.0)
                raise ToolError(
                    ErrorKind.IP_BANNED,
                    "Binance забанил IP (HTTP 418). Повторы продлевают бан, "
                    "работа приостановлена.",
                    retry_after_s=retry_after,
                )

            if response.status_code == 429:
                # Retry-After от биржи точнее любой своей формулы (PLAN §6.4).
                retry_after = _float_or_none(response.headers.get("retry-after"))
                delay = retry_after if retry_after is not None else 2**attempt
                if attempt == _MAX_ATTEMPTS or waited + delay > MAX_RETRY_WAIT_S:
                    raise ToolError(
                        ErrorKind.RATE_LIMITED,
                        "Превышен лимит запросов Binance.",
                        retry_after_s=delay,
                    )
                waited += delay
                await asyncio.sleep(delay)
                continue

            if 400 <= response.status_code < 500:
                raise self._client_error(response, params)

            # 5xx — состояние запроса неизвестно, но запросы читающие, повтор безопасен.
            if attempt == _MAX_ATTEMPTS:
                raise ToolError(
                    ErrorKind.UPSTREAM_ERROR,
                    f"Binance вернул {response.status_code} на {path}.",
                )
            await asyncio.sleep(2**attempt * 0.5)

        raise ToolError(
            ErrorKind.UPSTREAM_ERROR,
            f"Не удалось получить {path}: {last_exc}",
        )

    def _check_ban(self) -> None:
        remaining = self._banned_until - time.monotonic()
        if remaining > 0:
            raise ToolError(
                ErrorKind.IP_BANNED,
                f"IP забанен биржей, осталось ~{int(remaining)} с.",
                retry_after_s=remaining,
            )

    def _client_error(self, response: httpx.Response, params: dict[str, Any]) -> ToolError:
        try:
            payload = response.json()
            code = int(payload.get("code", 0))
            msg = str(payload.get("msg", response.text[:200]))
        except Exception:
            code, msg = 0, response.text[:200]

        if code in _UNKNOWN_SYMBOL_CODES:
            return unknown_symbol(str(params.get("symbol", "?")), self.market.name)
        if code in _CLOSED_SYMBOL_CODES:
            return symbol_closed(
                str(params.get("symbol", "?")), self.market.name, msg
            )
        if code in _BAD_PARAM_CODES:
            return bad_params(f"Binance отклонил параметры: {msg}", code=code, **params)
        return ToolError(
            ErrorKind.UPSTREAM_ERROR,
            f"Binance вернул {response.status_code}: {msg}",
            details={"code": code},
        )

    # -- рыночные данные ----------------------------------------------------

    async def klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
        cache_ttl_s: float = 0.0,
    ) -> list[list[Any]]:
        top = self.market.max_limit
        if not 1 <= limit <= top:
            raise bad_params(
                f"limit={limit} вне диапазона 1..{top} для рынка {self.market.name}",
                limit=limit,
            )
        return await self._request(
            self.market.path("klines"),
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
            weight=self.market.klines_weight(limit),
            cache_ttl_s=cache_ttl_s,
        )

    async def exchange_info(self) -> dict[str, Any]:
        # tickSize и статус символов меняются редко (PLAN §6.3).
        return await self._request(
            self.market.path("exchangeInfo"),
            weight=self.market.exchange_info_weight,
            cache_ttl_s=24 * 3600,
        )

    async def ticker_24hr(self, symbol: str | None = None) -> Any:
        return await self._request(
            self.market.path("ticker/24hr"),
            params={"symbol": symbol.upper() if symbol else None},
            weight=(
                self.market.ticker_one_weight if symbol
                else self.market.ticker_all_weight
            ),
            cache_ttl_s=30,
        )

    async def premium_index(self, symbol: str) -> dict[str, Any]:
        return await self._request(
            "/fapi/v1/premiumIndex",
            params={"symbol": symbol.upper()},
            weight=1,
            cache_ttl_s=30,
        )

    async def funding_info(self) -> list[dict[str, Any]]:
        """Интервалы начисления фандинга по символам с нестандартной настройкой.

        Проверено: 770 символов, из них 4h у большинства, 8h у трети, 1h у трёх.
        Символы вне списка используют 8 часов (PLAN §4.12).
        """
        return await self._request(
            "/fapi/v1/fundingInfo", weight=1, cache_ttl_s=24 * 3600
        )

    async def funding_rate(
        self,
        symbol: str,
        *,
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self._request(
            "/fapi/v1/fundingRate",
            params={
                "symbol": symbol.upper(),
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
            weight=1,
            cache_ttl_s=300,
        )

    async def open_interest(self, symbol: str) -> dict[str, Any]:
        return await self._request(
            "/fapi/v1/openInterest",
            params={"symbol": symbol.upper()},
            weight=2,
            cache_ttl_s=60,
        )

    async def open_interest_hist(
        self,
        symbol: str,
        period: str = "4h",
        *,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """История открытого интереса.

        Биржа хранит только последние 30 суток — проверено отказом
        -1130 на запрос с startTime 60 суток назад (PLAN §4.12).
        """
        return await self._futures_data(
            "openInterestHist", symbol, period, limit, start_time, end_time
        )

    async def long_short_ratio(
        self,
        symbol: str,
        kind: str = "global_account",
        period: str = "5m",
        *,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Соотношение длинных и коротких позиций.

        Три разных среза, и путать их нельзя: ``global_account`` — доля всех
        счетов, ``top_account`` — доля крупных счетов, ``top_position`` — доля
        их объёма. Первый показывает настроение толпы, третий — куда встали
        деньги, и расходятся они регулярно.
        """
        paths = {
            "global_account": "globalLongShortAccountRatio",
            "top_account": "topLongShortAccountRatio",
            "top_position": "topLongShortPositionRatio",
        }
        if kind not in paths:
            raise bad_params(
                f"Неизвестный срез long/short: {kind!r}. "
                f"Доступны: {', '.join(paths)}",
                kind=kind,
            )
        return await self._futures_data(
            paths[kind], symbol, period, limit, start_time, end_time
        )

    async def taker_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        *,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Доля агрессивных покупок в объёме тейкеров."""
        return await self._futures_data(
            "takerlongshortRatio", symbol, period, limit, start_time, end_time
        )

    async def _futures_data(
        self,
        endpoint: str,
        symbol: str,
        period: str,
        limit: int,
        start_time: int | None,
        end_time: int | None,
    ) -> list[dict[str, Any]]:
        """Общий доступ к разделу /futures/data/.

        Раздел живёт вне префикса рынка и существует только у фьючерсов.
        Глубина у всех пяти эндпоинтов одна и та же — 30 суток, дальше -1130;
        ``limit`` проверен до 1000 точек за запрос, то есть 83 часа с шагом 5m.
        """
        if not self.market.has_derivatives:
            raise bad_params(
                f"{endpoint} существует только на фьючерсах, "
                f"а клиент работает с рынком {self.market.name}",
                market=self.market.name,
            )
        return await self._request(
            f"/futures/data/{endpoint}",
            params={
                "symbol": symbol.upper(),
                "period": period,
                "limit": limit,
                "startTime": start_time,
                "endTime": end_time,
            },
            weight=1,
            cache_ttl_s=0 if start_time or end_time else 300,
        )

    @property
    def budget(self) -> WeightBudget:
        return self._budget


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
