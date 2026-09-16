"""Рынки Binance: фьючерсы и спот (расширение ТЗ, открытый вопрос №1).

Перпетуал ценово производен от спота, и подтверждение пробоя объёмом логично
искать там, где происходит поставка. Плюс часть монет на фьючерсах либо тонкая,
либо не торгуется вовсе: UAIUSDT есть на фьючерсах и нет на споте, обратных
случаев ещё больше.

Различия между рынками — данные, а не логика: те же свечи из двенадцати полей,
те же индикаторы поверх. Поэтому здесь описание рынка, а не второй клиент.

Все значения проверены запросами 02.09.2026, а не взяты из документации:

| | futures | spot |
|---|---|---|
| хост | fapi.binance.com | api.binance.com |
| префикс | /fapi/v1 | /api/v3 |
| лимит веса на IP | 2400/мин | 6000/мин |
| вес klines | 1/2/5/10 по limit | **2 при любом limit** |
| максимум свечей за запрос | 1500 | **1000** (limit=1500 молча вернул 1000) |
| вес exchangeInfo | 1 | **20** |
| вес ticker/24hr | 1 / 40 | 2 / 80 |
| деривативы | есть | нет |
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .ratelimit import klines_weight as futures_klines_weight


def _spot_klines_weight(limit: int) -> int:
    """У спота вес klines не зависит от limit — проверено на 3 и на 1000."""
    return 2


def _futures_depth_weight(limit: int) -> int:
    """Вес /fapi/v1/depth, замеренный 16.09.2026 (README задания).

    Таблица из документации Binance оказалась таблицей спота: у фьючерсов
    уровни 5--50 стоят 2, 100 -- 5, 500 -- 10, 1000 -- 20 единиц. Значение
    5000 futures отвергает ещё до расхода веса, поэтому его нет в enum рынка.
    """
    if limit <= 50:
        return 2
    if limit == 100:
        return 5
    if limit == 500:
        return 10
    return 20


def _spot_depth_weight(limit: int) -> int:
    """Вес /api/v3/depth, замеренный 16.09.2026 (README задания)."""
    if limit <= 100:
        return 5
    if limit == 500:
        return 25
    if limit == 1000:
        return 50
    return 250


@dataclass(frozen=True)
class Market:
    """Всё, чем один рынок Binance отличается от другого."""

    name: str
    base_url: str
    prefix: str
    #: Максимум свечей в одном ответе. Спот молча обрезает до 1000.
    max_limit: int
    #: Лимит веса на IP в минуту. Пулы у рынков раздельные.
    weight_limit: int
    exchange_info_weight: int
    ticker_one_weight: int
    ticker_all_weight: int
    #: Вес /ticker/price для одного символа.
    ticker_price_weight: int
    #: Как рынок называется в выдаче и что дописывается к символу.
    label: str
    #: То же имя в одну колонку таблицы: «USDⓈ-M perp» в строку списка не
    #: влезает, а знать рынок построчно обязательно — ряды у них разные.
    short: str
    suffix: str
    has_derivatives: bool
    klines_weight: Callable[[int], int] = field(compare=False)
    #: Допустимые значения limit и функция веса L2-стакана. Это свойства
    #: рынка: futures и spot расходятся по обоим пунктам (README задания).
    depth_limits: frozenset[int] = field(default_factory=frozenset)
    depth_weight: Callable[[int], int] = field(default=lambda _: 1, compare=False)

    def path(self, endpoint: str) -> str:
        return f"{self.prefix}/{endpoint}"


FUTURES = Market(
    name="futures",
    base_url="https://fapi.binance.com",
    prefix="/fapi/v1",
    max_limit=1500,
    weight_limit=2400,
    exchange_info_weight=1,
    ticker_one_weight=1,
    ticker_all_weight=40,
    ticker_price_weight=1,
    label="USDⓈ-M perp",
    short="перп",
    suffix=".P",
    has_derivatives=True,
    klines_weight=futures_klines_weight,
    depth_limits=frozenset({5, 10, 20, 50, 100, 500, 1000}),
    depth_weight=_futures_depth_weight,
)

SPOT = Market(
    name="spot",
    base_url="https://api.binance.com",
    prefix="/api/v3",
    max_limit=1000,
    weight_limit=6000,
    exchange_info_weight=20,
    ticker_one_weight=2,
    ticker_all_weight=80,
    ticker_price_weight=2,
    label="спот",
    short="спот",
    suffix="",
    has_derivatives=False,
    klines_weight=_spot_klines_weight,
    depth_limits=frozenset({5, 10, 20, 50, 100, 500, 1000, 5000}),
    depth_weight=_spot_depth_weight,
)

MARKETS: dict[str, Market] = {FUTURES.name: FUTURES, SPOT.name: SPOT}


def market_short(name: str | None) -> str:
    """Рынок одним словом для таблиц и уведомлений.

    Пустое значение — не спот и не фьючерс, а «неизвестно»: так выглядят
    эпизоды, открытые до того, как рынок начали хранить. Подставлять им
    умолчание нельзя — именно молчаливое умолчание и выдавало спотовые числа
    за фьючерсные.
    """
    market = MARKETS.get(name or "")
    return market.short if market else "?"
