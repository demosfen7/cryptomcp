"""Справочник символов и точность цен (PLAN §6.3).

Без tickSize SUI печатается как 0.7350000000000001, а BTC — как 108432.09999.

Справочник обслуживает оба рынка. У спота в exchangeInfo нет поля contractType
(бессрочных контрактов там не существует), зато есть isSpotTradingAllowed:
символ бывает в статусе TRADING, но недоступен для спотовой торговли.
"""

from __future__ import annotations

from dataclasses import dataclass

from .client import BinanceClient
from .errors import unknown_symbol

#: Котируемые активы, которые считаем стейблкоинами при отборе пар (ТЗ §8).
#: USD1, RLUSD и U добавлены по спотовой выдаче 02.09.2026: все трое стоят
#: 0.999–1.001 с суточным размахом в сотые доли процента и в сканере бесполезны.
#: Золотые PAXG и XAUT сюда не входят — они ходят вместе с металлом.
STABLE_BASES = frozenset({
    "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USDE", "EUR", "AEUR",
    "USD1", "RLUSD", "U",
})


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    base: str
    quote: str
    tick_size: float
    price_precision: int
    contract_type: str
    status: str
    market: str = "futures"
    spot_allowed: bool = True
    #: Момент листинга по бирже (`onboardDate`). У спота такого поля нет, там
    #: возраст берётся из архива по первой свече, и потому здесь 0, а не
    #: выдуманное число.
    onboard_ms: int = 0

    @property
    def is_perpetual(self) -> bool:
        return self.contract_type == "PERPETUAL"

    @property
    def is_stable_pair(self) -> bool:
        return self.base in STABLE_BASES


class SymbolRegistry:
    """Кэширует exchangeInfo и отдаёт сведения по символу."""

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._symbols: dict[str, SymbolInfo] | None = None

    async def all(self) -> dict[str, SymbolInfo]:
        if self._symbols is None:
            info = await self._client.exchange_info()
            result: dict[str, SymbolInfo] = {}
            for row in info.get("symbols", []):
                tick = _tick_size(row)
                market = self._client.market
                result[row["symbol"]] = SymbolInfo(
                    symbol=row["symbol"],
                    base=row.get("baseAsset", ""),
                    quote=row.get("quoteAsset", ""),
                    tick_size=tick,
                    price_precision=_precision_from_tick(tick),
                    contract_type=row.get(
                        "contractType", "" if market.has_derivatives else "SPOT"
                    ),
                    status=row.get("status", ""),
                    market=market.name,
                    spot_allowed=bool(row.get("isSpotTradingAllowed", True)),
                    onboard_ms=int(row.get("onboardDate") or 0),
                )
            self._symbols = result
        return self._symbols

    async def get(self, symbol: str) -> SymbolInfo:
        symbols = await self.all()
        info = symbols.get(symbol.upper())
        if info is None:
            raise unknown_symbol(symbol, self._client.market.name)
        return info

    async def tradable(self) -> list[SymbolInfo]:
        """Торгуемые пары к USDT без стейблкоин-пар.

        На фьючерсах — только перпетуалы: квартальные контракты живут по своему
        календарю и с бессрочными несопоставимы. На споте перпетуалов нет, зато
        есть символы в статусе TRADING с запрещённой спотовой торговлей.
        """
        symbols = await self.all()
        return [
            info for info in symbols.values()
            if info.status == "TRADING"
            and info.quote == "USDT" and not info.is_stable_pair
            and (info.is_perpetual if info.market == "futures" else info.spot_allowed)
        ]


def _tick_size(row: dict) -> float:
    for filt in row.get("filters", []):
        if filt.get("filterType") == "PRICE_FILTER":
            return float(filt["tickSize"])
    return 0.0


def _precision_from_tick(tick: float) -> int:
    """Сколько знаков после запятой имеет смысл печатать."""
    if tick <= 0:
        return 2
    text = f"{tick:.12f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def format_price(value: float, precision: int = 2) -> str:
    """Цена с разделителями тысяч и осмысленной точностью."""
    if value != value:  # NaN
        return "n/a"
    return f"{value:,.{precision}f}"
