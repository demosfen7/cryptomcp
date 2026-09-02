"""Тесты поддержки двух рынков (расширение ТЗ, открытый вопрос №1).

Числа в описании рынков взяты из живых ответов Binance, а не из документации,
и здесь закреплены: спот отличается не только хостом, но и потолком свечей в
ответе, весами и отсутствием деривативов.
"""

from __future__ import annotations

import pytest

from cryptomcp.analysis import TimeframeView
from cryptomcp.errors import ErrorKind, unknown_symbol
from cryptomcp.fetcher import pages_needed
from cryptomcp.indicators import Metric
from cryptomcp.journal import Journal
from cryptomcp.levels import VolumeProfile
from cryptomcp.markets import FUTURES, MARKETS, SPOT
from cryptomcp.render import render_snapshot
from cryptomcp.symbols import STABLE_BASES, SymbolInfo, SymbolRegistry
from cryptomcp.volume import VolumeContext

INFO = SymbolInfo("TESTUSDT", "TEST", "USDT", 0.0001, 4, "PERPETUAL", "TRADING")


def view(**overrides) -> TimeframeView:
    """Заготовка представления.

    Своя, а не общая с test_render: импорт одного тест-модуля из другого
    требует корня репозитория в sys.path, а он там оказывается только при
    editable-установке. В CI такой импорт падал.
    """
    defaults = dict(
        interval="4h", price=100.0, atr_value=2.0, atr_pct=2.0, rsi_value=50.0,
        ema_state="above", structure="HH/HL", position_in_range=0.5,
        bbw=Metric("BBW", 0.04, pct_rank=10.0, n_obs=360, span_days=90),
        atr_metric=Metric("ATR", 2.0, pct_rank=10.0, n_obs=360, span_days=90),
        atr_declining_bars=10,
        range_low=95.0, range_high=105.0, range_width=0.10, range_width_atr=5.0,
        range_threshold=0.06, narrow_bars=0,
        volume=VolumeContext(0.7, "медиана слота", 250, 0.6, 3, 0.55),
        profile=VolumeProfile(100.0, 95.0, 105.0, 1e6, 60),
        divergence=None,
        meta={"closed_through_ms": 4 * 3_600_000 - 1, "missing": 0},
    )
    defaults.update(overrides)
    return TimeframeView(**defaults)


class FakeClient:
    def __init__(self, market, rows):
        self.market = market
        self._rows = rows

    async def exchange_info(self):
        return {"symbols": self._rows}


def spot_row(symbol="CAKEUSDT", base="CAKE", status="TRADING", allowed=True):
    return {
        "symbol": symbol, "baseAsset": base, "quoteAsset": "USDT",
        "status": status, "isSpotTradingAllowed": allowed,
        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.00100000"}],
    }


class TestMarketDescriptors:
    def test_paths_share_one_prefix(self):
        assert FUTURES.path("klines") == "/fapi/v1/klines"
        assert SPOT.path("klines") == "/api/v3/klines"
        assert SPOT.path("ticker/24hr") == "/api/v3/ticker/24hr"

    def test_spot_page_is_smaller(self):
        """Спот на limit=1500 молча вернул 1000 — потолок должен это знать."""
        assert FUTURES.max_limit == 1500
        assert SPOT.max_limit == 1000

    def test_spot_klines_weight_is_flat(self):
        """У спота limit=3 и limit=1000 стоили одинаково."""
        assert SPOT.klines_weight(3) == SPOT.klines_weight(1000) == 2

    def test_futures_klines_weight_grows_with_limit(self):
        assert FUTURES.klines_weight(100) == 1
        assert FUTURES.klines_weight(1500) == 10

    def test_weight_pools_are_separate_sizes(self):
        assert FUTURES.weight_limit == 2400
        assert SPOT.weight_limit == 6000

    def test_only_futures_have_derivatives(self):
        assert FUTURES.has_derivatives
        assert not SPOT.has_derivatives

    def test_registry_lists_both(self):
        assert set(MARKETS) == {"futures", "spot"}


class TestPaginationRespectsPageSize:
    def test_same_span_needs_more_pages_on_spot(self):
        futures_pages = pages_needed("15m", 60, FUTURES.max_limit)
        spot_pages = pages_needed("15m", 60, SPOT.max_limit)
        assert futures_pages == 4
        assert spot_pages == 6


class TestSpotSymbols:
    @pytest.mark.asyncio
    async def test_missing_contract_type_becomes_spot(self):
        registry = SymbolRegistry(FakeClient(SPOT, [spot_row()]))
        info = await registry.get("CAKEUSDT")
        assert info.contract_type == "SPOT"
        assert info.market == "spot"
        assert not info.is_perpetual

    @pytest.mark.asyncio
    async def test_tick_size_still_drives_precision(self):
        registry = SymbolRegistry(FakeClient(SPOT, [spot_row()]))
        assert (await registry.get("CAKEUSDT")).price_precision == 3

    @pytest.mark.asyncio
    async def test_trading_but_not_spot_allowed_is_excluded(self):
        """Статус TRADING на споте ещё не значит, что торговать можно."""
        rows = [spot_row(), spot_row("XXXUSDT", "XXX", allowed=False)]
        registry = SymbolRegistry(FakeClient(SPOT, rows))
        assert [i.symbol for i in await registry.tradable()] == ["CAKEUSDT"]

    @pytest.mark.asyncio
    async def test_futures_keeps_perpetual_only(self):
        rows = [
            {"symbol": "AUSDT", "baseAsset": "A", "quoteAsset": "USDT",
             "status": "TRADING", "contractType": "PERPETUAL", "filters": []},
            {"symbol": "BUSDT", "baseAsset": "B", "quoteAsset": "USDT",
             "status": "TRADING", "contractType": "CURRENT_QUARTER", "filters": []},
        ]
        registry = SymbolRegistry(FakeClient(FUTURES, rows))
        assert [i.symbol for i in await registry.tradable()] == ["AUSDT"]

    def test_pegged_assets_are_excluded(self):
        """USD1, RLUSD и U стоят 0.999–1.001 — в сканере им делать нечего."""
        for base in ("USD1", "RLUSD", "U"):
            assert SymbolInfo("X", base, "USDT", 0.01, 2, "SPOT", "TRADING").is_stable_pair

    def test_gold_is_not_a_stablecoin(self):
        assert "PAXG" not in STABLE_BASES
        assert "XAUT" not in STABLE_BASES


class TestUnknownSymbolPointsAtTheOtherMarket:
    def test_spot_error_suggests_futures(self):
        error = unknown_symbol("UAIUSDT", "spot")
        assert error.kind is ErrorKind.UNKNOWN_SYMBOL
        assert "рынке spot" in error.message
        assert "market='futures'" in error.message
        assert error.details["market"] == "spot"

    def test_futures_error_suggests_spot(self):
        assert "market='spot'" in unknown_symbol("XXXUSDT", "futures").message


class TestSpotRendering:
    def test_header_names_the_market(self):
        text = render_snapshot(
            INFO, {"4h": view()}, live_price=1.5, change_24h=1.0,
            quote_volume_24h=3.9e6, now_ms=0, market=SPOT,
        )
        assert text.startswith("TESTUSDT (спот)")
        assert "TESTUSDT.P" not in text

    def test_small_turnover_is_shown_in_millions(self):
        """У спота обороты на два порядка меньше: 0.00B ничего не сообщает."""
        text = render_snapshot(
            INFO, {"4h": view()}, live_price=1.5, change_24h=1.0,
            quote_volume_24h=3.9e6, now_ms=0, market=SPOT,
        )
        assert "оборот 3.9M USDT" in text

    def test_absent_derivatives_are_explained(self):
        text = render_snapshot(
            INFO, {"4h": view()}, live_price=1.5, change_24h=1.0,
            quote_volume_24h=3.9e6, now_ms=0, market=SPOT,
        )
        assert "у спота не существует" in text

    def test_futures_says_nothing_about_spot(self):
        text = render_snapshot(
            INFO, {"4h": view()}, live_price=1.5, change_24h=1.0,
            quote_volume_24h=2e9, now_ms=0,
        )
        assert text.startswith("TESTUSDT.P (USDⓈ-M perp)")
        assert "у спота не существует" not in text


class TestJournalKeepsMarketsApart:
    def test_market_is_written(self, tmp_path):
        import json

        path = tmp_path / "squeeze.jsonl"
        journal = Journal(str(path))
        journal.record("CAKEUSDT", view(squeeze_index=0.5), market="spot")

        entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert entry["market"] == "spot"
        assert entry["symbol"] == "CAKEUSDT"

    def test_futures_is_the_default(self, tmp_path):
        import json

        path = tmp_path / "squeeze.jsonl"
        Journal(str(path)).record("CAKEUSDT", view(squeeze_index=0.5))
        entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert entry["market"] == "futures"
