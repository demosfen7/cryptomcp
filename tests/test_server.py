"""Тесты составных инструментов (ТЗ §8, PLAN §5).

Главное здесь — частичный результат: нехватка истории на одном таймфрейме не
должна отменять остальные. Свежие листинги — целевая категория сканера, и
именно по ним снапшот падал целиком.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cryptomcp.errors import ErrorKind, ToolError, insufficient_history
from cryptomcp.markets import MARKETS
from cryptomcp.series import INTERVAL_MS, build_series
from cryptomcp.server import _merged, _views

H4 = INTERVAL_MS["4h"]


def synthetic_series(interval: str, count: int = 300):
    step = INTERVAL_MS[interval]
    rng = np.random.default_rng(7)
    raw = []
    price = 100.0
    for i in range(count):
        price += float(rng.normal(0, 0.3))
        raw.append([
            i * step, f"{price:.8f}", f"{price + 1:.8f}", f"{price - 1:.8f}",
            f"{price:.8f}", "1.0", (i + 1) * step - 1, "1000.0", 10,
            "0.5", "500.0", "0",
        ])
    return build_series(raw, "TESTUSDT", interval, count * step + 10_000, grace_ms=0)


class FakeFetcher:
    """Фетчер, у которого часть таймфреймов «свежий листинг».

    Отказ выдаётся только при заданном ``min_candles`` — ровно как у настоящего:
    запрос недельных свечей ради пивотов лимита не ставит и проходит.
    """

    def __init__(self, short: dict[str, tuple[int, int]] | None = None) -> None:
        self.short = short or {}
        self.calls: list[str] = []

    async def get(self, symbol, interval, *, limit=500, as_of_ms=None,
                  min_candles=None, target_span_days=None, max_pages=12):
        self.calls.append(interval)
        if min_candles is not None and interval in self.short:
            have, need = self.short[interval]
            raise insufficient_history(symbol, interval, have, need)
        return synthetic_series(interval)


@pytest.fixture(autouse=True)
def no_journal(monkeypatch):
    monkeypatch.setattr("cryptomcp.server.journal.enabled", False)


class TestPartialSnapshot:
    @pytest.mark.asyncio
    async def test_short_timeframe_does_not_kill_the_rest(self):
        fetcher = FakeFetcher(short={"1w": (43, 60)})

        views, skipped = await _views(
            fetcher, "UAIUSDT", ("1w", "1d", "4h", "1h"), None
        )

        assert list(views) == ["1d", "4h", "1h"]
        assert list(skipped) == ["1w"]
        assert skipped["1w"].kind is ErrorKind.INSUFFICIENT_HISTORY
        assert skipped["1w"].details == {
            "symbol": "UAIUSDT", "interval": "1w", "have": 43, "need": 60
        }

    @pytest.mark.asyncio
    async def test_weekly_pivots_survive_short_weekly_ladder(self):
        """Пивоты берутся отдельным запросом без лимита и остаются доступны."""
        fetcher = FakeFetcher(short={"1w": (43, 60)})

        views, _ = await _views(fetcher, "UAIUSDT", ("1w", "4h"), None)

        assert views["4h"].pivots_weekly is not None

    @pytest.mark.asyncio
    async def test_everything_missing_still_raises(self):
        """Когда не посчитан ни один ТФ, возвращать нечего — это ошибка."""
        fetcher = FakeFetcher(short={"4h": (10, 60), "1h": (12, 60)})

        with pytest.raises(ToolError) as caught:
            await _views(fetcher, "UAIUSDT", ("4h", "1h"), None)

        assert caught.value.kind is ErrorKind.INSUFFICIENT_HISTORY

    @pytest.mark.asyncio
    async def test_single_timeframe_tool_keeps_its_error(self):
        """get_squeeze_metrics по недоступному ТФ обязан ответить ошибкой."""
        fetcher = FakeFetcher(short={"1w": (43, 60)})

        with pytest.raises(ToolError):
            await _views(fetcher, "UAIUSDT", ("1w",), None)

    @pytest.mark.asyncio
    async def test_nothing_skipped_on_a_full_symbol(self):
        fetcher = FakeFetcher()

        views, skipped = await _views(fetcher, "CAKEUSDT", ("1d", "4h"), None)

        assert list(views) == ["1d", "4h"]
        assert skipped == {}


class TestRawKlineCap:
    """Потолок сырых свечей зависит от таймфрейма, а не задан одним числом."""

    def test_intraday_allows_two_hundred(self):
        """50 часовых свечей — двое суток, меньше, чем длится фаза поглощения."""
        from cryptomcp.server import max_raw_klines

        assert max_raw_klines("1h") == 200
        assert max_raw_klines("15m") == 200
        assert max_raw_klines("5m") == 200

    def test_higher_timeframes_keep_fifty(self):
        """50 дневных — два месяца; читать по ним форму бессмысленно."""
        from cryptomcp.server import max_raw_klines

        assert max_raw_klines("4h") == 50
        assert max_raw_klines("1d") == 50
        assert max_raw_klines("1w") == 50


class TestScanTimeframeAlias:
    """scan_pairs обязан понимать и timeframe, и timeframes.

    Единственный инструмент со списком: у остальных параметр в единственном
    числе. Лишний ключ MCP-сервер отбрасывает молча, поэтому написание
    timeframe="1d" уводило выдачу на дефолтные 4h без единого признака ошибки.
    """

    @staticmethod
    def _capture(monkeypatch):
        seen: list[tuple[str, ...]] = []

        async def fake_screen(intervals, **kwargs):
            seen.append(intervals)
            return ""

        monkeypatch.setattr("cryptomcp.server._scan_screen", fake_screen)
        return seen

    @pytest.mark.asyncio
    async def test_singular_reaches_the_screen(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(timeframe="1d")

        assert seen == [("1d",)]

    @pytest.mark.asyncio
    async def test_plural_still_works(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(timeframes=["1d", "1h"])

        assert seen == [("1d", "1h")]

    @pytest.mark.asyncio
    async def test_default_stays_four_hours(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs()

        assert seen == [("4h",)]

    @pytest.mark.asyncio
    async def test_both_spellings_merge_without_duplicates(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(timeframes=["4h"], timeframe="1d")

        assert seen == [("4h", "1d")]

    @pytest.mark.asyncio
    async def test_singular_is_published_in_the_schema(self):
        """Параметра нет в схеме — клиент отбросит ключ, не дойдя до функции."""
        from cryptomcp.server import server

        tools = {tool.name: tool for tool in await server.list_tools()}
        properties = tools["scan_pairs"].input_schema["properties"]

        assert "timeframe" in properties
        assert "timeframes" in properties


class TestScanSymbolAlias:
    """scan_pairs обязан понимать и symbol, и symbols.

    Потеря этого ключа дороже потерянного таймфрейма: она не меняла ТФ, а
    молча переключала режим — вместо пересчёта по бирже по названной монете
    уходил полный отбор по журналу.
    """

    @staticmethod
    def _capture(monkeypatch):
        seen: dict[str, object] = {}

        async def fake_explicit(symbols, intervals, market):
            seen["explicit"] = (symbols, intervals)
            return ""

        async def fake_screen(intervals, **kwargs):
            seen["screen"] = intervals
            return ""

        monkeypatch.setattr("cryptomcp.server._scan_explicit", fake_explicit)
        monkeypatch.setattr("cryptomcp.server._scan_screen", fake_screen)
        return seen

    @pytest.mark.asyncio
    async def test_singular_goes_to_the_exchange_path(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(symbol="BTCUSDT")

        assert seen["explicit"] == (["BTCUSDT"], ("4h",))
        assert "screen" not in seen

    @pytest.mark.asyncio
    async def test_plural_still_works(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(symbols=["BTCUSDT", "ETHUSDT"])

        assert seen["explicit"] == (["BTCUSDT", "ETHUSDT"], ("4h",))

    @pytest.mark.asyncio
    async def test_both_spellings_merge_without_duplicates(self, monkeypatch):
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs(symbols=["BTCUSDT"], symbol="ETHUSDT")

        assert seen["explicit"] == (["BTCUSDT", "ETHUSDT"], ("4h",))

        seen.clear()
        await scan_pairs(symbols=["BTCUSDT"], symbol="BTCUSDT")

        assert seen["explicit"] == (["BTCUSDT"], ("4h",))

    @pytest.mark.asyncio
    async def test_without_pairs_the_screen_still_runs(self, monkeypatch):
        """Отбор по журналу — режим по умолчанию, алиас его не отменяет."""
        from cryptomcp.server import scan_pairs

        seen = self._capture(monkeypatch)
        await scan_pairs()

        assert seen["screen"] == ("4h",)
        assert "explicit" not in seen

    @pytest.mark.asyncio
    async def test_singular_is_published_in_the_schema(self):
        """Параметра нет в схеме — клиент отбросит ключ, не дойдя до функции."""
        from cryptomcp.server import server

        tools = {tool.name: tool for tool in await server.list_tools()}
        properties = tools["scan_pairs"].input_schema["properties"]

        assert "symbol" in properties
        assert "symbols" in properties


class TestParameterNamesAcrossTools:
    """Расхождение имён между инструментами — тихая потеря ключа.

    MCP-сервер собирает модель аргументов без extra="forbid", поэтому
    незнакомый ключ не вызывает ошибки: он просто исчезает, и инструмент
    отвечает по умолчанию — правдоподобно выглядящей, но не тем ответом.
    Значит имена одной и той же величины обязаны совпадать по всему набору.
    """

    @staticmethod
    async def _schemas():
        from cryptomcp.server import server

        return {
            tool.name: set(tool.input_schema.get("properties", {}))
            for tool in await server.list_tools()
        }

    @pytest.mark.asyncio
    async def test_plural_parameter_always_has_a_singular_twin(self):
        """Проверка односторонняя: вызывающий пишет единственное число.

        Обратное неверно: get_derivatives работает по одной паре, и никакой
        symbols ему не нужен.
        """
        schemas = await self._schemas()
        singulars = {name for names in schemas.values() for name in names}

        missing = {
            tool: sorted(
                plural for plural in names
                if plural.endswith("s")
                and plural[:-1] in singulars
                and plural[:-1] not in names
            )
            for tool, names in schemas.items()
        }
        missing = {tool: gaps for tool, gaps in missing.items() if gaps}

        assert missing == {}

    @pytest.mark.asyncio
    async def test_volume_threshold_spelled_both_ways(self):
        """Один порог оборота, два исторических имени в разных инструментах."""
        schemas = await self._schemas()

        assert {"min_volume_usdt", "min_quote_volume_usdt"} <= schemas["list_symbols"]
        assert "min_volume_usdt" in schemas["scan_pairs"]


class TestSnapshotTimeframeAlias:
    """get_market_snapshot — второй инструмент со списком таймфреймов.

    Инструмент добирается до таймфреймов только после похода на биржу за
    символом, поэтому окружение подменяется целиком: тесты этого проекта в
    сеть не ходят.
    """

    @staticmethod
    def _capture(monkeypatch):
        seen: list[tuple[str, ...]] = []

        class FakeRegistry:
            async def get(self, symbol):
                return SimpleNamespace(symbol=symbol.upper())

        async def fake_ctx(market):
            return None, None, FakeRegistry(), None, MARKETS[market]

        async def fake_views(fetcher, symbol, intervals, as_of_ms, market=None):
            seen.append(intervals)
            # Останавливаем инструмент сразу после разбора таймфреймов: всё
            # дальше — рендер, к имени параметра он отношения не имеет.
            raise insufficient_history(symbol, intervals[0], 0, 1)

        monkeypatch.setattr("cryptomcp.server._ctx", fake_ctx)
        monkeypatch.setattr("cryptomcp.server._views", fake_views)
        return seen

    @pytest.mark.asyncio
    async def test_singular_narrows_the_ladder(self, monkeypatch):
        from cryptomcp.server import get_market_snapshot

        seen = self._capture(monkeypatch)
        await get_market_snapshot("BTCUSDT", timeframe="1d")

        assert seen == [("1d",)]

    @pytest.mark.asyncio
    async def test_plural_still_works(self, monkeypatch):
        from cryptomcp.server import get_market_snapshot

        seen = self._capture(monkeypatch)
        await get_market_snapshot("BTCUSDT", timeframes=["1d", "4h"])

        assert seen == [("1d", "4h")]

    @pytest.mark.asyncio
    async def test_default_ladder_survives(self, monkeypatch):
        """Без параметров остаётся лестница из конфига, а не пустой список."""
        from cryptomcp.server import config, get_market_snapshot

        seen = self._capture(monkeypatch)
        await get_market_snapshot("BTCUSDT")

        assert seen == [config.timeframes]


class TestMergedSpellings:
    """Слияние списка с одиночным значением — общий помощник трёх мест."""

    def test_singular_alone(self):
        assert _merged(None, "1d") == ["1d"]

    def test_plural_alone(self):
        assert _merged(["1d", "4h"], None) == ["1d", "4h"]

    def test_order_is_kept_and_duplicates_dropped(self):
        assert _merged(["4h"], "1d") == ["4h", "1d"]
        assert _merged(["4h"], "4h") == ["4h"]

    def test_nothing_given(self):
        """Пустой список — сигнал «бери значение по умолчанию», не ошибка."""
        assert _merged(None, None) == []


class TestListingAges:
    """Возраст листинга: два источника, и порядок между ними не случаен."""

    def info(self, symbol, onboard_ms=0):
        return SimpleNamespace(symbol=symbol, onboard_ms=onboard_ms)

    def test_onboard_date_is_preferred(self):
        from cryptomcp.server import _listing_ages

        now = 1_788_500_000_000
        ages = _listing_ages(
            [self.info("BTCUSDT", now - 60 * 86_400_000)], now, "futures"
        )
        assert ages == {"BTCUSDT": 60}

    def test_without_onboard_and_archive_the_age_is_absent(self, monkeypatch):
        """Прочерк, а не ноль: возраст неизвестен, а не равен нулю.

        Так у спота, где биржа даты листинга не отдаёт вовсе.
        """
        from cryptomcp import server

        monkeypatch.setattr(server, "archive_path", lambda: None)
        assert server._listing_ages([self.info("BTCUSDT")], 1_788_500_000_000,
                                    "spot") == {}


class TestOtherMarketVolumes:
    """Соседний рынок не обязан отвечать."""

    @pytest.mark.asyncio
    async def test_failure_leaves_the_column_empty(self, monkeypatch):
        from cryptomcp import server

        async def broken(_market):
            raise ToolError(ErrorKind.UPSTREAM_ERROR, "спот недоступен")

        monkeypatch.setattr(server, "_ctx", broken)
        volumes, label = await server._other_market_volumes("futures")

        assert volumes == {}
        assert label == "спот"

    @pytest.mark.asyncio
    async def test_label_follows_the_market(self, monkeypatch):
        from cryptomcp import server

        async def broken(_market):
            raise ToolError(ErrorKind.UPSTREAM_ERROR, "фьючерсы недоступны")

        monkeypatch.setattr(server, "_ctx", broken)
        _, label = await server._other_market_volumes("spot")
        assert label == "фьюч"
