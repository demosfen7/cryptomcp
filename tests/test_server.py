"""Тесты составных инструментов (ТЗ §8, PLAN §5).

Главное здесь — частичный результат: нехватка истории на одном таймфрейме не
должна отменять остальные. Свежие листинги — целевая категория сканера, и
именно по ним снапшот падал целиком.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.errors import ErrorKind, ToolError, insufficient_history
from cryptomcp.series import INTERVAL_MS, build_series
from cryptomcp.server import _views

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
