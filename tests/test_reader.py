"""Тесты чтения рядов из архива (PLAN §4.15).

Главное свойство — взаимозаменяемость источников: один и тот же запрос обязан
давать один и тот же ряд, пришёл он из базы или из биржи. Иначе повторяется
история с объёмной базой, где одна свеча получала разные числа в разных
инструментах.
"""

from __future__ import annotations

import pytest

from cryptomcp import storage
from cryptomcp.analysis import required_candles
from cryptomcp.reader import WARMUP, ArchiveReader
from cryptomcp.series import build_series

NOW = 1_788_000_000_000
HOUR = 3_600_000


def candles(count: int, step: int, last_open: int):
    """Свечи по возрастанию времени, заканчивая ``last_open``."""
    return [
        (last_open - (count - 1 - i) * step, 100.0 + i, 101.0 + i, 99.0 + i,
         100.5 + i, 10.0, 1000.0, 5, 5.0, 500.0)
        for i in range(count)
    ]


class FakeClient:
    def __init__(self, step: int, last_open: int):
        self.step, self.last_open = step, last_open
        self.calls = 0

    async def now_ms(self):
        return self.last_open + self.step + 10_000

    async def klines(self, symbol, interval, *, limit=500, start_time=None,
                     end_time=None, cache_ttl_s=0.0):
        self.calls += 1
        begin = start_time if start_time is not None else self.last_open
        rows, ts = [], begin
        while ts <= self.last_open and len(rows) < limit:
            rows.append([ts, "1.0", "2.0", "0.5", "1.5", "10.0", ts + self.step - 1,
                         "1000.0", 5, "5.0", "500.0", "0"])
            ts += self.step
        return rows


class FakeFetcher:
    def __init__(self, client: FakeClient, count: int = 5000):
        self._client = client
        self.count = count
        self.calls = 0

    @property
    def client(self):
        return self._client

    async def get(self, symbol, interval, *, limit=500, as_of_ms=None,
                  min_candles=None, target_span_days=None, max_pages=12):
        self.calls += 1
        step = self._client.step
        raw = [
            [ts, "1.0", "2.0", "0.5", "1.5", "10.0", ts + step - 1,
             "1000.0", 5, "5.0", "500.0", "0"]
            for ts, *_ in candles(min(limit, self.count), step, self._client.last_open)
        ]
        return build_series(raw, symbol, interval, await self._client.now_ms())


@pytest.fixture
def con(tmp_path):
    connection = storage.connect(str(tmp_path / "market.sqlite"))
    yield connection
    connection.close()


@pytest.fixture
def archive(tmp_path, con):
    def fill(symbol="CAKEUSDT", tf="4h", source="spot", count=5000,
             step=4 * HOUR, last_open=NOW):
        storage.upsert_ohlcv(con, symbol, tf, source,
                             candles(count, step, last_open))
        con.commit()
        return str(tmp_path / "market.sqlite")
    return fill


class TestCanonicalWindow:
    """Окно определяется требованиями §4.2, а не размером страницы ответа."""

    def test_percentile_window_dominates_on_high_timeframes(self):
        assert required_candles("1d") == 361
        assert required_candles("1w") == 361

    def test_span_dominates_on_low_timeframes(self):
        assert required_candles("1h") == 1441  # 60 суток по часу
        assert required_candles("15m") == 5761

    def test_four_hours_is_the_crossover(self):
        assert required_candles("4h") == 361


class TestSourceIsolation:
    @pytest.mark.asyncio
    async def test_other_market_goes_to_exchange(self, archive):
        """Спотовый запрос по монете из фьючерсного архива не должен склеиваться."""
        path = archive(source="futures")
        client = FakeClient(4 * HOUR, NOW)
        fetcher = FakeFetcher(client)
        reader = ArchiveReader(fetcher, "spot", path)

        await reader.get("CAKEUSDT", "4h")

        assert fetcher.calls == 1, "ряд обязан прийти с биржи"

    @pytest.mark.asyncio
    async def test_matching_market_uses_archive(self, archive):
        path = archive(source="spot")
        fetcher = FakeFetcher(FakeClient(4 * HOUR, NOW))
        reader = ArchiveReader(fetcher, "spot", path)

        await reader.get("CAKEUSDT", "4h")

        assert fetcher.calls == 0, "история должна прийти из базы"

    @pytest.mark.asyncio
    async def test_unknown_symbol_goes_to_exchange(self, archive):
        path = archive()
        fetcher = FakeFetcher(FakeClient(4 * HOUR, NOW))
        reader = ArchiveReader(fetcher, "spot", path)

        await reader.get("НЕТУUSDT", "4h")

        assert fetcher.calls == 1

    @pytest.mark.asyncio
    async def test_short_archive_goes_to_exchange(self, archive):
        """Неполное окно — это уже другой ряд, брать его вместо биржевого нельзя."""
        path = archive(count=100)
        fetcher = FakeFetcher(FakeClient(4 * HOUR, NOW))
        reader = ArchiveReader(fetcher, "spot", path)

        await reader.get("CAKEUSDT", "4h")

        assert fetcher.calls == 1

    @pytest.mark.asyncio
    async def test_without_archive_file_everything_goes_to_exchange(self):
        fetcher = FakeFetcher(FakeClient(4 * HOUR, NOW))
        reader = ArchiveReader(fetcher, "spot", None)

        await reader.get("CAKEUSDT", "4h")

        assert fetcher.calls == 1


class TestEquivalence:
    @pytest.mark.asyncio
    async def test_both_sources_give_the_same_length(self, archive):
        path = archive()
        window = required_candles("4h") + WARMUP

        from_archive = await ArchiveReader(
            FakeFetcher(FakeClient(4 * HOUR, NOW)), "spot", path
        ).get("CAKEUSDT", "4h")
        from_exchange = await ArchiveReader(
            FakeFetcher(FakeClient(4 * HOUR, NOW)), "spot", None
        ).get("CAKEUSDT", "4h")

        assert len(from_archive) == len(from_exchange) == window

    @pytest.mark.asyncio
    async def test_requested_limit_does_not_change_the_window(self, archive):
        """Иначе get_klines с разным limit снова считал бы разную базу объёма."""
        path = archive()
        lengths = set()
        for limit in (30, 50, 500):
            series = await ArchiveReader(
                FakeFetcher(FakeClient(4 * HOUR, NOW)), "spot", path
            ).get("CAKEUSDT", "4h", limit=limit)
            lengths.add(len(series))
        assert len(lengths) == 1


class TestTail:
    @pytest.mark.asyncio
    async def test_tail_is_requested_from_the_exchange(self, archive):
        """Сборщик ходит раз в час — без хвоста выдача отставала бы на час."""
        path = archive(last_open=NOW - 8 * HOUR)
        client = FakeClient(4 * HOUR, NOW)
        reader = ArchiveReader(FakeFetcher(client), "spot", path)

        series = await reader.get("CAKEUSDT", "4h")

        assert client.calls == 1
        assert int(series.df["open_time"].iloc[-1]) == NOW

    @pytest.mark.asyncio
    async def test_missing_tail_is_not_fatal(self, archive):
        """Архив без хвоста лучше отказа: отставание видно по штампу закрытия."""
        path = archive(last_open=NOW - 8 * HOUR)

        class Broken(FakeClient):
            async def klines(self, *a, **kw):
                raise RuntimeError("биржа недоступна")

        reader = ArchiveReader(FakeFetcher(Broken(4 * HOUR, NOW)), "spot", path)
        series = await reader.get("CAKEUSDT", "4h")

        assert len(series) > 0
        assert int(series.df["open_time"].iloc[-1]) == NOW - 8 * HOUR


class TestRetrospective:
    @pytest.mark.asyncio
    async def test_as_of_does_not_touch_the_exchange(self, archive):
        """Ради этого архив и заводился: массовый прогон по истории для калибровки."""
        path = archive()
        client = FakeClient(4 * HOUR, NOW)
        fetcher = FakeFetcher(client)
        reader = ArchiveReader(fetcher, "spot", path)

        await reader.get("CAKEUSDT", "4h", as_of_ms=NOW - 500 * 4 * HOUR)

        assert client.calls == 0
        assert fetcher.calls == 0

    @pytest.mark.asyncio
    async def test_as_of_cuts_the_series(self, archive):
        path = archive()
        cutoff = NOW - 500 * 4 * HOUR
        reader = ArchiveReader(FakeFetcher(FakeClient(4 * HOUR, NOW)), "spot", path)

        series = await reader.get("CAKEUSDT", "4h", as_of_ms=cutoff)

        assert int(series.df["open_time"].iloc[-1]) <= cutoff


class TestInsufficientHistory:
    @pytest.mark.asyncio
    async def test_min_candles_still_raises(self, archive):
        from cryptomcp.errors import ErrorKind, ToolError

        path = archive(count=400)
        fetcher = FakeFetcher(FakeClient(4 * HOUR, NOW), count=10)
        reader = ArchiveReader(fetcher, "spot", path)

        with pytest.raises(ToolError) as caught:
            await reader.get("CAKEUSDT", "4h", min_candles=60)

        assert caught.value.kind is ErrorKind.INSUFFICIENT_HISTORY
