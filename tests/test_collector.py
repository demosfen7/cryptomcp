"""Тесты фонового сборщика.

Главное здесь — направление листания. Раздел /futures/data/ не использует
startTime для позиции: запрос с startTime за 25 суток назад возвращает не
начало окна, а свежий хвост. Проверено на живой бирже; ошибка была ровно
такой и стоила бы двадцати семи суток истории при первом же бэкфилле.
"""

from __future__ import annotations

import pytest

from cryptomcp import storage
from cryptomcp.collector import PERIOD_MS, _series, collect, fetch_window
from cryptomcp.errors import unknown_symbol

NOW = 1_788_000_000_000


class FakeClient:
    """Биржа, отдающая последние ``page`` точек до endTime — как настоящая."""

    def __init__(self, first_ts: int, last_ts: int, page: int = 1000,
                 broken: set[str] | None = None):
        self.first_ts, self.last_ts, self.page = first_ts, last_ts, page
        self.broken = broken or set()
        self.calls: list[tuple[str, int | None]] = []

    def _rows(self, source: str, end_time: int | None):
        if source in self.broken:
            raise unknown_symbol("BROKENUSDT")
        end = min(end_time if end_time is not None else self.last_ts, self.last_ts)
        stamps = []
        ts = end - (end - self.first_ts) % PERIOD_MS
        while ts >= self.first_ts and len(stamps) < self.page:
            stamps.append(ts)
            ts -= PERIOD_MS
        return [{"timestamp": t, "sumOpenInterest": "1.0",
                 "sumOpenInterestValue": "2.0", "longShortRatio": "1.5",
                 "buySellRatio": "0.9"} for t in sorted(stamps)]

    async def open_interest_hist(self, symbol, period="5m", *, limit=500,
                                 start_time=None, end_time=None):
        self.calls.append(("open_interest_hist", end_time))
        return self._rows("open_interest_hist", end_time)

    async def long_short_ratio(self, symbol, kind="global_account", period="5m", *,
                               limit=500, start_time=None, end_time=None):
        self.calls.append((kind, end_time))
        return self._rows(kind, end_time)

    async def taker_long_short_ratio(self, symbol, period="5m", *, limit=500,
                                     start_time=None, end_time=None):
        self.calls.append(("taker", end_time))
        return self._rows("taker", end_time)

    async def now_ms(self):
        return self.last_ts


@pytest.fixture
def con(tmp_path):
    connection = storage.connect(str(tmp_path / "market.sqlite"))
    yield connection
    connection.close()


class TestBackwardPaging:
    @pytest.mark.asyncio
    async def test_single_page_window(self):
        client = FakeClient(NOW - 100 * PERIOD_MS, NOW)
        rows = await _series(client, "open_interest_hist", "BTCUSDT",
                             NOW - 10 * PERIOD_MS, NOW)
        assert len(rows) == 11
        assert min(rows) == NOW - 10 * PERIOD_MS

    @pytest.mark.asyncio
    async def test_window_longer_than_page_is_paged(self):
        """Тридцать суток с шагом 5m — это 8640 точек, девять страниц по 1000."""
        client = FakeClient(NOW - 20_000 * PERIOD_MS, NOW, page=1000)
        rows = await _series(client, "open_interest_hist", "BTCUSDT",
                             NOW - 8639 * PERIOD_MS, NOW)
        assert len(rows) == 8640
        assert len(client.calls) == 9

    @pytest.mark.asyncio
    async def test_cursor_moves_backwards(self):
        client = FakeClient(NOW - 20_000 * PERIOD_MS, NOW, page=1000)
        await _series(client, "open_interest_hist", "BTCUSDT",
                      NOW - 5000 * PERIOD_MS, NOW)
        ends = [end for _, end in client.calls]
        assert ends[0] == NOW
        assert ends == sorted(ends, reverse=True), "курсор обязан идти справа налево"

    @pytest.mark.asyncio
    async def test_stops_at_start_of_history(self):
        """Символ моложе окна: листание упирается в начало и не зацикливается."""
        client = FakeClient(NOW - 50 * PERIOD_MS, NOW, page=1000)
        rows = await _series(client, "open_interest_hist", "BTCUSDT",
                             NOW - 8639 * PERIOD_MS, NOW)
        assert len(rows) == 51
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_nothing_before_start(self):
        client = FakeClient(NOW - 100 * PERIOD_MS, NOW)
        rows = await _series(client, "open_interest_hist", "BTCUSDT",
                             NOW - 10 * PERIOD_MS, NOW)
        assert all(ts >= NOW - 10 * PERIOD_MS for ts in rows)


class TestMerge:
    @pytest.mark.asyncio
    async def test_five_sources_land_in_one_point(self):
        client = FakeClient(NOW - 100 * PERIOD_MS, NOW)
        rows = await fetch_window(client, "BTCUSDT", NOW - 3 * PERIOD_MS, NOW)
        point = rows[NOW]
        assert point == {
            "open_interest": 1.0, "open_interest_value": 2.0,
            "ls_global": 1.5, "ls_top_accounts": 1.5, "ls_top_positions": 1.5,
            "taker_ratio": 0.9,
        }

    @pytest.mark.asyncio
    async def test_one_broken_endpoint_does_not_lose_the_rest(self):
        """Отвалившийся эндпоинт не должен отменять запись остальных."""
        client = FakeClient(NOW - 100 * PERIOD_MS, NOW, broken={"taker"})
        rows = await fetch_window(client, "BTCUSDT", NOW - 3 * PERIOD_MS, NOW)
        assert rows[NOW]["open_interest"] == 1.0
        assert "taker_ratio" not in rows[NOW]


class TestIncremental:
    @pytest.mark.asyncio
    async def test_first_run_takes_whole_available_history(self, con):
        """Пустая база — окно от границы 30 суток, а не от «сейчас»."""
        client = FakeClient(NOW - 20_000 * PERIOD_MS, NOW)
        written, failed = await collect(client, con, ["BTCUSDT"], days=None)
        assert failed == []
        # 30 суток с шагом 5m — 8640 интервалов, то есть 8641 точка с обеими
        # границами. У живой биржи сетка не привязана к «сейчас», и там ровно 8640.
        assert written == 8641

    @pytest.mark.asyncio
    async def test_second_run_asks_only_for_the_tail(self, con):
        client = FakeClient(NOW - 20_000 * PERIOD_MS, NOW)
        await collect(client, con, ["BTCUSDT"], days=None)
        before = len(client.calls)

        client.last_ts = NOW + 10 * PERIOD_MS
        written, _ = await collect(client, con, ["BTCUSDT"], days=None)

        assert written == 10, "должен догрузиться только хвост"
        # Эндпоинтов пять: открытый интерес отдаёт сразу две колонки.
        assert len(client.calls) - before == 5, "по одному запросу на эндпоинт"

    @pytest.mark.asyncio
    async def test_nothing_new_is_not_an_error(self, con):
        client = FakeClient(NOW - 100 * PERIOD_MS, NOW)
        await collect(client, con, ["BTCUSDT"], days=None)
        written, failed = await collect(client, con, ["BTCUSDT"], days=None)
        assert (written, failed) == (0, [])

    @pytest.mark.asyncio
    async def test_failed_symbol_does_not_stop_the_run(self, con):
        class Failing(FakeClient):
            async def open_interest_hist(self, symbol, *a, **kw):
                if symbol == "BADUSDT":
                    raise unknown_symbol(symbol)
                return await super().open_interest_hist(symbol, *a, **kw)

        client = Failing(NOW - 100 * PERIOD_MS, NOW)
        written, failed = await collect(client, con, ["BTCUSDT", "BADUSDT"], days=None)
        assert written > 0
        assert storage.coverage(con, "BTCUSDT")["points"] > 0


class TestHealth:
    """Живость меряется результатом прогона, а не наличием процесса."""

    def test_no_runs_is_not_healthy(self, con):
        from cryptomcp.collector import health

        alive, message = health(con)
        assert not alive
        assert "не было" in message

    def test_fresh_run_is_healthy(self, con):
        from cryptomcp.collector import health

        storage.record_run(con, "incremental", symbols=52, rows=440, seconds=9.0)
        alive, message = health(con)
        assert alive
        assert "440" in message

    def test_stale_run_is_not_healthy(self, con):
        import datetime as dt

        from cryptomcp.collector import health

        storage.record_run(con, "incremental", symbols=52, rows=440, seconds=9.0)
        stale = int((dt.datetime.now(dt.UTC).timestamp() - 5 * 3600) * 1000)
        con.execute("UPDATE collector_runs SET ts_ms = ?", (stale,))
        alive, message = health(con, interval_s=3600)
        assert not alive
        assert "назад" in message

    def test_universe_run_alone_is_not_enough(self, con):
        """Снимок универсума прошёл, а деривативы — нет: это не здоровье."""
        from cryptomcp.collector import health

        storage.record_run(con, "universe", symbols=52, rows=52, seconds=1.0)
        assert not health(con)[0]


class FakeKlineClient:
    """Биржа, отдающая свечи ВПЕРЁД от startTime — как настоящий /klines."""

    def __init__(self, first_ts: int, last_ts: int, step_ms: int, page: int = 1000):
        self.first_ts, self.last_ts, self.step, self.page = first_ts, last_ts, step_ms, page
        self.market = type("M", (), {"max_limit": page})()
        self.starts: list[int] = []

    async def now_ms(self):
        # С запасом: свеча считается закрытой не в момент close_time, а через
        # CLOSE_GRACE_MS после — биржа успевает её доправить.
        return self.last_ts + self.step + 10_000

    async def klines(self, symbol, interval, *, limit=500, start_time=None,
                     end_time=None, cache_ttl_s=0.0):
        self.starts.append(start_time)
        begin = max(start_time or self.first_ts, self.first_ts)
        # Выравнивание на сетку, как у биржи.
        offset = (begin - self.first_ts) % self.step
        if offset:
            begin += self.step - offset
        out = []
        ts = begin
        while ts <= self.last_ts and len(out) < limit:
            out.append([ts, "1.0", "2.0", "0.5", "1.5", "10.0", ts + self.step - 1,
                        "1000.0", 7, "5.0", "500.0", "0"])
            ts += self.step
        return out


class TestCandleLoader:
    @pytest.mark.asyncio
    async def test_forward_paging_covers_the_window(self, con):
        from cryptomcp.collector import load_candles

        step = 3_600_000
        client = FakeKlineClient(NOW - 2500 * step, NOW, step, page=1000)
        written = await load_candles(client, con, "CAKEUSDT", "1h", 200, "spot")

        assert written == 2501
        assert len(client.starts) == 3, "две полные страницы и последняя неполная"
        assert client.starts == sorted(client.starts), "курсор обязан идти вперёд"

    @pytest.mark.asyncio
    async def test_depth_limits_first_load(self, con):
        """Глубина лестницы, а не вся история символа."""
        from cryptomcp.collector import load_candles

        step = 86_400_000
        client = FakeKlineClient(NOW - 3000 * step, NOW, step)
        written = await load_candles(client, con, "CAKEUSDT", "1d", 100, "spot")
        # Точное число зависит от выравнивания на сетку: startTime внутри свечи
        # отдаёт следующую. Важно, что взята глубина лестницы, а не вся история
        # символа — у него её три тысячи суток.
        assert 99 <= written <= 101

    @pytest.mark.asyncio
    async def test_second_run_refetches_the_last_candle(self, con):
        """Курсор ставится НА последнюю сохранённую свечу: биржа её правит."""
        from cryptomcp.collector import load_candles

        step = 3_600_000
        client = FakeKlineClient(NOW - 50 * step, NOW, step)
        await load_candles(client, con, "CAKEUSDT", "1h", 10, "spot")
        saved = storage.last_ohlcv_ts(con, "CAKEUSDT", "1h")

        client.starts.clear()
        client.last_ts = NOW + 3 * step
        written = await load_candles(client, con, "CAKEUSDT", "1h", 10, "spot")

        assert client.starts[0] == saved, "начинать надо с сохранённой, а не после неё"
        assert written == 4, "перезаписанная последняя плюс три новых"

    @pytest.mark.asyncio
    async def test_unclosed_candle_is_not_stored(self, con):
        """Последняя свеча ещё идёт: её объём неполон, в архиве это навсегда."""
        from cryptomcp.collector import load_candles

        step = 3_600_000
        client = FakeKlineClient(NOW - 10 * step, NOW, step)

        async def now_ms():
            return NOW  # последняя свеча ещё не закрыта

        client.now_ms = now_ms
        await load_candles(client, con, "CAKEUSDT", "1h", 10, "spot")
        # Отброшены две: идущая сейчас и закрывшаяся мгновение назад — вторая
        # попадает в пятисекундный запас, за который биржа её ещё правит.
        assert storage.last_ohlcv_ts(con, "CAKEUSDT", "1h") == NOW - 2 * step


class TestArchivePlan:
    @pytest.mark.asyncio
    async def test_spot_preferred_when_available(self, con):
        from cryptomcp.collector import LADDER, archive_plan

        plan = await archive_plan(con, ["CAKEUSDT"], [], {"CAKEUSDT"})
        assert plan == [("CAKEUSDT", "spot", LADDER)]

    @pytest.mark.asyncio
    async def test_futures_when_no_spot_pair(self, con):
        """Четверть ликвидных перпетуалов спота не имеет — HYPE, UAI и другие."""
        from cryptomcp.collector import archive_plan

        plan = await archive_plan(con, ["UAIUSDT"], [], set())
        assert plan[0][1] == "futures"

    @pytest.mark.asyncio
    async def test_existing_source_wins_over_new_spot_listing(self, con):
        """Появился спот позже — источник не меняем, иначе склеим два рынка."""
        from cryptomcp.collector import archive_plan

        storage.upsert_ohlcv(con, "UAIUSDT", "1d", "futures",
                             [(1000, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 1.0, 1.0)])
        plan = await archive_plan(con, ["UAIUSDT"], [], {"UAIUSDT"})
        assert plan[0][1] == "futures"

    @pytest.mark.asyncio
    async def test_mid_layer_gets_shorter_ladder(self, con):
        from cryptomcp.collector import MID_LADDER, archive_plan

        plan = await archive_plan(con, [], ["BTCUSDT"], {"BTCUSDT"})
        assert plan[0][2] == MID_LADDER
        assert [tf for tf, _ in plan[0][2]] == ["1d", "4h"]
