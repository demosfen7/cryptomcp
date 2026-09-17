"""Тесты фонового сборщика.

Главное здесь — направление листания. Раздел /futures/data/ не использует
startTime для позиции: запрос с startTime за 25 суток назад возвращает не
начало окна, а свежий хвост. Проверено на живой бирже; ошибка была ровно
такой и стоила бы двадцати семи суток истории при первом же бэкфилле.
"""

from __future__ import annotations

import pytest

from cryptomcp import SQUEEZE_FORMULA_VERSION, storage
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
        """Мид-слой мельче ядра, но 1h ему нужен: по нему идёт скан."""
        from cryptomcp.collector import LADDER, MID_LADDER, SCAN_TIMEFRAMES, archive_plan

        plan = await archive_plan(con, [], ["BTCUSDT"], {"BTCUSDT"})
        assert plan[0][2] == MID_LADDER

        mid = {tf for tf, _ in MID_LADDER}
        core = {tf for tf, _ in LADDER}
        assert mid < core, "мид-слой обязан быть подмножеством ядра"
        assert "1w" not in mid, "недельные свечи мид-слою не нужны"
        # Иначе скан по 1h видел бы только ядро — половину универсума,
        # отобранную по обороту, то есть ровно не ту.
        assert set(SCAN_TIMEFRAMES) <= mid


class TestScanAndOutcomes:
    """Скан считается из архива и не ходит в биржу вовсе."""

    def fill(self, con, symbol, tf, count, step_ms, start_ts=1_600_000_000_000):
        import random

        random.seed(3)
        price = 100.0
        rows = []
        for i in range(count):
            price *= 1 + random.gauss(0, 0.004)
            rows.append((start_ts + i * step_ms, price, price * 1.01, price * 0.99,
                         price, 10.0, 1000.0, 5, 5.0, 500.0))
        storage.upsert_ohlcv(con, symbol, tf, "spot", rows)
        con.commit()
        return rows

    def test_scan_writes_one_row_per_symbol(self, con):
        from cryptomcp.collector import run_scan

        self.fill(con, "CAKEUSDT", "4h", 400, 4 * 3_600_000)
        assert run_scan(con) == 1
        row = con.execute("SELECT * FROM scan_log").fetchone()
        assert row["symbol"] == "CAKEUSDT"
        assert row["tf"] == "4h"
        assert row["squeeze_index"] is not None

    def test_repeat_scan_adds_nothing(self, con):
        from cryptomcp.collector import run_scan

        self.fill(con, "CAKEUSDT", "4h", 400, 4 * 3_600_000)
        run_scan(con)
        assert run_scan(con) == 0

    def test_short_history_is_skipped(self, con):
        """Меньше шестидесяти свечей — считать нечего, и это не ошибка."""
        from cryptomcp.collector import run_scan

        self.fill(con, "НОВАЯUSDT", "4h", 30, 4 * 3_600_000)
        assert run_scan(con) == 0

    def test_outcome_measured_by_high_and_low(self, con):
        from cryptomcp.collector import settle_outcomes

        step = 3_600_000
        start = 1_600_000_000_000
        candles = [
            (start, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1, 1.0, 1.0),
            (start + step, 100.0, 130.0, 95.0, 101.0, 1.0, 1.0, 1, 1.0, 1.0),
            (start + 2 * step, 101.0, 102.0, 70.0, 99.0, 1.0, 1.0, 1, 1.0, 1.0),
        ]
        # Дальше ровный хвост: горизонт должен быть покрыт архивом целиком,
        # иначе запись справедливо откладывается до следующего прогона.
        candles += [
            (start + i * step, 100.0, 101.0, 99.0, 100.0, 1.0, 1.0, 1, 1.0, 1.0)
            for i in range(3, 30)
        ]
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", candles)
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "price, closed_through_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (start, "CAKEUSDT", "spot", "4h", SQUEEZE_FORMULA_VERSION,
             100.0, start),
        )
        con.commit()

        settled = settle_outcomes(con, now_ms=start + 100 * 86_400_000)

        assert settled >= 1
        row = con.execute(
            "SELECT * FROM outcomes WHERE horizon = '24h'"
        ).fetchone()
        assert row["max_price"] == 130.0
        assert row["min_price"] == 70.0
        assert row["max_pct"] == pytest.approx(30.0)
        assert row["min_pct"] == pytest.approx(-30.0)

    def test_horizon_not_covered_by_archive_is_postponed(self, con):
        """Половина окна дала бы заниженный размах — лучше подождать."""
        from cryptomcp.collector import settle_outcomes

        start = 1_600_000_000_000
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", [
            (start + 3_600_000, 100.0, 101.0, 99.0, 100.0, 1.0, 1.0, 1, 1.0, 1.0),
        ])
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "price, closed_through_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (start, "CAKEUSDT", "spot", "4h", SQUEEZE_FORMULA_VERSION,
             100.0, start),
        )
        con.commit()

        assert settle_outcomes(con, now_ms=start + 100 * 86_400_000) == 0
        assert con.execute("SELECT COUNT(*) c FROM outcomes").fetchone()["c"] == 0

    def test_settling_is_idempotent(self, con):
        from cryptomcp.collector import settle_outcomes

        step = 3_600_000
        start = 1_600_000_000_000
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", [
            (start + i * step, 100.0, 101.0, 99.0, 100.0, 1.0, 1.0, 1, 1.0, 1.0)
            for i in range(200)
        ])
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "price, closed_through_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (start, "CAKEUSDT", "spot", "4h", SQUEEZE_FORMULA_VERSION,
             100.0, start),
        )
        con.commit()
        now = start + 100 * 86_400_000

        first = settle_outcomes(con, now_ms=now)
        second = settle_outcomes(con, now_ms=now)

        assert first > 0
        assert second == 0


class TestScanTwin:
    """Соседний рынок: справка по верхушке списка, а не второй скан всего.

    Замер, ради которого проход заведён: HOMEUSDT 4h, свеча закрытия 03.09
    20:00 UTC — `узк 17` по споту и `36` по перпу, вдвое. Пока в журнале
    стояла одна величина, разница была невидима.
    """

    STEP = 4 * 3_600_000

    def scan(self, con, symbol, index, source="spot"):
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "squeeze_index, closed_through_ms) VALUES (?, ?, ?, '4h', ?, ?, ?)",
            (NOW, symbol, source, SQUEEZE_FORMULA_VERSION, index, NOW - 1),
        )
        con.commit()

    def twins(self, con):
        return {
            row["symbol"]: dict(row)
            for row in con.execute(
                "SELECT symbol, twin_market, twin_narrow_bars, twin_index "
                "FROM scan_log"
            )
        }

    @pytest.mark.asyncio
    async def test_only_the_top_is_measured(self, con):
        """Пятнадцать символов на ТФ, а не сто двадцать: запросы не бесплатны."""
        from cryptomcp.collector import scan_twin

        for i, index in enumerate((0.9, 0.8, 0.7)):
            self.scan(con, f"C{i}USDT", index)
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        written = await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=2,
            now_ms=NOW,
        )

        assert written == 2
        measured = {s: r for s, r in self.twins(con).items() if r["twin_market"]}
        assert set(measured) == {"C0USDT", "C1USDT"}, "мерился только верх списка"

    @pytest.mark.asyncio
    async def test_twin_volume_and_flow_are_recorded(self, con):
        """Объёмные признаки соседнего рынка — то, ради чего проход расширен.

        У 152 архивных монет из 183 спот даёт меньше 30% оборота фьючерсов
        (замер 17.09.2026), то есть объём и поток основного ряда считаются по
        меньшинству торговли.
        """
        from cryptomcp.collector import scan_twin

        self.scan(con, "HOMEUSDT", 0.9, source="spot")
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), now_ms=NOW,
        )

        row = con.execute(
            "SELECT twin_ma_ratio, twin_vol_ratio_12_30, twin_delta_quadrant, "
            "twin_delta_share_30, twin_flow_change_30 FROM scan_log"
        ).fetchone()
        assert row["twin_ma_ratio"] is not None
        assert row["twin_vol_ratio_12_30"] is not None
        assert row["twin_delta_quadrant"] is not None

    @pytest.mark.asyncio
    async def test_accumulation_entries_are_measured_too(self, con):
        """Монеты второго списка тоже получают соседний рынок (§4.42, §4.43).

        В верхушку индекса они обычно не попадают, а объёмные признаки нужны
        им больше всех: отбор идёт по набору, а набор считается по объёму.
        """
        from cryptomcp.collector import scan_twin

        self.scan(con, "TOPUSDT", 0.9)
        self.scan(con, "QUIETUSDT", 0.1)
        con.execute(
            "INSERT INTO accumulation (symbol, tf, market, status, entered_at) "
            "VALUES ('QUIETUSDT', '4h', 'spot', 'active', ?)", (NOW,),
        )
        con.commit()
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=1,
            now_ms=NOW,
        )

        measured = {s: r for s, r in self.twins(con).items() if r["twin_market"]}
        assert set(measured) == {"TOPUSDT", "QUIETUSDT"}

    @pytest.mark.asyncio
    async def test_twin_of_spot_is_the_perpetual(self, con):
        from cryptomcp.collector import scan_twin

        self.scan(con, "HOMEUSDT", 0.9, source="spot")
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=5,
            now_ms=NOW,
        )

        assert self.twins(con)["HOMEUSDT"]["twin_market"] == "futures"

    @pytest.mark.asyncio
    async def test_twin_of_futures_is_spot(self, con):
        from cryptomcp.collector import scan_twin

        self.scan(con, "UBUSDT", 0.9, source="futures")
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=5,
            now_ms=NOW,
        )

        assert self.twins(con)["UBUSDT"]["twin_market"] == "spot"

    @pytest.mark.asyncio
    async def test_missing_pair_is_not_a_failure(self, con):
        """Четверть перпов не имеет спотовой пары — это ответ биржи, не сбой."""
        from cryptomcp.collector import scan_twin

        self.scan(con, "НЕТУUSDT", 0.9)

        class NoPair(FakeKlineClient):
            async def klines(self, *args, **kwargs):
                raise unknown_symbol("НЕТУUSDT")

        client = NoPair(NOW - 3000 * self.STEP, NOW, self.STEP, page=1500)

        written = await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=5
        )

        assert written == 0
        assert self.twins(con)["НЕТУUSDT"]["twin_market"] is None

    @pytest.mark.asyncio
    async def test_short_page_is_skipped_rather_than_shortened(self, con):
        """Урезанное окно дало бы перцентиль, не сравнимый с основным."""
        from cryptomcp.collector import scan_twin

        self.scan(con, "C0USDT", 0.9)
        client = FakeKlineClient(NOW - 3000 * self.STEP, NOW, self.STEP, page=100)

        written = await scan_twin(
            {"spot": client, "futures": client}, con, timeframes=("4h",), limit=5
        )

        assert written == 0


class TestWatchlist:
    """Отбор рангом с гистерезисом плюс трое ворот на вход (§4.35).

    Набор ведётся только на дневке, поэтому и заготовка рынка — дневная.
    Свечей сжатия у заготовки десять, то есть десять суток: выше пола в семь.
    """

    NOW = 1_788_000_000_000
    STEP = 86_400_000
    TF = "1d"

    def scan(self, con, symbol, index, ts, price=100.0, low=95.0, high=105.0,
             tf=None, narrow_bars=10, change=1.0):
        con.execute(
            "INSERT OR REPLACE INTO scan_log (ts_ms, symbol, source, tf, "
            "formula_version, squeeze_index, price, range_low, range_high, "
            "closed_through_ms, narrow_bars, change_24h_pct) "
            "VALUES (?, ?, 'spot', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, tf or self.TF, SQUEEZE_FORMULA_VERSION, index, price,
             low, high, ts, narrow_bars, change),
        )

    def market(self, con, ts, count=50, overrides=None, price=100.0, **fields):
        overrides = overrides or {}
        for i in range(count):
            symbol = f"C{i:02d}USDT"
            self.scan(con, symbol, overrides.get(symbol, 0.60 - i * 0.01), ts,
                      price=price if symbol not in overrides else overrides.get(
                          f"{symbol}_price", price), **fields)
        con.commit()

    def test_short_squeeze_does_not_open_an_episode(self, con):
        """Сканер ищет недельные базы: полтора дня к задаче не относятся.

        В списке от 03.09 было 32 эпизода из 40 без единой свечи сжатия —
        длительность была колонкой, а не воротами.
        """
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, narrow_bars=6)
        changes = update_watchlist(con, self.NOW)

        assert changes["entered"] == []
        assert storage.open_episodes(con) == []

    def test_seven_days_is_enough(self, con):
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, narrow_bars=7)

        assert update_watchlist(con, self.NOW)["entered"] != []

    def test_coin_in_motion_is_rejected(self, con):
        """Шаг 1 протокола: монета в движении не кандидат на накопление."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, change=22.0)
        changes = update_watchlist(con, self.NOW)

        assert changes["entered"] == []

    def test_motion_is_symmetric(self, con):
        """Обвал — такое же движение, как рост."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, change=-22.0)

        assert update_watchlist(con, self.NOW)["entered"] == []

    def test_unknown_change_does_not_block_entry(self, con):
        """У записей прошлых версий поля нет; отказ по пустому полю — не отказ."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, change=None)

        assert update_watchlist(con, self.NOW)["entered"] != []

    def test_index_floor_is_off_by_default(self, con):
        """Порог индекса заведён, но значение выбирается замером, не сегодня."""
        from cryptomcp.collector import WATCH_MIN_INDEX

        assert WATCH_MIN_INDEX == 0.0

    def test_index_floor_filters_when_set(self, con, monkeypatch):
        from cryptomcp import collector

        monkeypatch.setattr(collector, "WATCH_MIN_INDEX", 0.55)
        self.market(con, self.NOW)
        changes = collector.update_watchlist(con, self.NOW)

        assert changes["entered"] != []
        assert all(
            con.execute(
                "SELECT squeeze_index FROM watchlist WHERE symbol = ?",
                (entry["symbol"],),
            ).fetchone()["squeeze_index"] >= 0.55
            for entry in changes["entered"]
        )

    def test_four_hours_no_longer_opens_episodes(self, con):
        """Скан по 4h идёт и пишется, но эпизодов там больше не открывает."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW, tf="4h", narrow_bars=60)
        changes = update_watchlist(con, self.NOW)

        assert changes["entered"] == []
        assert con.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"] == 50

    def test_open_four_hour_episode_still_lives_out_its_life(self, con):
        """Уже открытые эпизоды 4h обязаны дожить, а не зависнуть навсегда."""
        from cryptomcp.collector import update_watchlist

        storage.open_episode(
            con, "C00USDT", "4h", entered_at=self.NOW, entered_by="scanner",
            scan={"squeeze_index": 0.6, "price": 100.0,
                  "range_low": 95.0, "range_high": 105.0, "source": "spot"},
            rank=1,
        )
        con.commit()
        self.market(con, self.NOW + self.STEP, tf="4h", price=200.0)
        changes = update_watchlist(con, self.NOW + self.STEP)

        assert [e["reason"] for e in changes["exited"]] == ["пробой"]

    def test_market_of_the_episode_is_remembered(self, con):
        """Эпизод описывает тот ряд, по которому отобран, — спотовый или перп.

        Без этого поля список выглядел фьючерсным, не будучи им: архив
        предпочитает спот, и на HOMEUSDT 4h ряды разошлись вдвое — узк 17
        против 36 на одной и той же свече.
        """
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        changes = update_watchlist(con, self.NOW)

        assert {row["market"] for row in storage.open_episodes(con)} == {"spot"}
        assert {entry["market"] for entry in changes["entered"]} == {"spot"}

    def test_top_by_rank_enters_as_candidate(self, con):
        from cryptomcp.collector import WATCH_ENTER_RANK, update_watchlist

        self.market(con, self.NOW)
        changes = update_watchlist(con, self.NOW)

        assert len(changes["entered"]) == WATCH_ENTER_RANK
        assert {e["status"] for e in storage.open_episodes(con)} == {"candidate"}

    def test_second_scan_promotes_to_active(self, con):
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP)
        changes = update_watchlist(con, self.NOW + self.STEP)

        assert len(changes["promoted"]) == 15
        assert len(changes["entered"]) == 0
        assert {e["status"] for e in storage.open_episodes(con)} == {"active"}

    def test_hysteresis_keeps_a_slipping_coin(self, con):
        """Выпала из топ-15, но держится выше 40-го — остаётся."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP, overrides={"C00USDT": 0.40})
        update_watchlist(con, self.NOW + self.STEP)

        assert "C00USDT" in {e["symbol"] for e in storage.open_episodes(con)}

    def test_falling_below_exit_rank_closes_episode(self, con):
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP, overrides={"C00USDT": 0.001})
        changes = update_watchlist(con, self.NOW + self.STEP)

        assert {"symbol": "C00USDT", "tf": "1d", "market": "spot",
                "reason": "выпала по рангу", "narrow_bars": 10} in changes["exited"]
        assert "C00USDT" not in {e["symbol"] for e in storage.open_episodes(con)}

    def test_a_coin_that_stopped_being_scanned_closes(self, con):
        """Замер 14.09.2026: AIOTUSDT висел «active» шестые сутки.

        Правило «выпала из универсума» было в коде с самого начала, но не
        срабатывало никогда: ранг брался из последней строки журнала
        независимо от её возраста, поэтому у переставшего сканироваться
        символа он оставался прежним.
        """
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        # Новых сканов нет ни по кому: сборщик до этих символов больше не
        # доходит, а строки от NOW остались в журнале.
        changes = update_watchlist(con, self.NOW + 3 * self.STEP)

        assert storage.open_episodes(con) == []
        assert {e["reason"] for e in changes["exited"]} == {"перестала обновляться"}
        closed = storage.episodes(con, status="closed")
        assert all("не обновлялись" in row["exit_reason"] for row in closed)

    def test_a_frozen_row_does_not_hold_a_slot(self, con):
        """Живой кандидат входит вместо замороженного лидера, а не после него."""
        from cryptomcp.collector import WATCH_ENTER_RANK, update_watchlist

        # Замороженный лидер: индекс выше всех, но свеча трёхдневной давности.
        self.scan(con, "ЗАМЁРЗUSDT", 0.99, self.NOW - 3 * self.STEP)
        self.market(con, self.NOW)
        changes = update_watchlist(con, self.NOW)

        entered = {e["symbol"] for e in changes["entered"]}
        assert "ЗАМЁРЗUSDT" not in entered
        assert len(entered) == WATCH_ENTER_RANK

    def test_breakout_of_a_frozen_row_is_still_a_breakout(self, con):
        """Пробой до заморозки — состоявшийся факт, а не «перестала обновляться».

        Иначе журнал систематически терял бы именно сработавшие монеты:
        выстрелила, вылетела из универсума по обороту — и записана выбывшей.
        """
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.scan(con, "C00USDT", 0.50, self.NOW + self.STEP, price=200.0)
        con.commit()
        changes = update_watchlist(con, self.NOW + 4 * self.STEP)

        exited = {e["symbol"]: e["reason"] for e in changes["exited"]}
        assert exited["C00USDT"] == "пробой"

    def test_confirmed_distribution_dismisses(self, con):
        """Снятие с наблюдения по признаку, а не по рангу (§6.1).

        OAX за шесть месяцев двадцать раз подряд дал один и тот же ответ, а
        сканер держал бы монету в списке: «накопления нет» — пассивный ответ,
        монета просто опускается в ранге.
        """
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP)
        con.execute(
            "UPDATE scan_log SET dist_verdict = 'confirmed' "
            "WHERE symbol = 'C00USDT' AND ts_ms = ?", (self.NOW + self.STEP,)
        )
        con.commit()
        changes = update_watchlist(con, self.NOW + self.STEP)

        assert {"symbol": "C00USDT", "tf": "1d", "market": "spot",
                "reason": "распределение", "narrow_bars": 10} in changes["exited"]
        closed = [
            row for row in storage.episodes(con, status="dismissed")
            if row["symbol"] == "C00USDT"
        ]
        assert closed and closed[0]["exit_reason"] == "распределение подтверждено"

    def test_forming_distribution_keeps_the_episode(self, con):
        """Три события с плоскими максимумами — не снятие. Это намеренно."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP)
        con.execute("UPDATE scan_log SET dist_verdict = 'forming'")
        con.commit()
        update_watchlist(con, self.NOW + self.STEP)

        assert "C00USDT" in {e["symbol"] for e in storage.open_episodes(con)}

    def test_breakout_wins_over_distribution(self, con):
        """Выстрелила и распределяется — это сработавший сигнал (§6.1)."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP)
        self.scan(con, "C00USDT", 0.60, self.NOW + self.STEP, price=200.0)
        con.execute(
            "UPDATE scan_log SET dist_verdict = 'confirmed' "
            "WHERE symbol = 'C00USDT' AND ts_ms = ?", (self.NOW + self.STEP,)
        )
        con.commit()
        changes = update_watchlist(con, self.NOW + self.STEP)

        exited = {e["symbol"]: e["reason"] for e in changes["exited"]}
        assert exited["C00USDT"] == "пробой"

    def test_breakout_wins_over_rank(self, con):
        """Выстрелила и вылетела из топа — это сработавший сигнал, не выбывший."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP)
        self.scan(con, "C00USDT", 0.001, self.NOW + self.STEP, price=200.0)
        con.commit()
        changes = update_watchlist(con, self.NOW + self.STEP)

        assert {"symbol": "C00USDT", "tf": "1d", "market": "spot",
                "reason": "пробой", "narrow_bars": 10} in changes["exited"]
        row = con.execute(
            "SELECT status FROM watchlist WHERE symbol = 'C00USDT'"
        ).fetchone()
        assert row["status"] == "broken_out"

    def test_expiry_after_the_deadline(self, con):
        from cryptomcp.collector import WATCH_MAX_DAYS, update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        later = self.NOW + (WATCH_MAX_DAYS + 1) * 86_400_000
        self.market(con, later)
        changes = update_watchlist(con, later)

        assert {"symbol": "C00USDT", "tf": "1d", "market": "spot",
                "reason": "истёк срок", "narrow_bars": 10} in changes["exited"]

    def test_manual_entry_survives_low_rank(self, con):
        """Сканер видит только то, что умеет измерять."""
        from cryptomcp.collector import add_to_watchlist, update_watchlist

        self.market(con, self.NOW)
        add_to_watchlist(con, "C49USDT", "1d", self.NOW)
        update_watchlist(con, self.NOW + self.STEP)

        row = con.execute(
            "SELECT entered_by, exited_at FROM watchlist WHERE symbol = 'C49USDT'"
        ).fetchone()
        assert row["entered_by"] == "manual"
        assert row["exited_at"] is None

    def test_manual_entry_still_closes_on_breakout(self, con):
        from cryptomcp.collector import add_to_watchlist, update_watchlist

        self.market(con, self.NOW)
        add_to_watchlist(con, "C49USDT", "1d", self.NOW)
        self.market(con, self.NOW + self.STEP)
        self.scan(con, "C49USDT", 0.11, self.NOW + self.STEP, price=200.0)
        con.commit()
        update_watchlist(con, self.NOW + self.STEP)

        row = con.execute(
            "SELECT status FROM watchlist WHERE symbol = 'C49USDT'"
        ).fetchone()
        assert row["status"] == "broken_out"

    def test_history_of_episodes_is_kept(self, con):
        """Монета попадает в список не раз в жизни — прошлый эпизод не затирать."""
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        self.market(con, self.NOW + self.STEP, overrides={"C00USDT": 0.001})
        update_watchlist(con, self.NOW + self.STEP)
        self.market(con, self.NOW + 2 * self.STEP)
        update_watchlist(con, self.NOW + 2 * self.STEP)

        episodes = con.execute(
            "SELECT COUNT(*) c FROM watchlist WHERE symbol = 'C00USDT'"
        ).fetchone()["c"]
        assert episodes == 2, "новый вход — новая строка, а не перезапись"

    def test_only_one_open_episode_per_pair(self, con):
        from cryptomcp.collector import update_watchlist

        self.market(con, self.NOW)
        update_watchlist(con, self.NOW)
        update_watchlist(con, self.NOW + 1000)

        opened = con.execute(
            "SELECT COUNT(*) c FROM watchlist "
            "WHERE symbol = 'C00USDT' AND exited_at IS NULL"
        ).fetchone()["c"]
        assert opened == 1


class TestScanWithoutWatch:
    """1h считается и логируется, но эпизодов не открывает.

    Третий ТФ в списке поднял бы потолок с 80 эпизодов до 120 (топ-40 на
    выход × число ТФ), а список и без того длинный. Гипотеза о 1h проверяется
    журналом и исходами, а не тем, что он занимает место в списке.
    """

    NOW = 1_756_000_000_000

    def test_watch_narrower_than_scan(self):
        from cryptomcp.collector import SCAN_TIMEFRAMES, WATCH_TIMEFRAMES

        assert set(WATCH_TIMEFRAMES) < set(SCAN_TIMEFRAMES)
        assert "1h" in SCAN_TIMEFRAMES
        assert "1h" not in WATCH_TIMEFRAMES

    def test_hourly_scan_opens_no_episodes(self, con):
        from cryptomcp import SQUEEZE_FORMULA_VERSION
        from cryptomcp.collector import update_watchlist

        for i in range(30):
            con.execute(
                "INSERT OR REPLACE INTO scan_log (ts_ms, symbol, source, tf, "
                "formula_version, squeeze_index, price, range_low, range_high, "
                "closed_through_ms) VALUES (?, ?, 'futures', '1h', ?, ?, "
                "100.0, 90.0, 110.0, ?)",
                (self.NOW, f"H{i:02d}USDT", SQUEEZE_FORMULA_VERSION,
                 0.90 - i * 0.01, self.NOW),
            )
        con.commit()

        changes = update_watchlist(con, self.NOW)

        assert changes["entered"] == []
        assert storage.open_episodes(con) == []


class FakeFundingClient:
    """Биржа, отдающая начисления ВПЕРЁД от startTime — как настоящая.

    Направление здесь противоположно /futures/data/, и именно поэтому у него
    отдельный фейк: общий скрывал бы главное различие.
    """

    def __init__(self, first_ts: int, last_ts: int, step_ms: int, page: int = 1000):
        self.first_ts, self.last_ts, self.step_ms = first_ts, last_ts, step_ms
        self.page = page
        self.calls: list[int | None] = []

    async def funding_rate(self, symbol, *, limit=1000, start_time=None,
                           end_time=None):
        self.calls.append(start_time)
        start = max(start_time or self.first_ts, self.first_ts)
        end = min(end_time or self.last_ts, self.last_ts)
        stamps, ts = [], start - (start - self.first_ts) % self.step_ms
        if ts < start:
            ts += self.step_ms
        while ts <= end and len(stamps) < min(limit, self.page):
            stamps.append(ts)
            ts += self.step_ms
        return [
            {"symbol": symbol, "fundingTime": t, "fundingRate": "-0.0001"}
            for t in stamps
        ]

    async def now_ms(self):
        return self.last_ts


class TestFundingCollection:
    """Фандинг нужен строкой скана, поэтому лежит в архиве, а не берётся живьём."""

    STEP = 8 * 3_600_000

    def client(self, days=400, page=1000):
        return FakeFundingClient(
            NOW - int(days * 86_400_000), NOW, self.STEP, page=page
        )

    @pytest.mark.asyncio
    async def test_first_run_takes_a_year(self, con):
        from cryptomcp.collector import FUNDING_BACKFILL_DAYS, collect_funding

        client = self.client()
        written, failed = await collect_funding(client, con, ["BTCUSDT"])

        assert failed == []
        # Год начислений с шагом 8 часов — три в сутки.
        assert written == pytest.approx(FUNDING_BACKFILL_DAYS * 3, abs=2)

    @pytest.mark.asyncio
    async def test_paging_goes_forward(self, con):
        """fundingRate листается вперёд по startTime — обратно к /futures/data/."""
        from cryptomcp.collector import collect_funding

        client = self.client(page=200)
        await collect_funding(client, con, ["BTCUSDT"])

        starts = [s for s in client.calls if s is not None]
        assert starts == sorted(starts), "курсор обязан идти слева направо"
        assert len(starts) > 1, "год по 200 точек за страницу — не одна страница"

    @pytest.mark.asyncio
    async def test_second_run_asks_only_for_the_tail(self, con):
        from cryptomcp.collector import collect_funding

        client = self.client()
        await collect_funding(client, con, ["BTCUSDT"])
        before = len(client.calls)

        client.last_ts = NOW + 3 * self.STEP
        written, _ = await collect_funding(client, con, ["BTCUSDT"])

        assert written == 3, "должен догрузиться только хвост"
        assert len(client.calls) - before == 1

    @pytest.mark.asyncio
    async def test_repeat_run_is_idempotent(self, con):
        from cryptomcp.collector import collect_funding

        client = self.client(days=30)
        await collect_funding(client, con, ["BTCUSDT"])
        written, failed = await collect_funding(client, con, ["BTCUSDT"])

        assert (written, failed) == (0, [])
        rows = con.execute("SELECT COUNT(*) AS n FROM funding").fetchone()["n"]
        assert rows == 30 * 3 + 1


class TestAccumulationLogging:
    """Признаки накопления пишутся в scan_log с первого дня.

    Через два месяца вопрос будет не «работает ли накопление», а «какой из
    признаков работает», и ответить можно только по журналу.
    """

    NOW = 1_788_400_000_000
    H1 = 3_600_000

    def fill_hourly(self, con, symbol, count=1800):
        rows = []
        for i in range(count):
            ts = self.NOW - (count - i) * self.H1
            volume = 5000.0 if i == count - 20 else 1000.0
            rows.append((ts, 100.0, 100.5, 99.5, 100.0, 10.0, volume, 10,
                         5.0, volume * 0.6))
        storage.upsert_ohlcv(con, symbol, "1h", "futures", rows)
        con.commit()

    def test_context_reads_only_the_archive(self, con):
        from cryptomcp.collector import accumulation_context

        self.fill_hourly(con, "AAAUSDT")
        context = accumulation_context(con, "AAAUSDT", self.NOW)

        assert context["absorption_tf"] == "1h"
        assert "taker_streak" in context and "lead_state" in context
        # Деривативов и фандинга в базе нет — колонки просто отсутствуют.
        assert "oi_reading" not in context
        assert "funding_annual" not in context

    def test_funding_interval_derived_from_data(self, con):
        """Интервал начисления берётся из промежутков, а не спрашивается у биржи."""
        from cryptomcp.collector import funding_annual

        step = 8 * 3_600_000
        settlements = [(self.NOW - (20 - i) * step, 0.0001) for i in range(20)]
        assert funding_annual(settlements) == pytest.approx(10.95, abs=0.01)

        step = 4 * 3_600_000
        settlements = [(self.NOW - (20 - i) * step, 0.0001) for i in range(20)]
        assert funding_annual(settlements) == pytest.approx(21.9, abs=0.01)

    def test_too_short_history_gives_nothing(self, con):
        from cryptomcp.collector import funding_annual

        assert funding_annual([(self.NOW, 0.0001)]) is None

    def fill_open_interest(self, con, symbol, *, end, jump_at=None):
        """Открытый интерес с шагом 5m за двое суток до ``end``, со скачком."""
        step = 300_000
        rows = {}
        for ts in range(end - 2 * 86_400_000, end + 1, step):
            contracts = 1_000_000.0 * (1.5 if jump_at is not None and ts >= jump_at else 1.0)
            rows[ts] = {"open_interest": contracts, "open_interest_value": contracts * 0.1}
        storage.upsert_derivatives(con, symbol, rows)
        con.commit()

    def test_context_does_not_see_hourly_candles_after_close(self, con):
        """Свеча, открывшаяся после закрытия строки, в признаки не попадает.

        Строка по свече пишется позже её закрытия — у дневок в половине случаев
        позже часа. Всплеск объёма в эти часы был бы заглядыванием вперёд.
        """
        from cryptomcp.collector import accumulation_context

        self.fill_hourly(con, "AAAUSDT")
        close = self.NOW - 3 * self.H1
        before = accumulation_context(con, "AAAUSDT", close)

        spike = [(self.NOW + i * self.H1, 100.0, 130.0, 99.0, 129.0, 10.0, 90_000.0, 10,
                  5.0, 80_000.0) for i in range(3)]
        storage.upsert_ohlcv(con, "AAAUSDT", "1h", "futures", spike)
        con.commit()

        assert accumulation_context(con, "AAAUSDT", close) == before

    def test_context_does_not_see_open_interest_after_close(self, con):
        """Разбор ONEUSDT 16.09: «приток +37%» в строке был самим выстрелом."""
        from cryptomcp.collector import accumulation_context

        self.fill_hourly(con, "AAAUSDT")
        close = self.NOW - 3 * self.H1
        self.fill_open_interest(con, "AAAUSDT", end=self.NOW, jump_at=close + self.H1)

        context = accumulation_context(con, "AAAUSDT", close)
        assert context["oi_change_24h"] == pytest.approx(0.0, abs=1e-9)

        later = accumulation_context(con, "AAAUSDT", self.NOW)
        assert later["oi_change_24h"] == pytest.approx(50.0, abs=0.01)

    def test_scan_takes_context_at_candle_close_not_at_run_time(self, con):
        """Строка 4h получает открытый интерес на своё закрытие.

        Прогон идёт позже закрытия, и к этому моменту в архиве уже лежат точки
        после свечи. Раньше контекст брался на время прогона, и у ONEUSDT
        16.09 строка по свече 20:00, записанная в 23:45, получила «приток
        +37%» — сам выстрел.
        """
        import random

        from cryptomcp.collector import run_scan

        step = 4 * self.H1
        random.seed(5)
        price, candles = 100.0, []
        for i in range(400):
            price *= 1 + random.gauss(0, 0.004)
            candles.append((self.NOW - (400 - i) * step, price, price * 1.01, price * 0.99,
                            price, 10.0, 1000.0, 5, 5.0, 500.0))
        storage.upsert_ohlcv(con, "AAAUSDT", "4h", "futures", candles)
        self.fill_hourly(con, "AAAUSDT")
        self.fill_open_interest(
            con, "AAAUSDT", end=self.NOW + 2 * self.H1, jump_at=self.NOW + self.H1
        )

        run_scan(con)

        row = con.execute(
            "SELECT closed_through_ms, oi_change_24h FROM scan_log WHERE tf = '4h'"
        ).fetchone()
        assert row["closed_through_ms"] == self.NOW - 1
        assert row["oi_change_24h"] == pytest.approx(0.0, abs=1e-9)

    def test_hourly_rows_get_no_accumulation(self, con):
        """У записей 1h младшего ряда в архиве нет: колонки остаются пустыми."""
        from cryptomcp.collector import run_scan

        self.fill_hourly(con, "AAAUSDT")
        run_scan(con)

        rows = con.execute(
            "SELECT tf, absorption_tf FROM scan_log ORDER BY tf"
        ).fetchall()
        by_tf = {row["tf"]: row["absorption_tf"] for row in rows}
        assert by_tf.get("1h") is None


class TestAccumulationList:
    """Второй список: отбор по кластеру набора, а не по тишине (PLAN §4.43).

    Замер 17.09.2026: верхушка индекса на исходах не отличается от базы (72ч,
    доля от +10%: 1d 44% против 45% у всех), а кластер даёт 51% против 39%.
    Кластер есть лишь у 6% строк верхушки, поэтому он не ворота к первому
    списку, а основание второго.
    """

    NOW = 1_788_400_000_000

    def scan(self, con, symbol, index, *, clusters, change=0.0, ts=None):
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "squeeze_index, closed_through_ms, absorption_clusters, absorption_tf, "
            "change_24h_pct, price) VALUES (?, ?, 'spot', '1d', ?, ?, ?, ?, '1h', ?, 1.0)",
            (ts or self.NOW, symbol, SQUEEZE_FORMULA_VERSION, index,
             (ts or self.NOW) - 1, clusters, change),
        )
        con.commit()

    def names(self, con, status="active"):
        return [row["symbol"] for row in storage.accumulation_entries(con, status=status)]

    def test_only_coins_with_a_cluster_enter(self, con):
        from cryptomcp.collector import update_accumulation

        self.scan(con, "CLUSTERUSDT", 0.40, clusters=1)
        self.scan(con, "QUIETUSDT", 0.95, clusters=0)

        changes = update_accumulation(con, now_ms=self.NOW)

        assert self.names(con) == ["CLUSTERUSDT"]
        assert [e["symbol"] for e in changes["entered"]] == ["CLUSTERUSDT"]

    def test_coin_already_in_motion_is_not_taken(self, con):
        """Ворота хода за сутки те же, что у первого списка: +20% — не набор."""
        from cryptomcp.collector import update_accumulation

        self.scan(con, "FLYINGUSDT", 0.80, clusters=2, change=20.0)

        update_accumulation(con, now_ms=self.NOW)

        assert self.names(con) == []

    def test_short_squeeze_duration_does_not_block(self, con):
        """Порог семи суток сжатия сюда не переносится: накопление бывает коротким."""
        from cryptomcp.collector import update_accumulation

        self.scan(con, "FRESHUSDT", 0.30, clusters=1)

        update_accumulation(con, now_ms=self.NOW)

        assert self.names(con) == ["FRESHUSDT"]

    def test_entry_closes_when_the_cluster_is_gone(self, con):
        from cryptomcp.collector import update_accumulation

        self.scan(con, "CLUSTERUSDT", 0.80, clusters=1)
        update_accumulation(con, now_ms=self.NOW)

        later = self.NOW + 86_400_000
        self.scan(con, "CLUSTERUSDT", 0.80, clusters=0, ts=later)
        changes = update_accumulation(con, now_ms=later)

        assert self.names(con) == []
        assert changes["exited"][0]["reason"] == "кластер пропал"

    def test_hysteresis_keeps_a_slipping_entry(self, con):
        """Вход в топ-15, выход из топ-25 — иначе запись мигала бы у границы."""
        from cryptomcp.collector import update_accumulation

        self.scan(con, "SLIPUSDT", 0.90, clusters=1)
        update_accumulation(con, now_ms=self.NOW)

        later = self.NOW + 86_400_000
        for i in range(20):
            self.scan(con, f"BETTER{i}USDT", 0.95, clusters=1, ts=later)
        self.scan(con, "SLIPUSDT", 0.10, clusters=1, ts=later)

        update_accumulation(con, now_ms=later)

        assert "SLIPUSDT" in self.names(con), "ранг 21 ещё внутри топ-25"

    def test_entry_closes_below_the_exit_rank(self, con):
        from cryptomcp.collector import update_accumulation

        self.scan(con, "SLIPUSDT", 0.90, clusters=1)
        update_accumulation(con, now_ms=self.NOW)

        later = self.NOW + 86_400_000
        for i in range(30):
            self.scan(con, f"BETTER{i}USDT", 0.95, clusters=1, ts=later)
        self.scan(con, "SLIPUSDT", 0.10, clusters=1, ts=later)

        changes = update_accumulation(con, now_ms=later)

        assert "SLIPUSDT" not in self.names(con)
        assert any(e["symbol"] == "SLIPUSDT" for e in changes["exited"])

    def test_watchlist_is_untouched(self, con):
        """Второй список ничего не меняет в первом — сравнение должно быть честным."""
        from cryptomcp.collector import update_accumulation

        self.scan(con, "CLUSTERUSDT", 0.90, clusters=1)
        update_accumulation(con, now_ms=self.NOW)

        assert con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 0


class TestUniverseHysteresis:
    """Гистерезис архивного слоя: вход по 3M, выход ниже 2M и не за один день.

    Замер на ONEUSDT: 11–14.09 оборот 2.27, 1.87, 2.07, 2.07M — монета выпала
    из скана на тихой фазе перед выстрелом 16.09.
    """

    @staticmethod
    def rows(*pairs):
        return [{"symbol": s, "quote_volume_24h": v} for s, v in pairs]

    def test_new_symbol_needs_the_entry_threshold(self):
        from cryptomcp.collector import kept_rows

        kept = kept_rows(self.rows(("НОВАЯUSDT", 2_500_000.0)), {}, {})

        assert kept == []

    def test_known_symbol_stays_above_keep_threshold(self):
        from cryptomcp.collector import kept_rows

        kept = kept_rows(
            self.rows(("ONEUSDT", 2_270_000.0)), {"ONEUSDT": 1}, {"ONEUSDT": 3_000_000.0}
        )

        assert [row["symbol"] for row in kept] == ["ONEUSDT"]

    def test_single_quiet_day_does_not_evict(self):
        """12.09 у ONEUSDT было 1.87M, но соседние дни выше порога удержания."""
        from cryptomcp.collector import kept_rows

        kept = kept_rows(
            self.rows(("ONEUSDT", 1_870_000.0)), {"ONEUSDT": 1}, {"ONEUSDT": 2_270_000.0}
        )

        assert [row["symbol"] for row in kept] == ["ONEUSDT"]

    def test_three_quiet_days_in_a_row_evict(self):
        from cryptomcp.collector import kept_rows

        kept = kept_rows(
            self.rows(("DEADUSDT", 1_500_000.0)), {"DEADUSDT": 1}, {"DEADUSDT": 1_800_000.0}
        )

        assert kept == []

    def test_peak_of_recent_snapshots_is_what_holds(self, con):
        """Оконный максимум берётся из снимков универсума, а не из среднего."""
        con.execute(
            "INSERT INTO universe_daily (date, symbol, quote_volume_24h) VALUES "
            "('2026-09-15', 'ONEUSDT', 3050000), ('2026-09-16', 'ONEUSDT', 1900000)"
        )
        con.commit()

        peaks = storage.recent_universe_volume(con, days=2)

        assert peaks["ONEUSDT"] == pytest.approx(3_050_000.0)


class TestAdmission:
    """Понижение порога приводит сотню новых монет разом (PLAN §4.30).

    Растягивание приёма ничего не стоит: окно деривативов у новичка всё равно
    начинается от границы доступных 30 суток, а не от момента приёма, — монета,
    впущенная через пять часов, получит ту же историю.
    """

    def rows(self, count, start=100_000_000):
        """Универсум, отсортированный по обороту, как его отдаёт universe_rows."""
        return [
            {"symbol": f"C{i:03d}USDT", "quote_volume_24h": start - i * 1_000_000}
            for i in range(count)
        ]

    def test_new_symbols_are_capped(self, con):
        from cryptomcp.collector import admit

        admitted, waiting = admit(con, self.rows(30), limit=10)

        assert len(admitted) == 10
        assert waiting == 20

    def test_biggest_new_symbols_go_first(self, con):
        from cryptomcp.collector import admit

        admitted, _ = admit(con, self.rows(30), limit=3)
        assert [row["symbol"] for row in admitted] == [
            "C000USDT", "C001USDT", "C002USDT"
        ]

    def test_known_symbols_are_never_held_back(self, con):
        """Ограничитель тормозит приём, а не обслуживание уже принятых."""
        from cryptomcp.collector import admit
        from cryptomcp.markets import FUTURES

        rows = self.rows(30)
        for row in rows[10:]:
            storage.remember_symbol(con, row["symbol"], FUTURES.name, 0)
        con.commit()

        admitted, waiting = admit(con, rows, limit=2)

        assert len(admitted) == 22          # 20 знакомых плюс двое новых
        assert waiting == 8
        assert {row["symbol"] for row in rows[10:]} <= {
            row["symbol"] for row in admitted
        }

    def test_nothing_new_means_no_queue(self, con):
        from cryptomcp.collector import admit
        from cryptomcp.markets import FUTURES

        rows = self.rows(5)
        for row in rows:
            storage.remember_symbol(con, row["symbol"], FUTURES.name, 0)
        con.commit()

        admitted, waiting = admit(con, rows, limit=1)
        assert len(admitted) == 5
        assert waiting == 0
