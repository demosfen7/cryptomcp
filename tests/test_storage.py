"""Тесты хранилища (SQLite).

Проверяется главным образом одно свойство: повторный прогон не должен ни
падать, ни портить уже записанное. Окна перекрываются всегда — сборщик
дозаписывает от последней точки, а биржа отдаёт с запасом.
"""

from __future__ import annotations

import sqlite3

import pytest

from cryptomcp import storage


@pytest.fixture
def con(tmp_path):
    connection = storage.connect(str(tmp_path / "market.sqlite"))
    yield connection
    connection.close()


class TestSchema:
    def test_tables_created(self, con):
        names = {
            row["name"]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "derivatives", "funding", "universe_daily", "symbols", "collector_runs"
        } <= names

    def test_wal_enabled(self, con):
        """Без WAL читатель ловит database is locked на каждой записи."""
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_tables_are_without_rowid(self, con):
        """WITHOUT ROWID: замерено 98 байт на строку против 128."""
        for table in ("derivatives", "funding", "universe_daily", "symbols"):
            sql = con.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (table,)
            ).fetchone()["sql"]
            assert "WITHOUT ROWID" in sql, table


class TestDerivatives:
    def test_write_and_read_back(self, con):
        written = storage.upsert_derivatives(
            con, "BTCUSDT", {1000: {"open_interest": 5.0, "ls_global": 1.2}}
        )
        assert written == 1
        row = con.execute("SELECT * FROM derivatives").fetchone()
        assert row["open_interest"] == 5.0
        assert row["ls_global"] == 1.2
        assert row["taker_ratio"] is None

    def test_repeat_does_not_duplicate(self, con):
        rows = {1000: {"open_interest": 5.0}, 1300: {"open_interest": 6.0}}
        storage.upsert_derivatives(con, "BTCUSDT", rows)
        storage.upsert_derivatives(con, "BTCUSDT", rows)
        assert con.execute("SELECT COUNT(*) c FROM derivatives").fetchone()["c"] == 2

    def test_later_run_fills_missing_column(self, con):
        """Ряды приходят с разной свежестью, точка дописывается по частям."""
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"open_interest": 5.0}})
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"taker_ratio": 1.4}})
        row = con.execute("SELECT * FROM derivatives").fetchone()
        assert row["open_interest"] == 5.0
        assert row["taker_ratio"] == 1.4

    def test_absent_value_does_not_erase_stored_one(self, con):
        """Прогон без колонки не должен затирать её NULL-ом."""
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"open_interest": 5.0}})
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"ls_global": 1.1}})
        assert con.execute("SELECT open_interest FROM derivatives").fetchone()[0] == 5.0

    def test_new_value_overwrites(self, con):
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"open_interest": 5.0}})
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"open_interest": 7.0}})
        assert con.execute("SELECT open_interest FROM derivatives").fetchone()[0] == 7.0

    def test_symbols_do_not_mix(self, con):
        storage.upsert_derivatives(con, "BTCUSDT", {1000: {"open_interest": 5.0}})
        storage.upsert_derivatives(con, "ETHUSDT", {1000: {"open_interest": 9.0}})
        assert storage.coverage(con, "BTCUSDT")["points"] == 1
        assert storage.coverage(con, "ETHUSDT")["points"] == 1

    def test_empty_write_is_noop(self, con):
        assert storage.upsert_derivatives(con, "BTCUSDT", {}) == 0


class TestLastTs:
    def test_none_on_empty(self, con):
        assert storage.last_ts(con, "derivatives", "BTCUSDT") is None

    def test_returns_latest(self, con):
        storage.upsert_derivatives(
            con, "BTCUSDT", {1000: {"open_interest": 1.0}, 4000: {"open_interest": 2.0}}
        )
        assert storage.last_ts(con, "derivatives", "BTCUSDT") == 4000

    def test_unknown_table_rejected(self, con):
        with pytest.raises(ValueError):
            storage.last_ts(con, "ohlcv; DROP TABLE derivatives", "BTCUSDT")


class TestUniverse:
    def test_snapshot_written(self, con):
        storage.upsert_universe(con, "2026-09-02", [
            {"symbol": "BTCUSDT", "quote_volume_24h": 1e10, "price": 63000.0,
             "open_interest_usdt": 8e9, "has_spot": True},
        ])
        row = con.execute("SELECT * FROM universe_daily").fetchone()
        assert row["symbol"] == "BTCUSDT"
        assert row["has_spot"] == 1

    def test_same_day_updates_in_place(self, con):
        for volume in (1e9, 2e9):
            storage.upsert_universe(
                con, "2026-09-02", [{"symbol": "BTCUSDT", "quote_volume_24h": volume}]
            )
        rows = con.execute("SELECT quote_volume_24h FROM universe_daily").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 2e9

    def test_different_days_accumulate(self, con):
        for date in ("2026-09-01", "2026-09-02"):
            storage.upsert_universe(con, date, [{"symbol": "BTCUSDT"}])
        assert con.execute("SELECT COUNT(*) c FROM universe_daily").fetchone()["c"] == 2


class TestSymbols:
    def test_listing_date_remembered_once(self, con):
        """Дата листинга не меняется, повторный опрос не должен её переписывать."""
        storage.remember_symbol(con, "BTCUSDT", "futures", 1_500_000_000_000)
        storage.remember_symbol(con, "BTCUSDT", "futures", None)
        assert storage.known_symbols(con, "futures") == {"BTCUSDT": 1_500_000_000_000}

    def test_markets_kept_apart(self, con):
        storage.remember_symbol(con, "CAKEUSDT", "futures", 111)
        storage.remember_symbol(con, "CAKEUSDT", "spot", 222)
        assert storage.known_symbols(con, "spot") == {"CAKEUSDT": 222}


class TestRunLog:
    def test_run_recorded(self, con):
        storage.record_run(con, "incremental", symbols=60, rows=720, seconds=12.5)
        row = con.execute("SELECT * FROM collector_runs").fetchone()
        assert row["kind"] == "incremental"
        assert row["rows"] == 720
        assert row["error"] is None

    def test_failure_recorded(self, con):
        storage.record_run(
            con, "incremental", symbols=1, rows=0, seconds=0.1, error="таймаут"
        )
        assert con.execute("SELECT error FROM collector_runs").fetchone()[0] == "таймаут"


class TestOhlcv:
    def candle(self, ts, close=100.0):
        return (ts, close, close + 1, close - 1, close, 10.0, 1000.0, 42, 5.0, 500.0)

    def test_write_and_read_back(self, con):
        assert storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot",
                                    [self.candle(1000)]) == 1
        row = con.execute("SELECT * FROM ohlcv").fetchone()
        assert (row["symbol"], row["tf"], row["source"]) == ("CAKEUSDT", "1d", "spot")
        assert row["trades"] == 42

    def test_same_candle_is_overwritten(self, con):
        """Биржа доправляет последнюю свечу — сохранить надо исправленную."""
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot", [self.candle(1000, 100.0)])
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot", [self.candle(1000, 105.0)])
        rows = con.execute("SELECT c FROM ohlcv").fetchall()
        assert len(rows) == 1
        assert rows[0]["c"] == 105.0

    def test_timeframes_do_not_collide(self, con):
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot", [self.candle(1000)])
        storage.upsert_ohlcv(con, "CAKEUSDT", "4h", "spot", [self.candle(1000)])
        assert con.execute("SELECT COUNT(*) c FROM ohlcv").fetchone()["c"] == 2

    def test_last_ts_per_timeframe(self, con):
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot",
                             [self.candle(1000), self.candle(9000)])
        storage.upsert_ohlcv(con, "CAKEUSDT", "4h", "spot", [self.candle(2000)])
        assert storage.last_ohlcv_ts(con, "CAKEUSDT", "1d") == 9000
        assert storage.last_ohlcv_ts(con, "CAKEUSDT", "4h") == 2000
        assert storage.last_ohlcv_ts(con, "CAKEUSDT", "1h") is None

    def test_source_is_remembered_from_the_data(self, con):
        """Отдельного реестра источников нет — он рассинхронизировался бы."""
        assert storage.archive_source(con, "CAKEUSDT") is None
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot", [self.candle(1000)])
        assert storage.archive_source(con, "CAKEUSDT") == "spot"

    def test_coverage_groups_by_timeframe(self, con):
        storage.upsert_ohlcv(con, "CAKEUSDT", "1d", "spot",
                             [self.candle(1000), self.candle(2000)])
        storage.upsert_ohlcv(con, "UAIUSDT", "1d", "futures", [self.candle(3000)])
        rows = {(r["symbol"], r["tf"]): r for r in storage.ohlcv_coverage(con)}
        assert rows[("CAKEUSDT", "1d")]["candles"] == 2
        assert rows[("UAIUSDT", "1d")]["source"] == "futures"


class FakeView:
    """Минимальное представление, какого хватает журналу скана."""

    def __init__(self, interval="4h", closed_through_ms=1000, index=0.42,
                 shock=None):
        self.interval = interval
        self.squeeze_index = index
        self.components = {"volatility": 0.5, "range": 0.1}
        self.excluded = []
        self.price = 100.0
        self.range_low, self.range_high = 95.0, 105.0
        self.range_width = 0.1
        self.narrow_bars = 3
        self.change_24h = 1.5
        self.atr_pct = 2.0
        self.rsi_value = 55.0
        self.ema_state, self.structure = "above", "HH/HL"
        self.bbw = type("M", (), {"pct_rank": 12.0})()
        self.volume = type(
            "V", (), {"ratio": 1.2, "taker_buy_mean": 0.51, "ma_ratio": 0.63}
        )()
        # Шока может не быть вовсе — сжатия нет или оно короче порога.
        self.shock = shock
        self.meta = {"closed_through_ms": closed_through_ms}


class TestScanLog:
    def test_row_written_with_components(self, con):
        scan_id = storage.record_scan(con, "CAKEUSDT", "spot", FakeView(),
                                      formula_version="v2")
        assert scan_id > 0
        row = con.execute("SELECT * FROM scan_log").fetchone()
        assert row["symbol"] == "CAKEUSDT"
        assert row["formula_version"] == "v2"
        assert '"volatility"' in row["components"]
        assert row["narrow_bars"] == 3

    def test_same_candle_written_once(self, con):
        """Сканер ходит раз в час, свеча 4h закрывается раз в четыре."""
        view = FakeView(closed_through_ms=5000)
        first = storage.record_scan(con, "CAKEUSDT", "spot", view, formula_version="v2")
        second = storage.record_scan(con, "CAKEUSDT", "spot", view, formula_version="v2")
        assert first > 0
        assert second == 0
        assert con.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"] == 1

    def test_next_candle_is_a_new_row(self, con):
        storage.record_scan(con, "CAKEUSDT", "spot", FakeView(closed_through_ms=5000),
                            formula_version="v2")
        storage.record_scan(con, "CAKEUSDT", "spot", FakeView(closed_through_ms=9000),
                            formula_version="v2")
        assert con.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"] == 2

    def test_timeframes_are_separate_rows(self, con):
        for tf in ("4h", "1d"):
            storage.record_scan(con, "CAKEUSDT", "spot",
                                FakeView(interval=tf, closed_through_ms=5000),
                                formula_version="v2")
        assert con.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"] == 2


class TestScanLogShockColumns:
    """Признаки §4.26 пишутся с первого дня, как и признаки накопления."""

    def test_shock_columns_written(self, con):
        from cryptomcp.analysis import Shock

        shock = Shock(bars_ago=9, range_atr=4.0, range_share=0.94, volume_ratio=5.6)
        storage.record_scan(con, "HOMEUSDT", "futures", FakeView(shock=shock),
                            formula_version="v4")

        row = con.execute("SELECT * FROM scan_log").fetchone()
        assert row["ma_ratio"] == pytest.approx(0.63)
        assert row["shock_atr"] == pytest.approx(4.0)
        assert row["shock_share"] == pytest.approx(0.94)
        assert row["shock_volume"] == pytest.approx(5.6)
        assert row["shock_bars_ago"] == 9

    def test_no_squeeze_leaves_shock_empty(self, con):
        """Пусто значит «сжатия нет», и это не то же самое, что ноль."""
        storage.record_scan(con, "BTCUSDT", "futures", FakeView(),
                            formula_version="v4")

        row = con.execute("SELECT * FROM scan_log").fetchone()
        assert row["shock_atr"] is None
        assert row["shock_bars_ago"] is None
        assert row["ma_ratio"] == pytest.approx(0.63)


class TestOutcomeQueries:
    def scan(self, con, ts_ms, price=100.0):
        return storage.record_scan(
            con, "CAKEUSDT", "spot",
            FakeView(closed_through_ms=ts_ms), formula_version="v2", ts_ms=ts_ms,
        )

    def test_only_matured_rows_are_pending(self, con):
        now = 10_000_000
        fresh = self.scan(con, now - 1000)
        old = self.scan(con, now - 5_000_000)
        pending = storage.pending_outcomes(con, "24h", 3_600_000, now)
        ids = {row["id"] for row in pending}
        assert old in ids
        assert fresh not in ids

    def test_already_settled_is_not_pending_again(self, con):
        now = 10_000_000
        scan_id = self.scan(con, now - 5_000_000)
        storage.record_outcome(con, scan_id, "24h", {
            "max_price": 110.0, "min_price": 90.0, "max_pct": 10.0,
            "min_pct": -10.0, "close_pct": 1.0, "candles": 24,
        })
        assert storage.pending_outcomes(con, "24h", 3_600_000, now) == []

    def test_horizons_are_independent(self, con):
        now = 10_000_000
        scan_id = self.scan(con, now - 5_000_000)
        storage.record_outcome(con, scan_id, "24h", {
            "max_price": 1.0, "min_price": 1.0, "max_pct": 0.0,
            "min_pct": 0.0, "close_pct": 0.0, "candles": 1,
        })
        assert len(storage.pending_outcomes(con, "72h", 3_600_000, now)) == 1

    def test_outcome_is_overwritten_not_duplicated(self, con):
        scan_id = self.scan(con, 1000)
        values = {"max_price": 1.0, "min_price": 1.0, "max_pct": 0.0,
                  "min_pct": 0.0, "close_pct": 0.0, "candles": 1}
        storage.record_outcome(con, scan_id, "24h", values)
        storage.record_outcome(con, scan_id, "24h", {**values, "max_pct": 5.0})
        rows = con.execute("SELECT max_pct FROM outcomes").fetchall()
        assert len(rows) == 1
        assert rows[0]["max_pct"] == 5.0


class TestArchiveQueries:
    def candle(self, ts, high, low, close=100.0):
        return (ts, close, high, low, close, 1.0, 1.0, 1, 1.0, 1.0)

    def test_candles_come_back_in_time_order(self, con):
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", [
            self.candle(3000, 1, 1), self.candle(1000, 1, 1), self.candle(2000, 1, 1),
        ])
        assert [row[0] for row in storage.load_candles(con, "CAKEUSDT", "1h")] == [
            1000, 2000, 3000
        ]

    def test_finest_timeframe_wins(self, con):
        for tf in ("1d", "4h", "1h"):
            storage.upsert_ohlcv(con, "CAKEUSDT", tf, "spot", [self.candle(1000, 1, 1)])
        assert storage.finest_tf(con, "CAKEUSDT") == "1h"

    def test_finest_of_what_exists(self, con):
        storage.upsert_ohlcv(con, "XUSDT", "1d", "spot", [self.candle(1000, 1, 1)])
        assert storage.finest_tf(con, "XUSDT") == "1d"
        assert storage.finest_tf(con, "НЕТУUSDT") is None

    def test_extremes_use_high_and_low(self, con):
        """Цель, задетую и откатившуюся, по закрытию было бы не видно."""
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", [
            self.candle(2000, high=120.0, low=99.0, close=100.0),
            self.candle(3000, high=101.0, low=80.0, close=95.0),
        ])
        window = storage.price_extremes(con, "CAKEUSDT", "1h", 1000, 3000)
        assert window["high"] == 120.0
        assert window["low"] == 80.0
        assert window["close"] == 95.0
        assert window["candles"] == 2

    def test_window_excludes_the_scan_candle_itself(self, con):
        storage.upsert_ohlcv(con, "CAKEUSDT", "1h", "spot", [
            self.candle(1000, high=999.0, low=1.0),
            self.candle(2000, high=110.0, low=90.0),
        ])
        window = storage.price_extremes(con, "CAKEUSDT", "1h", 1000, 2000)
        assert window["high"] == 110.0

    def test_empty_window_is_none(self, con):
        assert storage.price_extremes(con, "CAKEUSDT", "1h", 0, 100) is None


class TestFormulaVersionIsolation:
    """Индексы разных формул между собой не сравниваются, а ранг — сравнение."""

    def row(self, con, symbol, version, index, ts=1000):
        con.execute(
            "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
            "squeeze_index, closed_through_ms) VALUES (?, ?, 'spot', '4h', ?, ?, ?)",
            (ts, symbol, version, index, ts),
        )
        con.commit()

    def test_ranking_sees_only_the_current_formula(self, con):
        self.row(con, "СТАРАЯUSDT", "v1", 0.99)
        self.row(con, "НОВАЯUSDT", "v2", 0.10)
        symbols = [r["symbol"] for r in storage.latest_scan(con, "4h", "v2")]
        assert symbols == ["НОВАЯUSDT"]

    def test_same_candle_recalculated_under_new_formula(self, con):
        """После смены формулы текущая свеча должна пересчитаться сразу."""
        self.row(con, "CAKEUSDT", "v1", 0.30)
        self.row(con, "CAKEUSDT", "v2", 0.55)
        assert con.execute("SELECT COUNT(*) c FROM scan_log").fetchone()["c"] == 2
        assert storage.latest_scan(con, "4h", "v2")[0]["squeeze_index"] == 0.55

    def test_duplicate_within_one_version_still_blocked(self, con):
        self.row(con, "CAKEUSDT", "v2", 0.30)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO scan_log (ts_ms, symbol, source, tf, formula_version, "
                "squeeze_index, closed_through_ms) "
                "VALUES (1000, 'CAKEUSDT', 'spot', '4h', 'v2', 0.9, 1000)"
            )


class TestWatchlistQueries:
    """Выдача эпизодов наружу: фильтры, порядок и ранг при входе."""

    NOW = 1_756_000_000_000

    def episode(self, con, symbol, tf="4h", *, rank=1, index=0.5, price=100.0):
        return storage.open_episode(
            con, symbol, tf, entered_at=self.NOW, entered_by="scanner",
            scan={"squeeze_index": index, "price": price,
                  "range_low": price * 0.9, "range_high": price * 1.1},
            rank=rank,
        )

    def test_rank_at_entry_survives_touch(self, con):
        """Ранг входа не должен затираться текущим: их и просят рядом."""
        episode_id = self.episode(con, "AAAUSDT", rank=3)
        storage.touch_episode(con, episode_id, rank=27, index=0.31)

        row = storage.episodes(con)[0]
        assert row["rank_at_entry"] == 3
        assert row["last_rank"] == 27

    def test_open_only_by_default(self, con):
        self.episode(con, "AAAUSDT")
        closed = self.episode(con, "BBBUSDT")
        storage.close_episode(
            con, closed, status="broken_out", reason="пробой", ts_ms=self.NOW + 1
        )

        assert [row["symbol"] for row in storage.episodes(con)] == ["AAAUSDT"]
        assert len(storage.episodes(con, status="all")) == 2
        assert [row["symbol"] for row in storage.episodes(con, status="closed")] == [
            "BBBUSDT"
        ]

    def test_filters_by_status_and_timeframe(self, con):
        self.episode(con, "AAAUSDT", tf="4h")
        self.episode(con, "AAAUSDT", tf="1d")

        assert len(storage.episodes(con, tf="1d")) == 1
        assert len(storage.episodes(con, status="candidate")) == 2
        assert storage.episodes(con, status="active") == []

    def test_episode_without_rank_sorts_last(self, con):
        """Выпавшая из универсума монета не должна всплывать наверх из-за NULL."""
        lost = self.episode(con, "AAAUSDT", rank=1)
        storage.touch_episode(con, lost, rank=None, index=None)
        self.episode(con, "BBBUSDT", rank=9)

        assert [row["symbol"] for row in storage.episodes(con)] == [
            "BBBUSDT", "AAAUSDT"
        ]


class TestScanHistory:
    NOW = 1_756_000_000_000
    STEP = 4 * 3_600_000

    def scan(self, con, symbol, ts, index, version=None):
        from cryptomcp import SQUEEZE_FORMULA_VERSION

        con.execute(
            "INSERT OR REPLACE INTO scan_log (ts_ms, symbol, source, tf, "
            "formula_version, squeeze_index, components, price, closed_through_ms) "
            "VALUES (?, ?, 'futures', '4h', ?, ?, ?, 100.0, ?)",
            (ts, symbol, version or SQUEEZE_FORMULA_VERSION, index,
             '{"volatility": 0.5}', ts),
        )
        con.commit()

    def test_newest_first_and_limited(self, con):
        for i in range(5):
            self.scan(con, "AAAUSDT", self.NOW + i * self.STEP, 0.10 * i)

        rows = storage.scan_history(con, "aaausdt", "4h", limit=2)
        assert [row["ts_ms"] for row in rows] == [
            self.NOW + 4 * self.STEP, self.NOW + 3 * self.STEP
        ]

    def test_other_formula_version_excluded(self, con):
        """Индексы разных версий формулы в одной таблице несопоставимы."""
        self.scan(con, "AAAUSDT", self.NOW, 0.5)
        self.scan(con, "AAAUSDT", self.NOW + self.STEP, 0.9, version="v0")

        rows = storage.scan_history(con, "AAAUSDT", "4h")
        assert [row["squeeze_index"] for row in rows] == [0.5]


class TestMigration:
    """Колонка, добавленная после выпуска, на базе с уже накопленной историей.

    CREATE TABLE IF NOT EXISTS существующую таблицу не трогает, а база на
    сервере переживает выкладку. Без ALTER деплой уронил бы чтение эпизодов, и
    заметили бы это уже в проде.
    """

    OLD_SCHEMA = """
    CREATE TABLE watchlist (
        id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, tf TEXT NOT NULL,
        status TEXT NOT NULL, entered_at INTEGER NOT NULL,
        entered_by TEXT NOT NULL, squeeze_index REAL, accumulation_score REAL,
        price_at_entry REAL, range_low REAL, range_high REAL,
        last_rank INTEGER, last_index REAL, exited_at INTEGER, exit_reason TEXT
    );
    """

    def old_base(self, tmp_path):
        """База прошлой выкладки: все таблицы на месте, watchlist без колонки."""
        path = str(tmp_path / "old.sqlite")
        con = sqlite3.connect(path)
        con.executescript(storage.SCHEMA)
        con.execute("DROP TABLE watchlist")
        con.executescript(self.OLD_SCHEMA)
        con.execute(
            "INSERT INTO watchlist (symbol, tf, status, entered_at, entered_by,"
            " last_rank) VALUES ('OLDUSDT', '4h', 'active', 1, 'scanner', 5)"
        )
        con.commit()
        con.close()
        return path

    def test_existing_episodes_survive(self, tmp_path):
        path = self.old_base(tmp_path)
        con = storage.connect(path)
        try:
            rows = storage.episodes(con)
            assert [row["symbol"] for row in rows] == ["OLDUSDT"]
            assert rows[0]["rank_at_entry"] is None
            assert rows[0]["last_rank"] == 5
        finally:
            con.close()

    def test_migration_is_idempotent(self, tmp_path):
        path = self.old_base(tmp_path)
        storage.connect(path).close()
        con = storage.connect(path)
        try:
            columns = [
                row["name"] for row in con.execute("PRAGMA table_info(watchlist)")
            ]
            assert columns.count("rank_at_entry") == 1
        finally:
            con.close()

    def test_read_only_base_without_column_still_renders(self, tmp_path):
        """Сервер может открыть базу раньше, чем сборщик её дополнит."""
        from cryptomcp.collector import watchlist_view

        path = self.old_base(tmp_path)
        con = storage.connect(path, read_only=True)
        try:
            assert "OLDUSDT" in watchlist_view(con)
        finally:
            con.close()


class TestScreenScan:
    """Отбор идёт по журналу, а не пересчётом: пересчёт стоил бы запроса
    к бирже за свежим хвостом по каждому символу."""

    NOW = 1_788_400_000_000

    def scan(self, con, symbol, index, *, narrow=0, bars=None, above=None,
             lead=None, clusters=None):
        from cryptomcp import SQUEEZE_FORMULA_VERSION

        con.execute(
            "INSERT OR REPLACE INTO scan_log (ts_ms, symbol, source, tf, "
            "formula_version, squeeze_index, price, narrow_bars, "
            "absorption_bars, taker_above, volume_lead, absorption_clusters, "
            "closed_through_ms) "
            "VALUES (?, ?, 'futures', '4h', ?, ?, 100.0, ?, ?, ?, ?, ?, ?)",
            (self.NOW, symbol, SQUEEZE_FORMULA_VERSION, index, narrow,
             bars, above, lead, clusters, self.NOW - 1),
        )
        con.commit()

    def test_sorted_by_squeeze_by_default(self, con):
        self.scan(con, "AAAUSDT", 0.30)
        self.scan(con, "BBBUSDT", 0.70)

        rows = storage.screen_scan(con, "4h")
        assert [r["symbol"] for r in rows] == ["BBBUSDT", "AAAUSDT"]

    def test_duration_sort_puts_long_squeeze_first(self, con):
        self.scan(con, "AAAUSDT", 0.70, narrow=0)
        self.scan(con, "BBBUSDT", 0.30, narrow=36)

        rows = storage.screen_scan(con, "4h", sort_by="duration")
        assert [r["symbol"] for r in rows] == ["BBBUSDT", "AAAUSDT"]

    def test_accumulation_sort_uses_components(self, con):
        """Сводного числа нет, поэтому порядок — по составляющим."""
        self.scan(con, "AAAUSDT", 0.70, bars=0, above=0, lead=0)
        self.scan(con, "BBBUSDT", 0.30, bars=3, above=7, lead=8)

        rows = storage.screen_scan(con, "4h", sort_by="accumulation")
        assert [r["symbol"] for r in rows] == ["BBBUSDT", "AAAUSDT"]

    def test_earlier_versions_seen_after_a_bump(self, con):
        """Сразу после подъёма версии журнал пуст, но данные в нём есть."""
        from cryptomcp import SQUEEZE_FORMULA_VERSION

        self.scan(con, "AAAUSDT", 0.5)
        con.execute(
            "UPDATE scan_log SET formula_version = 'v0' WHERE symbol = 'AAAUSDT'"
        )
        con.commit()

        assert storage.latest_scan(con, "4h") == []
        assert storage.earlier_versions(con, "4h") == ["v0"]
        assert SQUEEZE_FORMULA_VERSION not in storage.earlier_versions(con, "4h")

    def test_cluster_outranks_single_bars(self, con):
        """Кластер важнее счётчика одиночных баров (§4.25).

        Замерено на живых данных: ASTER (набор) дал 3 бара и 1 кластер, UAI
        (реакция на движение) — 4 бара и 0 кластеров. По барам порядок вышел
        бы обратным правильному.
        """
        self.scan(con, "UAIUSDT", 0.50, bars=4, above=7, lead=4, clusters=0)
        self.scan(con, "ASTERUSDT", 0.50, bars=3, above=18, lead=29, clusters=1)

        rows = storage.screen_scan(con, "4h", sort_by="accumulation")
        assert [r["symbol"] for r in rows] == ["ASTERUSDT", "UAIUSDT"]

    def test_allowed_and_exclude(self, con):
        for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
            self.scan(con, symbol, 0.5)

        rows = storage.screen_scan(
            con, "4h", allowed={"AAAUSDT", "BBBUSDT"}, exclude={"BBBUSDT"}
        )
        assert [r["symbol"] for r in rows] == ["AAAUSDT"]

    def test_min_narrow_bars(self, con):
        self.scan(con, "AAAUSDT", 0.70, narrow=0)
        self.scan(con, "BBBUSDT", 0.30, narrow=5)

        rows = storage.screen_scan(con, "4h", min_narrow_bars=1)
        assert [r["symbol"] for r in rows] == ["BBBUSDT"]

    def test_returns_everything_matched_not_a_page(self, con):
        """Вызывающему нужно знать, сколько прошло: «показано 5» иначе
        неотличимо от «пятеро и есть весь рынок»."""
        for i in range(5):
            self.scan(con, f"C{i}USDT", 0.10 * i)

        rows = storage.screen_scan(con, "4h")
        assert len(rows) == 5
        assert [r["symbol"] for r in rows[:2]] == ["C4USDT", "C3USDT"]

    def test_listing_dates_from_archive(self, con):
        storage.remember_symbol(con, "AAAUSDT", "futures", 1_600_000_000_000)
        con.commit()
        assert storage.listing_dates(con) == {"AAAUSDT": 1_600_000_000_000}
