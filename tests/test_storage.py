"""Тесты хранилища (SQLite).

Проверяется главным образом одно свойство: повторный прогон не должен ни
падать, ни портить уже записанное. Окна перекрываются всегда — сборщик
дозаписывает от последней точки, а биржа отдаёт с запасом.
"""

from __future__ import annotations

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
