"""Приёмка хранилища части B: допуск, истечение, retention и готовый diff."""

from __future__ import annotations

import sqlite3

import pytest

from cryptomcp.orderbook import build_order_book
from cryptomcp.orderbook_watch import (
    RETENTION_AFTER_END_MIN,
    WatchAdmissionError,
    cleanup,
    connect,
    diffs,
    get_watch,
    record_snapshot,
    record_trades,
    start,
    trade_gaps,
    trades,
)

NOW = 1_789_870_000_000


@pytest.fixture
def db(tmp_path):
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    yield con
    con.close()


def _book(ts=NOW, *, extra_bid=False):
    bids = [["100", "10"], ["99.8", "20"]]
    if extra_bid:
        bids.append(["99.9", "7"])
    return build_order_book(
        {"bids": bids, "asks": [["100.2", "15"], ["100.4", "25"]]},
        timestamp_ms=ts,
        last_price=100.1,
        limit=100,
        depth_pcts=(0.25,),
        turnover_24h_usdt=5_000_000,
    )


def _start(db, **overrides):
    kwargs = dict(
        symbol="ARBUSDT", market="futures", interval_sec=5, duration_min=60,
        depth_pcts=(0.25,), depth_weight=5, market_weight_limit=2400, started_at=NOW,
    )
    kwargs.update(overrides)
    return start(db, **kwargs)


def test_b10_2_budget_rejection_happens_before_any_snapshot_is_written(db):
    """B.10 №2: превышение половины IP-бюджета не создаёт ни снимка, ни сессии."""
    with pytest.raises(WatchAdmissionError, match="просит 1500"):
        _start(db, interval_sec=1, depth_weight=25)

    assert db.execute("SELECT COUNT(*) FROM order_book_watches").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM watch_snapshots").fetchone()[0] == 0


def test_b10_3_expired_watch_stops_accepting_new_snapshots_after_collector_cleanup(db, tmp_path):
    """B.10 №3: следующий часовой шаг переводит закончившуюся active-сессию в expired."""
    watch = _start(db, duration_min=1)
    path = str(tmp_path / "order_book_watch.sqlite")

    result = cleanup(path, current_ms=watch["ends_at"] + 1)
    assert result["expired"] == 1
    assert get_watch(db, watch["watch_id"])["status"] == "expired"
    with pytest.raises(ValueError, match="больше не принимает"):
        record_snapshot(db, watch["watch_id"], _book(watch["ends_at"] + 1))


def test_b10_4_retention_cleanup_deletes_all_session_data(db, tmp_path):
    """B.10 №4: после retention исчезают строка сессии, raw-снимки и diffs."""
    watch = _start(db, duration_min=1)
    record_snapshot(db, watch["watch_id"], _book())
    record_snapshot(db, watch["watch_id"], _book(NOW + 1, extra_bid=True))
    path = str(tmp_path / "order_book_watch.sqlite")

    cleanup(path, current_ms=watch["ends_at"] + 1)
    result = cleanup(
        path,
        current_ms=watch["ends_at"] + (RETENTION_AFTER_END_MIN + 1) * 60_000,
    )
    assert result["deleted"] == 1
    assert get_watch(db, watch["watch_id"]) is None
    assert db.execute("SELECT COUNT(*) FROM watch_snapshots").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM watch_diffs").fetchone()[0] == 0


def test_b10_5_new_level_is_written_as_appeared_diff_during_second_snapshot(db):
    """B.10 №5: diff пишется при записи, а не вычисляется позже при чтении."""
    watch = _start(db)
    record_snapshot(db, watch["watch_id"], _book())
    record_snapshot(db, watch["watch_id"], _book(NOW + 1, extra_bid=True))

    events = diffs(db, watch["watch_id"])
    appeared = next(event for event in events if event["event_type"] == "appeared")
    assert appeared == {
        "watch_id": watch["watch_id"], "ts": NOW + 1, "side": "bid",
        "price": 99.9, "qty_before": 0.0, "qty_after": 7.0, "event_type": "appeared",
    }


def test_watch_database_sets_busy_timeout_for_server_and_collector_writers(db):
    """Р3: отдельный файл имеет busy_timeout, иначе уборка может сорвать тик."""
    assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert isinstance(db, sqlite3.Connection)


def test_c2_four_futures_watches_at_five_seconds_fit_but_fifth_is_rejected(db):
    """С2 №9: 60 веса depth + 240 веса aggTrades = 300 на сессию."""
    for index in range(4):
        _start(db, symbol=f"ARB{index}USDT", trade_weight=20)

    with pytest.raises(WatchAdmissionError, match=r"просит 300 .*занято 1200.*потолок.*1200"):
        _start(db, symbol="TOOFASTUSDT", trade_weight=20)


def test_c3_stores_raw_trades_and_records_aggregate_id_gap(db):
    """С1--С3: ручная сверка уровня видит сделки и недостающий интервал a."""
    session = _start(db)
    result = record_trades(
        db,
        session["watch_id"],
        [
            {"a": 10, "T": NOW + 1, "p": "100", "q": "2", "m": False},
            {"a": 12, "T": NOW + 3, "p": "100", "q": "3", "m": True},
        ],
        request_count=2,
        extra_pages=1,
    )

    assert result == {"stored": 2, "gaps": 1}
    stored_trades = trades(db, session["watch_id"], from_ts=NOW, to_ts=NOW + 5)
    assert [row["agg_id"] for row in stored_trades] == [10, 12]
    assert trade_gaps(db, session["watch_id"], from_ts=NOW, to_ts=NOW + 5) == [
        {
            "watch_id": session["watch_id"],
            "before_agg_id": 10,
            "after_agg_id": 12,
            "started_at": NOW + 1,
            "ended_at": NOW + 3,
        }
    ]
    stored = get_watch(db, session["watch_id"])
    assert stored["trade_request_count"] == 2
    assert stored["trade_extra_page_count"] == 1
    assert stored["trade_gap_count"] == 1
