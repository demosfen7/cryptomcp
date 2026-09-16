"""Приёмка серверных инструментов части B без сети и реальных ожиданий."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cryptomcp import orderbook_watch as watch
from cryptomcp.markets import FUTURES
from cryptomcp.orderbook import build_order_book
from cryptomcp.server import (
    _watch_tasks,
    get_order_book_watch_data,
    start_order_book_watch,
)

NOW = 1_789_870_000_000


def _book(ts=NOW, *, extra_bid=False):
    bids = [["100", "10"], ["99.8", "20"]]
    if extra_bid:
        bids.append(["99.6", "7"])
    return build_order_book(
        {"bids": bids, "asks": [["100.2", "15"], ["100.4", "25"]]},
        timestamp_ms=ts,
        last_price=100.1,
        limit=100,
        depth_pcts=(0.25,),
        turnover_24h_usdt=5_000_000,
    )


def _watch_db(monkeypatch, tmp_path):
    path = str(tmp_path / "order_book_watch.sqlite")
    real_connect, real_read_only = watch.connect, watch.read_only
    monkeypatch.setattr("cryptomcp.server.watch.connect", lambda: real_connect(path))
    monkeypatch.setattr("cryptomcp.server.watch.read_only", lambda: real_read_only(path))
    monkeypatch.setattr("cryptomcp.server.watch.now_ms", lambda: NOW)
    return path, real_connect


def _ctx(monkeypatch):
    registry = SimpleNamespace()

    async def get(symbol):
        return SimpleNamespace(symbol=symbol.upper())

    registry.get = get

    async def fake_ctx(market):
        assert market == "futures"
        return SimpleNamespace(), None, registry, None, FUTURES

    monkeypatch.setattr("cryptomcp.server._ctx", fake_ctx)


@pytest.mark.asyncio
async def test_b10_1_start_creates_watch_and_first_snapshot_within_interval(monkeypatch, tmp_path):
    """B.10 №1: запуск отдаёт id, а тикер сразу пишет первый снимок без sleep."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    _ctx(monkeypatch)

    async def fake_runner(watch_id):
        con = watch.connect()
        try:
            watch.record_snapshot(con, watch_id, _book(NOW + 1))
        finally:
            con.close()

    monkeypatch.setattr("cryptomcp.server._run_order_book_watch", fake_runner)
    text = await start_order_book_watch("ARBUSDT", interval_sec=5, duration_min=60)
    watch_id = text.split()[1]
    task = _watch_tasks.pop(watch_id)
    await task

    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        row = watch.get_watch(con, watch_id)
        assert row["snapshot_count"] == 1
        assert row["last_snapshot_at"] - row["started_at"] <= row["interval_sec"] * 1000
    finally:
        con.close()


@pytest.mark.asyncio
async def test_b10_6_diff_format_uses_written_events(monkeypatch, tmp_path):
    """B.10 №6: format=diff читает готовую watch_diffs, а raw-блок не нужен."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=60, depth_pcts=(0.25,), depth_weight=5,
            market_weight_limit=2400, started_at=NOW,
        )
        watch.record_snapshot(con, session["watch_id"], _book())
        watch.record_snapshot(con, session["watch_id"], _book(NOW + 1, extra_bid=True))
    finally:
        con.close()

    text = await get_order_book_watch_data(session["watch_id"], format="diff")

    assert "Diff между последовательными снимками" in text
    assert "appeared · bid 99.6" in text
    assert "Сырые снимки:" not in text


@pytest.mark.asyncio
async def test_b10_7_budget_rejects_fast_watch_and_accepts_fitting_one(monkeypatch, tmp_path):
    """B.10 №7: проверка использует остаток рынка до первого снимка, не 429 постфактум.

    При реальном futures-весе 5 и минимальном интервале 1с один watch просит
    300 ед/мин. В сочетании с лимитом в 10 сессий буквально занять половину
    потолка и одновременно отвергнуть более быстрый watch невозможно, поэтому
    ряд оставляет 60 ед/мин: 300 отклоняются, 60 принимаются.
    """
    _, connect = _watch_db(monkeypatch, tmp_path)
    _ctx(monkeypatch)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        for index in range(3):
            watch.start(
                con, symbol=f"FAST{index}USDT", market="futures", interval_sec=1,
                duration_min=60, depth_pcts=(0.25,), depth_weight=5,
                market_weight_limit=2400, started_at=NOW,
            )
        for index in range(4):
            watch.start(
                con, symbol=f"NORMAL{index}USDT", market="futures", interval_sec=5,
                duration_min=60, depth_pcts=(0.25,), depth_weight=5,
                market_weight_limit=2400, started_at=NOW,
            )
    finally:
        con.close()

    async def fake_runner(_watch_id):
        return None

    monkeypatch.setattr("cryptomcp.server._run_order_book_watch", fake_runner)
    rejected = await start_order_book_watch("FASTNEWUSDT", interval_sec=1)
    accepted = await start_order_book_watch("FITTINGUSDT", interval_sec=5)
    accepted_id = accepted.split()[1]
    await _watch_tasks.pop(accepted_id)

    assert json.loads(rejected)["error"]["kind"] == "bad_params"
    assert "просит 300 ед/мин" in rejected
    assert accepted.startswith("Сессия watch_")
