"""Приёмка серверных инструментов части B без сети и реальных ожиданий."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cryptomcp import orderbook_watch as watch
from cryptomcp.markets import FUTURES
from cryptomcp.orderbook import build_order_book
from cryptomcp.render import MAX_WATCH_RESPONSE_CHARS, render_order_book_watch_data
from cryptomcp.server import (
    _record_order_book_watch_snapshot,
    _record_order_book_watch_trades,
    _watch_tasks,
    get_order_book_watch_data,
    start_order_book_watch,
)

NOW = 1_789_870_000_000


def _book(ts=NOW, *, extra_bid=False):
    bids = [["100", "10"], ["99.8", "20"]]
    if extra_bid:
        bids[1] = ["99.8", "27"]
    return build_order_book(
        {"bids": bids, "asks": [["100.2", "15"], ["100.4", "25"]]},
        timestamp_ms=ts,
        last_price=100.1,
        limit=100,
        depth_pcts=(0.25,),
        turnover_24h_usdt=5_000_000,
    )


def _shifted_book(ts, *, shift: float):
    """Та же ликвидность, но край limit-окна сдвинут одним тиком."""
    return build_order_book(
        {
            "bids": [[str(100.0 + shift), "10"], [str(99.8 + shift), "10"]],
            "asks": [[str(100.2 + shift), "15"], [str(100.4 + shift), "15"]],
        },
        timestamp_ms=ts,
        last_price=100.1 + shift,
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
    client = SimpleNamespace()

    async def get(symbol):
        return SimpleNamespace(symbol=symbol.upper())

    registry.get = get

    async def ticker_24hr(_symbol):
        return {"quoteVolume": "5000000"}

    async def now_ms():
        return NOW

    client.ticker_24hr = ticker_24hr
    client.now_ms = now_ms

    async def fake_ctx(market):
        assert market == "futures"
        return client, None, registry, None, FUTURES

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
    assert "grew · bid 99.8" in text
    assert "Сырые снимки:" not in text


@pytest.mark.asyncio
async def test_shifted_limit_window_does_not_render_appeared_or_disappeared(monkeypatch, tmp_path):
    """Блокер 2: край двух L2-окон не превращается в биржевое событие."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=60, depth_pcts=(0.25,), depth_weight=5,
            market_weight_limit=2400, started_at=NOW,
        )
        watch.record_snapshot(con, session["watch_id"], _shifted_book(NOW, shift=0.0))
        watch.record_snapshot(con, session["watch_id"], _shifted_book(NOW + 5_000, shift=0.2))
    finally:
        con.close()

    text = await get_order_book_watch_data(session["watch_id"], format="diff")

    assert "appeared ·" not in text
    assert "disappeared ·" not in text
    assert "событий пока нет" in text


@pytest.mark.asyncio
async def test_b10_7_budget_rejects_fast_watch_and_accepts_fitting_one(monkeypatch, tmp_path):
    """B.10 №7: проверка использует остаток рынка до первого снимка, не 429 постфактум.

    После С2 watch 5с просит 300 ед/мин, а watch 1с — 540. Три обычных
    сессии занимают 900: быстрая не помещается, четвёртая обычная — ровно да.
    """
    _, connect = _watch_db(monkeypatch, tmp_path)
    _ctx(monkeypatch)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        for index in range(3):
            watch.start(
                con, symbol=f"NORMAL{index}USDT", market="futures", interval_sec=5,
                duration_min=60, depth_pcts=(0.25,), depth_weight=5,
                trade_weight=20, market_weight_limit=2400, started_at=NOW,
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
    assert "просит 540 ед/мин" in rejected
    assert accepted.startswith("Сессия watch_")


@pytest.mark.asyncio
async def test_c1_full_agg_trade_page_fetches_and_counts_next_page(monkeypatch, tmp_path):
    """С1, С5 №8: ответ из 1000 сделок немедленно дочитывается по fromId."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=60, depth_pcts=(0.25,), depth_weight=5, trade_weight=20,
            market_weight_limit=2400, started_at=NOW,
        )
    finally:
        con.close()

    class FakeAggTradesClient:
        calls: list[int | None] = []

        async def agg_trades(self, _symbol, *, limit, from_id):
            assert limit == 1000
            self.calls.append(from_id)
            if from_id is None:
                return [
                    {"a": index, "T": NOW + index, "p": "100", "q": "1", "m": False}
                    for index in range(1000)
                ]
            return [{"a": 1000, "T": NOW + 1000, "p": "100", "q": "1", "m": False}]

    client = FakeAggTradesClient()
    await _record_order_book_watch_trades(client, session, fetched_at=NOW + 5_000)

    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        stored = watch.get_watch(con, session["watch_id"])
        assert client.calls == [None, 1000]
        assert stored["trade_request_count"] == 2
        assert stored["trade_extra_page_count"] == 1
        assert con.execute("SELECT COUNT(*) FROM watch_trades").fetchone()[0] == 1001
    finally:
        con.close()


@pytest.mark.asyncio
async def test_c1_trade_polling_uses_client_clock_not_futures_matching_timestamp(
    monkeypatch, tmp_path
):
    """C1: джиттер `T` снимка на миллисекунду не пропускает следующий trade-poll."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=60, depth_pcts=(0.25,), depth_weight=5, trade_weight=20,
            market_weight_limit=2400, started_at=NOW,
        )
    finally:
        con.close()

    class FakeClient:
        async def order_book(self, _symbol, *, limit):
            assert limit == 100
            return {"T": NOW + 4_999, "bids": [["100", "10"]], "asks": [["100.2", "10"]]}

        async def now_ms(self):
            return NOW + 5_000

    async def fake_ctx(_market):
        return FakeClient(), None, None, None, FUTURES

    fetched: list[int] = []

    async def fake_trades(_client, _session, *, fetched_at):
        fetched.append(fetched_at)

    monkeypatch.setattr("cryptomcp.server._ctx", fake_ctx)
    monkeypatch.setattr("cryptomcp.server._record_order_book_watch_trades", fake_trades)

    assert await _record_order_book_watch_snapshot(session["watch_id"])
    assert fetched == [NOW + 5_000]


@pytest.mark.asyncio
async def test_c1_trade_polling_does_not_skip_a_tick_for_one_millisecond_jitter(
    monkeypatch, tmp_path
):
    """C1: разница 4.999 с — это плановый 5-секундный тик, а не новый 10-секундный."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=60, depth_pcts=(0.25,), depth_weight=5, trade_weight=20,
            market_weight_limit=2400, started_at=NOW,
        )
        watch.mark_trade_fetch(con, session["watch_id"], NOW)
        session = watch.get_watch(con, session["watch_id"])

        class FakeClient:
            calls: list[int | None] = []

            async def agg_trades(self, _symbol, *, limit, from_id):
                self.calls.append(from_id)
                return []

        client = FakeClient()
        await _record_order_book_watch_trades(client, session, fetched_at=NOW + 4_999)
        assert client.calls == [None]
    finally:
        con.close()


@pytest.mark.asyncio
async def test_boundary_snapshot_expires_watch_without_counting_an_error(monkeypatch, tmp_path):
    """Мелкое 12: ответ, пришедший после ends_at, штатно завершает сессию."""
    _, connect = _watch_db(monkeypatch, tmp_path)
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        session = watch.start(
            con, symbol="ARBUSDT", market="futures", interval_sec=5,
            duration_min=1, depth_pcts=(0.25,), depth_weight=5, trade_weight=20,
            market_weight_limit=2400, started_at=NOW,
        )
    finally:
        con.close()

    class FakeClient:
        async def order_book(self, _symbol, *, limit):
            assert limit == 100
            return {
                "T": NOW + 60_000,
                "bids": [["100", "10"]],
                "asks": [["100.2", "10"]],
            }

        async def now_ms(self):
            return NOW + 60_000

    async def fake_ctx(_market):
        return FakeClient(), None, None, None, FUTURES

    monkeypatch.setattr("cryptomcp.server._ctx", fake_ctx)

    assert not await _record_order_book_watch_snapshot(session["watch_id"])
    con = connect(str(tmp_path / "order_book_watch.sqlite"))
    try:
        stored = watch.get_watch(con, session["watch_id"])
        assert stored["status"] == "expired"
        assert stored["error_count"] == 0
        assert stored["snapshot_count"] == 0
    finally:
        con.close()


def test_watch_output_is_paginated_below_context_limit_for_every_format():
    """Блокер 1: текст, а не объект, остаётся читаемым на сотнях снимков."""
    levels = [
        {
            "price": 100 - index / 100,
            "qty": 1.0,
            "notional_usdt": 100.0,
            "cum_notional_usdt": (index + 1) * 100.0,
        }
        for index in range(100)
    ]
    snapshots = [
        {
            "ts": NOW + index * 5_000,
            "mid_price": 100.0,
            "best_bid": 99.99,
            "best_ask": 100.01,
            "spread_pct": 0.02,
            "bids": levels,
            "asks": levels,
        }
        for index in range(300)
    ]
    events = [
        {
            "ts": NOW + index * 5_000,
            "side": "bid",
            "price": 99.0,
            "qty_before": 1.0,
            "qty_after": 0.0,
            "event_type": "disappeared",
        }
        for index in range(300)
    ]
    session = {
        "watch_id": "watch_page",
        "symbol": "ARBUSDT",
        "market": "futures",
        "interval_sec": 5,
        "duration_min": 60,
        "started_at": NOW,
        "ends_at": NOW + 3_600_000,
        "status": "active",
        "last_snapshot_at": NOW,
    }

    for format in ("summary", "raw", "diff", "both"):
        text = render_order_book_watch_data(
            session,
            snapshots,
            events,
            format=format,
            now_ms=NOW,
            total_snapshots=300,
            total_events=300,
        )
        assert len(text) <= MAX_WATCH_RESPONSE_CHARS
        assert "снимков 300 · событий 300" in text
        if format in {"raw", "diff", "both"}:
            assert "показано " in text


def test_liquidity_summary_text_orders_real_outcomes_before_early_removal():
    """С4: модель получает вывод о настоящей ликвидности, не ярлык намерения."""
    level = {"price": 100.0, "qty": 10.0, "notional_usdt": 1000.0, "cum_notional_usdt": 1000.0}
    snapshot = {
        "ts": NOW,
        "mid_price": 100.0,
        "best_bid": 100.0,
        "best_ask": 100.1,
        "spread_pct": 0.1,
        "bids": [level],
        "asks": [{**level, "price": 100.1}],
    }
    after = {**snapshot, "ts": NOW + 5_000}
    watch_row = {
        "watch_id": "watch_text",
        "symbol": "ARBUSDT",
        "market": "futures",
        "interval_sec": 5,
        "duration_min": 60,
        "started_at": NOW,
        "ends_at": NOW + 3_600_000,
        "status": "active",
        "last_snapshot_at": NOW + 5_000,
        "trade_request_count": 1,
        "trade_extra_page_count": 0,
        "trade_gap_count": 0,
    }
    text = render_order_book_watch_data(
        watch_row,
        [snapshot, after],
        [],
        format="summary",
        now_ms=NOW + 5_000,
        trades=[{"ts": NOW + 1, "price": 100.0, "qty": 3.0, "buyer_is_maker": True}],
    )

    assert "устоял: 1 уровней" in text
    assert text.index("устоял:") < text.index("исполнен:") < text.index("снят заранее:")
    assert "спуфинг" not in text
