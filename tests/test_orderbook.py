"""Приёмка части A стакана: расчёт, частичный охват и изоляция рынков."""

from __future__ import annotations

from cryptomcp.markets import FUTURES, SPOT
from cryptomcp.orderbook import build_order_book, normalise_depth_pcts
from cryptomcp.render import render_order_book

NOW = 1_789_870_000_000


def _snapshot(*, bid_shift: float = 0.0, ask_shift: float = 0.0):
    return {
        "lastUpdateId": 42,
        "bids": [[str(100.0 + bid_shift), "10"], [str(99.8 + bid_shift), "20"]],
        "asks": [[str(100.2 + ask_shift), "15"], [str(100.4 + ask_shift), "25"]],
    }


def _book(*, depth_pcts=(0.25, 5.0), turnover=5_000_000.0, snapshot=None):
    return build_order_book(
        snapshot or _snapshot(),
        timestamp_ms=NOW,
        last_price=100.1,
        limit=100,
        depth_pcts=depth_pcts,
        turnover_24h_usdt=turnover,
    )


def test_a9_2_incomplete_depth_coverage_is_printed_with_reason():
    """A.9 №2: limit не расширяется молча, когда ±5% за ним не помещается."""
    text = render_order_book(_book(), "THINUSDT", precision=2)

    assert "Диапазон ±5%" in text
    assert "покрытие неполное" in text
    assert "дальше уровней в ответе Binance нет" in text


def test_a9_3_cumulative_notional_matches_manual_sum_inside_depth():
    """A.9 №3: по сырому блоку вручную проверяется любая производная величина."""
    book = _book(depth_pcts=(0.25,))
    threshold = book.mid_price * (1 - 0.25 / 100)
    levels = [level for level in book.bids if level.price >= threshold]
    manually_summed = sum(level.notional_usdt for level in levels)

    assert levels[-1].cum_notional_usdt == manually_summed
    assert book.ranges[0].bid_notional_usdt == manually_summed


def test_a9_4_low_turnover_makes_imbalance_na_but_keeps_raw_book():
    """A.9 №4: тонкий рынок не теряет уровни, только неоправданный imbalance."""
    text = render_order_book(_book(turnover=1_999_999.0), "THINUSDT", precision=2)

    assert "imbalance n/a — оборот 24ч" in text
    assert "Сырые уровни bids" in text
    assert "Сырые уровни asks" in text


def test_a9_5_walls_and_clusters_are_na_without_a_historical_baseline():
    """A.9 №5: медиана одного снимка не подменяет историю размеров уровней."""
    text = render_order_book(_book(), "BTCUSDT", precision=2)

    assert "Крупные уровни: n/a — нет базы" in text
    assert "Кластеры: n/a — нет базы" in text


def test_a9_6_spot_and_futures_render_as_independent_books():
    """A.9 №6: рынок входит в снимок, поэтому цены книг не смешиваются."""
    futures = _book(snapshot=_snapshot())
    spot = _book(snapshot=_snapshot(bid_shift=-1.0, ask_shift=-1.0))

    futures_text = render_order_book(futures, "ARBUSDT", market=FUTURES, precision=2)
    spot_text = render_order_book(spot, "ARBUSDT", market=SPOT, precision=2)

    assert "ARBUSDT.P (USDⓈ-M perp)" in futures_text
    assert "ARBUSDT (спот)" in spot_text
    assert futures.best_bid != spot.best_bid


def test_depth_pct_accepts_scalar_and_list_without_losing_order():
    """A.2: один диапазон и список ведут в ту же ветку без неявного округления."""
    assert normalise_depth_pcts(0.5) == (0.5,)
    assert normalise_depth_pcts([0.5, 1, 0.5]) == (0.5, 1.0)
