"""Клиент L2 использует измеренные, а не документированные веса."""

from __future__ import annotations

import json

import httpx
import pytest

from cryptomcp.client import BinanceClient
from cryptomcp.errors import ErrorKind, ToolError
from cryptomcp.markets import FUTURES, SPOT


@pytest.mark.parametrize(
    ("market", "limit", "expected_weight"),
    [
        (FUTURES, 50, 2),
        (FUTURES, 100, 5),
        (FUTURES, 500, 10),
        (FUTURES, 1000, 20),
        (SPOT, 50, 5),
        (SPOT, 500, 25),
        (SPOT, 1000, 50),
        (SPOT, 5000, 250),
    ],
)
def test_depth_weight_uses_measurement_for_each_market(market, limit, expected_weight):
    """Таблица A.3 относится к споту: futures 50 и 100 расходятся по весу."""
    assert market.depth_weight(limit) == expected_weight


def test_futures_depth_rejects_spot_only_limit_5000():
    """Не отправляем Binance futures-параметр, который он отвергнет с -1130."""
    assert 5000 not in FUTURES.depth_limits
    assert 5000 in SPOT.depth_limits


@pytest.mark.asyncio
async def test_order_book_reserves_measured_depth_weight_before_request():
    """Неверный резерв позволил бы сессиям исчерпать общий IP-бюджет внезапно."""
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"lastUpdateId": 1, "bids": [], "asks": []})

    client = BinanceClient(FUTURES, transport=httpx.MockTransport(handler))
    try:
        await client.order_book("BTCUSDT", limit=100)
        assert client.budget.snapshot()["used"] == 5
        assert seen[0].url.path == "/fapi/v1/depth"
        assert seen[0].url.params["limit"] == "100"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_order_book_rejects_invalid_limit_without_network():
    """A.9 №1: enum проверяется локально, а не округляется и не уходит в сеть."""
    client = BinanceClient(FUTURES, transport=httpx.MockTransport(lambda _: None))
    try:
        with pytest.raises(ToolError) as caught:
            await client.order_book("BTCUSDT", limit=30)
        assert caught.value.kind is ErrorKind.BAD_PARAMS
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ticker_price_uses_separate_measured_weight():
    """Живой тикер не подменяется 24h-тикером: он нужен с отдельной пометкой."""
    request = httpx.Request("GET", "https://example.test/api/v3/ticker/price")
    response = httpx.Response(200, text=json.dumps({"price": "1.25"}), request=request)
    client = BinanceClient(SPOT, transport=httpx.MockTransport(lambda _: response))
    try:
        assert (await client.ticker_price("ARBUSDT"))["price"] == "1.25"
        assert client.budget.snapshot()["used"] == 2
    finally:
        await client.aclose()
