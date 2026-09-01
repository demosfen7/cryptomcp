"""Тесты бюджета веса (PLAN §6.4).

Проверяется главное свойство: вес резервируется ДО отправки, поэтому залп из
полусотни запросов не может проскочить лимит целиком и получить 429 постфактум.
"""

from __future__ import annotations

import asyncio

import pytest

from cryptomcp.ratelimit import WeightBudget, klines_weight


class FakeClock:
    """Управляемые часы: тест не должен ждать реальную минуту."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


def budget(clock, limit=100, safety=1.0):
    return WeightBudget(limit=limit, safety_factor=safety, clock=clock)


class TestKlinesWeight:
    """Вес зависит от limit, а не константа. Проверено на бирже: limit=1 → 1."""

    @pytest.mark.parametrize(
        "limit,expected",
        [(1, 1), (100, 1), (101, 2), (500, 2), (501, 5), (1000, 5), (1001, 10), (1500, 10)],
    )
    def test_weight_tiers(self, limit, expected):
        assert klines_weight(limit) == expected


class TestReservation:
    @pytest.mark.asyncio
    async def test_fits_within_budget_does_not_wait(self, clock):
        b = budget(clock)
        for _ in range(10):
            await b.reserve(10)
        assert b.used() == 100

    @pytest.mark.asyncio
    async def test_safety_factor_reduces_usable(self, clock):
        b = budget(clock, limit=100, safety=0.8)
        assert b.usable == 80

    @pytest.mark.asyncio
    async def test_weight_larger_than_budget_is_rejected(self, clock):
        b = budget(clock, limit=10)
        with pytest.raises(ValueError, match="превышает доступный бюджет"):
            await b.reserve(11)

    @pytest.mark.asyncio
    async def test_waits_until_window_frees_up(self, clock, monkeypatch):
        """Переполнение окна приводит к ожиданию, а не к отправке запроса."""
        b = budget(clock, limit=100)
        await b.reserve(100)

        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)
            clock.advance(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await b.reserve(50)

        assert slept, "запрос должен был подождать освобождения окна"
        assert slept[0] == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_old_events_leave_the_window(self, clock):
        b = budget(clock, limit=100)
        await b.reserve(100)
        clock.advance(61)
        assert b.used() == 0
        await b.reserve(100)  # не должно заблокироваться


class TestExchangeReconciliation:
    """Сервер публичный: лимит расходуется не только нами (PLAN §6.4, §7.3)."""

    def test_exchange_report_overrides_local_estimate(self, clock):
        b = budget(clock)
        b.observe_header(500)
        assert b.used() == 500

    def test_stale_exchange_report_is_ignored(self, clock):
        b = budget(clock)
        b.observe_header(500)
        clock.advance(61)
        assert b.used() == 0

    def test_local_estimate_wins_when_higher(self, clock):
        b = budget(clock, limit=1000)
        b._events.append((clock(), 700))
        b.observe_header(100)
        assert b.used() == 700

    def test_missing_header_is_noop(self, clock):
        b = budget(clock)
        b.observe_header(None)
        assert b.used() == 0

    @pytest.mark.asyncio
    async def test_waits_when_exchange_says_we_are_full(self, clock, monkeypatch):
        b = budget(clock, limit=100)
        b.observe_header(100)

        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)
            clock.advance(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        await b.reserve(10)

        assert slept and slept[0] == pytest.approx(60.0)
