"""Тесты фандинга и открытого интереса (PLAN §4.12, §6.8)."""

from __future__ import annotations

import pytest

from cryptomcp.derivatives import (
    DEFAULT_OI_THRESHOLD,
    Funding,
    OpenInterest,
    classify_price_oi,
)


def funding(rate: float, interval_hours: int) -> Funding:
    return Funding(
        symbol="TESTUSDT",
        rate=rate,
        interval_hours=interval_hours,
        next_funding_ms=0,
        mark_price=100.0,
        index_price=100.0,
    )


class TestFundingNormalisation:
    """Без приведения к годовым сравнение между монетами бессмысленно."""

    def test_same_rate_different_intervals_gives_different_annual(self):
        rate = 0.0001  # 0.01% за интервал
        hourly = funding(rate, 1).annualized_pct
        four_hourly = funding(rate, 4).annualized_pct
        eight_hourly = funding(rate, 8).annualized_pct

        assert hourly == pytest.approx(87.6)
        assert four_hourly == pytest.approx(21.9)
        assert eight_hourly == pytest.approx(10.95)
        # Ровно та восьмикратная разница, ради которой нормализация и делается.
        assert hourly == pytest.approx(eight_hourly * 8)

    def test_negative_rate_annualises_negative(self):
        assert funding(-0.0001, 8).annualized_pct == pytest.approx(-10.95)

    def test_zero_rate(self):
        assert funding(0.0, 8).annualized_pct == pytest.approx(0.0)

    def test_basis_from_mark_and_index(self):
        f = Funding("T", 0.0, 8, 0, mark_price=101.0, index_price=100.0)
        assert f.basis_pct == pytest.approx(1.0)

    def test_payload_carries_interval(self):
        payload = funding(0.0001, 4).to_dict()
        assert payload["interval_hours"] == 4
        assert payload["annualized_pct"] == pytest.approx(21.9)


class TestQuadrantClassification:
    """Четыре квадранта «цена × OI» (PLAN §4.12)."""

    def test_price_up_oi_up_is_new_money(self):
        assert classify_price_oi(0.05, 0.05) == "приток новых денег"

    def test_price_up_oi_down_is_short_covering(self):
        assert classify_price_oi(0.05, -0.05) == "закрытие шортов"

    def test_price_down_oi_up_is_new_shorts(self):
        assert classify_price_oi(-0.05, 0.05) == "новые шорты"

    def test_price_down_oi_down_is_long_unwinding(self):
        assert classify_price_oi(-0.05, -0.05) == "закрытие лонгов"

    def test_below_threshold_is_no_signal(self):
        assert classify_price_oi(0.0005, 0.0005) == "без выраженного потока"

    def test_oi_grows_while_price_stands_is_accumulation(self):
        """Регрессия по живым данным SUIUSDT: OI +3.43% при цене −0.81%.

        Раньше это читалось как «без выраженного потока», хотя набор позиций
        при стоящей цене — это ровно то, что ТЗ §4.2 называет набором позиции
        без движения цены, главный признак накопления.
        """
        assert classify_price_oi(-0.008, 0.034) == "набор позиций без движения цены"
        assert classify_price_oi(0.0001, 0.05) == "набор позиций без движения цены"

    def test_oi_falls_while_price_stands_is_unwinding(self):
        """Знак изменения OI обязан различаться.

        Регрессия по боевой выдаче SUIUSDT: OI −1.56% при цене +0.53%
        подписывалось как «набор позиций», хотя падение открытого интереса —
        это закрытие позиций, ровно обратное по смыслу.
        """
        assert classify_price_oi(0.0053, -0.0156) == "разгрузка позиций без движения цены"
        assert classify_price_oi(-0.0001, -0.05) == "разгрузка позиций без движения цены"

    def test_accumulation_and_unwinding_are_distinct(self):
        rising = classify_price_oi(0.001, 0.05)
        falling = classify_price_oi(0.001, -0.05)
        assert rising != falling

    def test_price_moves_without_oi_is_rotation(self):
        assert classify_price_oi(0.05, 0.0001) == "движение без притока (ротация)"

    def test_threshold_is_configurable(self):
        assert classify_price_oi(0.02, 0.02, threshold=0.05) == "без выраженного потока"
        assert classify_price_oi(0.02, 0.02, threshold=0.01) == "приток новых денег"

    def test_default_threshold_is_one_percent(self):
        assert DEFAULT_OI_THRESHOLD == 0.01


class TestOpenInterestReading:
    def test_windows_render_with_reading(self):
        oi = OpenInterest(
            symbol="TESTUSDT",
            contracts=1000.0,
            notional_usdt=100_000.0,
            change={"1h": 0.03, "24h": -0.04},
            price_change={"1h": 0.02, "24h": 0.05},
        )
        payload = oi.to_dict()

        assert payload["windows"]["1h"]["reading"] == "приток новых денег"
        assert payload["windows"]["24h"]["reading"] == "закрытие шортов"
        assert payload["windows"]["1h"]["oi_change_pct"] == pytest.approx(3.0)

    def test_missing_window_is_neutral(self):
        oi = OpenInterest("T", 1.0, 1.0, {}, {})
        assert oi.quadrant("1h") == "без выраженного потока"


class TestContractsNotNotional:
    """Динамика считается по контрактам, а не по стоимости в USDT.

    При росте цены на 5% стоимость OI вырастет на те же 5%, даже если ни одна
    позиция не открылась. Классификация по стоимости показывала бы приток новых
    денег там, где его нет.
    """

    def test_price_rise_alone_must_not_read_as_inflow(self):
        contracts_before, contracts_after = 1000.0, 1000.0
        price_before, price_after = 100.0, 105.0

        oi_change_by_contracts = (contracts_after - contracts_before) / contracts_before
        notional_before = contracts_before * price_before
        notional_after = contracts_after * price_after
        oi_change_by_notional = (notional_after - notional_before) / notional_before
        price_change = (price_after - price_before) / price_before

        # По контрактам: позиции не открывались, движение — ротация между
        # уже существующими держателями.
        assert classify_price_oi(price_change, oi_change_by_contracts) == (
            "движение без притока (ротация)"
        )
        # По стоимости: OI «вырос» ровно на величину роста цены, и это было бы
        # прочитано как приход новых денег. Именно эта ошибка и исключается.
        assert oi_change_by_notional == pytest.approx(price_change)
        assert classify_price_oi(price_change, oi_change_by_notional) == (
            "приток новых денег"
        )


class TestRetrospective:
    """Ретроспектива деривативов: один сборщик на биржу и архив."""

    NOW = 1_787_184_000_000
    STEP_5M = 5 * 60_000

    def rows(self, count=300, start_oi=1000.0, price=0.6):
        return [
            (self.NOW - (count - 1 - i) * self.STEP_5M,
             start_oi + i, (start_oi + i) * price)
            for i in range(count)
        ]

    def test_same_arithmetic_for_both_sources(self):
        """Биржа и архив обязаны давать одинаковые числа на одном ряду."""
        from cryptomcp.derivatives import _oi_rows, build_open_interest

        rows = self.rows()
        raw = [
            {"timestamp": ts, "sumOpenInterest": str(oi),
             "sumOpenInterestValue": str(value)}
            for ts, oi, value in rows
        ]
        from_archive = build_open_interest(
            "AAAUSDT", rows, 5, rows[::12], 60,
            windows=("1h", "4h"), source="архив",
        )
        from_exchange = build_open_interest(
            "AAAUSDT", _oi_rows(raw), 5, _oi_rows(raw[::12]), 60,
            windows=("1h", "4h"),
        )
        assert from_archive.change == from_exchange.change
        assert from_archive.price_change == from_exchange.price_change
        assert from_archive.contracts == from_exchange.contracts
        assert from_archive.source == "архив"

    def test_contracts_taken_from_series_when_not_given(self):
        """В прошлом «текущего» OI не существует — берётся последняя точка ряда."""
        from cryptomcp.derivatives import build_open_interest

        rows = self.rows()
        view = build_open_interest("AAAUSDT", rows, 5, rows, 5, windows=("1h",))
        assert view.contracts == rows[-1][1]

    def test_empty_series_is_not_a_crash(self):
        from cryptomcp.derivatives import build_open_interest

        view = build_open_interest("AAAUSDT", [], 5, [], 60, windows=("1h",))
        assert view.change == {} and view.contracts == 0.0

    def test_funding_rate_defaults_to_last_settlement(self):
        """В ретроспективе ставка — последнее начисление, а не текущая из premium."""
        from cryptomcp.derivatives import build_funding

        settlements = [(self.NOW - i * 4 * 3_600_000, 0.0001 * (10 - i))
                       for i in range(10, 0, -1)]
        funding = build_funding("AAAUSDT", settlements, 4, source="архив")

        assert funding.rate == settlements[-1][1]
        assert funding.source == "архив"
        # Базиса в прошлом нет: маркировочная и индексная цены только «сейчас».
        assert funding.mark_price == 0.0
        assert funding.basis_pct != funding.basis_pct  # NaN


class TestSideOfFlow:
    """Сторона набора: без неё сигнал читается ровно наоборот.

    Случай HOMEUSDT 02.09.2026: OI +7.05% за сутки при цене +0.47%, то есть
    «набор позиций без движения цены». Но фандинг -303% годовых — набиралась
    короткая сторона.
    """

    def test_negative_funding_means_shorts(self):
        from cryptomcp.derivatives import side_of_flow

        assert side_of_flow("набор позиций без движения цены", -303.0) == (
            "набор ШОРТОВ без движения цены"
        )

    def test_positive_funding_means_longs(self):
        from cryptomcp.derivatives import side_of_flow

        assert side_of_flow("набор позиций без движения цены", +120.0) == (
            "набор ЛОНГОВ без движения цены"
        )

    def test_neutral_band_admits_it_knows_nothing(self):
        """Полоса ±20% накрывает 77% рынка: там фандинг о стороне молчит."""
        from cryptomcp.derivatives import side_of_flow

        for rate in (8.5, -0.7, 11.0, None):
            assert "без явной стороны" in side_of_flow(
                "набор позиций без движения цены", rate
            )

    def test_other_quadrants_untouched(self):
        """Где сторона видна по цене, подписывать её заново незачем."""
        from cryptomcp.derivatives import side_of_flow

        for quadrant in ("приток новых денег", "закрытие шортов",
                         "движение без притока (ротация)",
                         "разгрузка позиций без движения цены"):
            assert side_of_flow(quadrant, -303.0) == quadrant


class TestStructuralFunding:
    NOW = 1_788_400_000_000
    STEP = 4 * 3_600_000

    def settlements(self, rate, count=200):
        return [(self.NOW - (count - 1 - i) * self.STEP, rate) for i in range(count)]

    def test_deep_negative_is_flagged_as_normal_for_the_coin(self):
        from cryptomcp.derivatives import build_funding

        # Ставка хранится долей: -0.001333 это -0.1333% за период.
        funding = build_funding("HOMEUSDT", self.settlements(-0.001333), 4)
        assert funding.structurally_negative
        assert funding.median_annual_pct == pytest.approx(-291.9, abs=1.0)

    def test_ordinary_funding_is_not_flagged(self):
        from cryptomcp.derivatives import build_funding

        funding = build_funding("BTCUSDT", self.settlements(0.0001), 8)
        assert not funding.structurally_negative

    def test_median_window_is_time_not_count(self):
        """Интервал начисления разный: «последние 90 начислений» — не 30 суток."""
        from cryptomcp.derivatives import build_funding

        old = [(self.NOW - 200 * 86_400_000 + i * self.STEP, -0.5)
               for i in range(100)]
        recent = self.settlements(0.0001, count=180)
        funding = build_funding("AAAUSDT", old + recent, 4)

        assert not funding.structurally_negative
