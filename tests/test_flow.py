"""Поток тейкеров, поглощение и расширение объёма (SPEC-flow-and-absorption-v2).

Приёмочные тесты ТЗ прогонялись на живых данных через `as_of_ms`; результат
записан в docstring TestSpecCases. Здесь — синтетика, которая держит поведение
при правках.
"""

from __future__ import annotations

import pytest

from cryptomcp import flow as F
from cryptomcp.series import INTERVAL_MS, build_series


def series(bars, interval="1d"):
    """bars: список (open, close, quote_volume, taker_buy_quote)."""
    step = INTERVAL_MS[interval]
    raw = [
        [
            i * step, f"{o}", f"{max(o, c) + 0.1}", f"{min(o, c) - 0.1}", f"{c}",
            "10.0", (i + 1) * step - 1, f"{q}", 10, "5.0", f"{buy}", "0",
        ]
        for i, (o, c, q, buy) in enumerate(bars)
    ]
    return build_series(raw, "TESTUSDT", interval, len(bars) * step + 10_000,
                        grace_ms=0)


def flat(n, price=100.0, volume=1000.0, buy_share=0.5):
    return [(price, price, volume, volume * buy_share)] * n


class TestDelta:
    """Дельта — это 2×куплено − оборот, и ничего больше."""

    def test_neutral_flow_is_zero(self):
        data = F.flow(series(flat(30)), 30)
        assert data.delta_sum == pytest.approx(0.0)
        assert data.delta_share == pytest.approx(0.0)

    def test_selling_pressure_is_negative(self):
        data = F.flow(series(flat(30, buy_share=0.4)), 30)
        assert data.delta_sum < 0
        assert data.delta_share == pytest.approx(-20.0)

    def test_share_is_normalised_by_turnover(self):
        """Дельта −1.14M на обороте 45.68M и та же дельта на обороте 3M —
        разные утверждения, поэтому доля печатается рядом."""
        thin = F.flow(series(flat(30, volume=100.0, buy_share=0.4)), 30)
        thick = F.flow(series(flat(30, volume=10_000.0, buy_share=0.4)), 30)
        assert thin.delta_sum > thick.delta_sum  # обе отрицательны, тонкая ближе к нулю
        assert thin.delta_share == pytest.approx(thick.delta_share)

    def test_short_series_gives_nothing(self):
        assert F.flow(series(flat(10)), 30) is None


class TestQuadrants:
    """Четыре сочетания знаков (§1.3). Без хода цены метрика бессмысленна."""

    def build(self, *, buy_share, last_price):
        bars = flat(29, buy_share=buy_share)
        bars.append((100.0, last_price, 1000.0, 1000.0 * buy_share))
        return F.flow(series(bars), 30)

    def test_absorption_is_selling_into_a_rise(self):
        data = self.build(buy_share=0.4, last_price=120.0)
        assert data.quadrant == F.ABSORPTION
        assert data.flagged

    def test_ordinary_demand(self):
        assert self.build(buy_share=0.6, last_price=120.0).quadrant == F.DEMAND

    def test_ordinary_exit(self):
        assert self.build(buy_share=0.4, last_price=80.0).quadrant == F.EXIT

    def test_distribution_is_buying_into_a_fall(self):
        data = self.build(buy_share=0.6, last_price=80.0)
        assert data.quadrant == F.DISTRIBUTION
        assert not data.flagged

    def test_same_buy_share_different_verdict(self):
        """AGLD и ANKR имеют долю покупок 48.8% обе; разводит их только цена."""
        up = self.build(buy_share=0.45, last_price=120.0)
        down = self.build(buy_share=0.45, last_price=80.0)
        assert up.delta_share == pytest.approx(down.delta_share, abs=1.0)
        assert up.quadrant != down.quadrant


class TestAbsorptionRatio:
    """Знак дельты не отвечает, дорого ли поток обошёлся рынку (§2)."""

    def build(self, *, buy_share, last_price):
        bars = flat(29, buy_share=buy_share)
        bars.append((100.0, last_price, 1000.0, 1000.0 * buy_share))
        return F.flow(series(bars), 30)

    def test_negative_means_price_went_against_the_flow(self):
        assert self.build(buy_share=0.4, last_price=120.0).absorption_ratio < 0

    def test_noise_is_na_with_a_reason(self):
        """|доля| ниже 0.5% — деление малого на малое, метрика взорвалась бы."""
        data = self.build(buy_share=0.5, last_price=120.0)
        assert data.absorption_ratio is None
        assert "в пределах шума" in data.ratio_note

    def test_flat_price_is_printed_but_marked(self):
        data = self.build(buy_share=0.4, last_price=100.2)
        assert data.absorption_ratio is not None
        assert "цена стоит" in data.ratio_note

    def test_never_divides_by_zero(self):
        assert self.build(buy_share=0.5, last_price=100.0).absorption_ratio is None


class TestVolRatio:
    """Окна фиксированные и непересекающиеся; неполный знаменатель запрещён."""

    def test_measures_recent_against_previous(self):
        bars = flat(30, volume=100.0) + flat(12, volume=200.0)
        assert F.vol_ratio(series(bars)).value == pytest.approx(2.0)

    def test_collapse_is_visible(self):
        """ROBOUSDT: колонка MA20/MA100 давала 1.92, факт — 0.58x."""
        bars = flat(30, volume=100.0) + flat(12, volume=58.0)
        assert F.vol_ratio(series(bars)).value == pytest.approx(0.58)

    def test_incomplete_base_is_na_not_partial(self):
        """Считать по тому, что есть, запрещено: это и даёт неполный
        знаменатель, из-за которого колонка МА непригодна."""
        data = F.vol_ratio(series(flat(20)))
        assert data.value is None
        assert "нужно 42" in data.reason


class TestAbsorptionEvent:
    """Одиночный бар как событие — кодифицированное исключение (§5)."""

    def build(self, *, low=97.0, close=100.0, volume=9000.0):
        step = INTERVAL_MS["1h"]
        raw = [
            [i * step, "100.0", "100.2", "99.9", "100.0", "10.0",
             (i + 1) * step - 1, "1000.0", 10, "5.0", "500.0", "0"]
            for i in range(80)
        ]
        raw.append([
            80 * step, "100.0", "100.2", f"{low}", f"{close}", "10.0",
            81 * step - 1, f"{volume}", 10, "5.0", f"{volume * 0.38}", "0",
        ])
        s = build_series(raw, "TESTUSDT", "1h", 81 * step + 10_000, grace_ms=0)
        return F.absorption_events(s, window=72)

    def test_long_lower_wick_on_volume_is_an_event(self):
        assert len(self.build()) == 1

    def test_volume_alone_is_not_enough(self):
        assert self.build(low=99.9, close=100.0) == ()

    def test_big_body_is_a_stop_hunt_not_absorption(self):
        """У выноса тело большое и закрытие у края диапазона."""
        assert self.build(low=97.0, close=97.1) == ()

    def test_taker_side_does_not_veto(self):
        """В эталонном баре takerB 0.38 — агрессором был продавец, а
        покупатель стоял лимитом. Метрика указывала бы в другую сторону."""
        events = self.build()
        assert events and events[0].side == "buy"

    def test_mirror_is_the_other_wick(self):
        """Бар раздачи — тот же порог на верхнем фитиле, та же процедура."""
        step = INTERVAL_MS["1h"]
        raw = [
            [i * step, "100.0", "100.2", "99.9", "100.0", "10.0",
             (i + 1) * step - 1, "1000.0", 10, "5.0", "500.0", "0"]
            for i in range(80)
        ]
        # Длинный ВЕРХНИЙ фитиль: цена уходила на 103 и вернулась.
        raw.append([
            80 * step, "100.0", "103.0", "99.9", "100.0", "10.0",
            81 * step - 1, "9000.0", 10, "5.0", "3420.0", "0",
        ])
        s = build_series(raw, "TESTUSDT", "1h", 81 * step + 10_000, grace_ms=0)

        assert F.absorption_events(s, window=72, side="sell") != ()
        assert F.absorption_events(s, window=72, side="buy") == ()


class TestFlagNeedsThreeThings:
    """Флаг ⚑ — утверждение, и оно требует всех трёх условий."""

    def build(self, *, buy_share, last_price, narrow_bars):
        from cryptomcp.render import render_flow

        bars = flat(29, buy_share=buy_share)
        bars.append((100.0, last_price, 1000.0, 1000.0 * buy_share))
        data = F.flow(series(bars), 30)
        return render_flow([data], narrow_bars=narrow_bars)

    def test_flagged_when_squeezed_and_flow_is_real(self):
        assert "⚑" in self.build(
            buy_share=0.4, last_price=120.0, narrow_bars=11
        )

    def test_no_flag_without_squeeze(self):
        """Вне сжатия первый квадрант стоял все четыре дня роста ASTER."""
        text = self.build(buy_share=0.4, last_price=120.0, narrow_bars=0)
        assert "⚑" not in text
        assert "узк 0" in text

    def test_no_flag_when_flow_is_noise(self):
        """HYPERUSDT спот 15.09.2026: дельта −0.1% оборота, флаг стоял.

        Полосу |доля| < 0.5% ТЗ само называет шумом в §2 и отказывается
        считать на ней absorption_ratio. Вешать на неё вердикт — утверждать
        больше, чем измерено.
        """
        text = self.build(buy_share=0.499, last_price=120.0, narrow_bars=11)
        assert "⚑" not in text
        assert "в пределах шума" in text


class TestSpecCases:
    """Приёмочные тесты ТЗ на живых данных, 15.09.2026, перпетуал, `as_of_ms`.

    | # | ждали | получили |
    |---|-------|----------|
    | 1 | AGLD 1d: 45.68M / 22.27M / −1.14M / +12.7% / −5.1 | **совпало на окне 34** |
    | 2 | ANKR 1d окно 10: −162K, ratio +2.6 | **−165K, ratio +2.5, всё прочее до знака** |
    | 3 | ANKR 1h 16.08 18:00: фитиль 88%, тело 4% | **найдено, фитиль 88%, тело 4%** |
    | 4 | ANKR 1d: событий нет | **0 событий** |
    | 7 | ENSO vol_ratio = 1.13x | **1.13x** |
    | 8 | ROBO vol_ratio = 0.58x | **0.58x** |

    Два расхождения, оба в ТЗ:

    - **тест 1 просит окно 30, а его же числа получены на 34.** §1.3 называет
      диапазон 01.08–03.09.2026 и «34 свечи», и на 34 всё сходится до знака;
      на 30 выходит 39.45M / 19.15M / −1.16M / +12.4% / −4.2.
    - **объём эталонного бара 7.35x, а не 9.53x.** ТЗ считает от базы «~12K»,
      проект — от сезонной базы, которую печатает `get_klines` (тест 11
      требует именно её). Событие находится при обеих: порог 5x.
    """

    def test_this_is_documentation(self):
        """Живые кейсы требуют сети; здесь зафиксирован их результат."""
