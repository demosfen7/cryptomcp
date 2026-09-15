"""Детектор распределения и его зеркало (SPEC-distribution-detector).

Приёмочные тесты ТЗ прогонялись отдельно, на живых исторических данных через
`as_of_ms`; их результат записан в docstring класса TestSpecCases. Здесь —
синтетика, которая держит поведение при правках: живой прогон требует сети и
в CI не воспроизводится.
"""

from __future__ import annotations

import numpy as np

from cryptomcp import distribution as D
from cryptomcp.series import INTERVAL_MS, build_series

DAY = INTERVAL_MS["1d"]


def series(bars, interval="1d"):
    """bars: список (high, low, open, close, quote_volume)."""
    step = INTERVAL_MS[interval]
    raw = [
        [
            i * step, f"{o}", f"{h}", f"{low}", f"{c}", "10.0",
            (i + 1) * step - 1, f"{q}", 10, "5.0", f"{q / 2}", "0",
        ]
        for i, (h, low, o, c, q) in enumerate(bars)
    ]
    return build_series(raw, "TESTUSDT", interval, len(bars) * step + 10_000,
                        grace_ms=0)


def quiet(price=100.0, volume=1e6):
    """Свеча без длинных фитилей и без всплеска — фон, на котором считается база.

    Открытие и закрытие РАЗНЕСЕНЫ: у свечи с open == close обе доли фитиля
    ровно по 0.5, то есть она проходит порог с обеих сторон, и фон перестал бы
    быть фоном.
    """
    return (price + 0.2, price - 0.2, price - 0.1, price + 0.1, volume)


def upper(top, body=100.0, volume=4e6):
    """Свеча с верхним фитилём больше половины диапазона."""
    return (top, body - 0.1, body, body, volume)


def lower(bottom, body=100.0, volume=4e6):
    """Зеркальная: нижний фитиль больше половины диапазона."""
    return (body + 0.1, bottom, body, body, volume)


def atrs(n, value=1.0):
    return np.full(n, value)


class TestBar:
    """Свеча становится баром по двум условиям и ни по одному больше."""

    def test_volume_alone_is_not_enough(self):
        bars = [quiet()] * 40 + [(100.2, 99.8, 100.0, 100.15, 9e6)]
        result = D.detect(series(bars), atrs(41), min_turnover=0.0)
        assert result.count == 0

    def test_wick_alone_is_not_enough(self):
        bars = [quiet()] * 40 + [upper(105.0, volume=1.1e6)]
        result = D.detect(series(bars), atrs(41), min_turnover=0.0)
        assert result.count == 0

    def test_both_together_make_a_bar(self):
        bars = [quiet()] * 40 + [upper(105.0)]
        result = D.detect(series(bars), atrs(41), min_turnover=0.0)
        assert result.count == 1

    def test_zero_range_candle_is_not_a_bar(self):
        """Деление не выполняется: у свечи с high == low доли фитиля нет."""
        bars = [quiet()] * 40 + [(100.0, 100.0, 100.0, 100.0, 9e6)]
        result = D.detect(series(bars), atrs(41), min_turnover=0.0)
        assert result.count == 0

    def test_climax_is_filtered_by_the_wick(self):
        """ORN 18.06.2024: объём 5.86x, максимум движения, тело −7.73%, но
        верхний фитиль 19% — свеча открылась у максимума и падала весь день.
        Порог фитиля отсекает её без специального условия."""
        bars = [quiet()] * 40 + [(100.5, 92.0, 100.3, 92.3, 9e6)]
        result = D.detect(series(bars), atrs(41), min_turnover=0.0)
        assert result.count == 0


class TestCollapse:
    """Схлопывание в событие: один спайк и его отдача — одно наблюдение."""

    def test_consecutive_bars_are_one_event(self):
        bars = [quiet()] * 40 + [upper(105.0), upper(104.0)]
        result = D.detect(series(bars), atrs(42), min_turnover=0.0)
        assert result.count == 1
        assert result.events[0].bars == 2

    def test_through_one_candle_is_still_one_event(self):
        """Правило §1.2: у OAX так стоят пары 25–26.07 и 09–10.09."""
        bars = [quiet()] * 40 + [upper(105.0), quiet(), upper(104.0)]
        result = D.detect(series(bars), atrs(43), min_turnover=0.0)
        assert result.count == 1

    def test_two_candles_apart_are_two_events(self):
        bars = [quiet()] * 40 + [upper(105.0), quiet(), quiet(), upper(104.0)]
        result = D.detect(series(bars), atrs(44), min_turnover=0.0)
        assert result.count == 2

    def test_event_keeps_the_highest_extreme_and_the_first_date(self):
        bars = [quiet()] * 40 + [upper(104.0), upper(107.0)]
        event = D.detect(series(bars), atrs(42), min_turnover=0.0).events[0]
        assert event.extreme == 107.0
        assert event.ts_ms == 40 * DAY


class TestVerdict:
    """Три исхода §2.1 и то, что их разделяет."""

    def falling(self, tops, atr_value=1.0):
        bars = [quiet()] * 40
        for top in tops:
            bars += [upper(top), quiet(), quiet()]
        return D.detect(series(bars), atrs(len(bars), atr_value),
                        min_turnover=0.0)

    def test_two_events_are_never_a_verdict(self):
        assert self.falling([110.0, 105.0]).verdict == D.VERDICT_NONE

    def test_rising_maxima_are_none_whatever_the_count(self):
        """Единственное, что отделяет распределение от отката после пробоя.

        Эталон ORN 01–03.06.2024: два бара проходят по объёму и фитилю, но
        максимумы растут 1.794 → 1.810. Дальше цена прошла +19% за две недели,
        и ложное срабатывание стоило бы дороже пропуска.
        """
        assert self.falling([105.0, 107.0, 109.0, 111.0]).verdict == (
            D.VERDICT_NONE
        )

    def test_small_shift_is_forming(self):
        """Три события с плоскими максимумами — не снятие с наблюдения.

        Эталон OAX на 29.05.2024: наклон слегка отрицательный, смещение много
        меньше 1 ATR, монета ещё ходила в диапазоне.
        """
        result = self.falling([110.0, 109.6, 109.2], atr_value=10.0)
        assert result.verdict == D.VERDICT_FORMING
        assert result.shift_atr < D.SHIFT_ATR

    def test_shift_beyond_one_atr_is_confirmed(self):
        result = self.falling([110.0, 106.0, 102.0], atr_value=1.0)
        assert result.verdict == D.VERDICT_CONFIRMED
        assert result.shift_atr >= D.SHIFT_ATR

    def test_shift_is_measured_in_atr_not_percent(self):
        """§2.2: масштаб смещения при сжатом ATR становится строже
        пропорционально волатильности — это нормировка, а не баг."""
        wide = self.falling([110.0, 106.0, 102.0], atr_value=20.0)
        tight = self.falling([110.0, 106.0, 102.0], atr_value=1.0)
        assert wide.verdict == D.VERDICT_FORMING
        assert tight.verdict == D.VERDICT_CONFIRMED


class TestMirror:
    """Одна процедура с параметром стороны, а не две (§3)."""

    def bars(self, extremes, make):
        bars = [quiet()] * 40
        for value in extremes:
            bars += [make(value), quiet(), quiet()]
        return series(bars), atrs(len(bars))

    def test_accumulation_is_the_same_procedure(self):
        s, a = self.bars([90.0, 94.0, 98.0], lower)
        result = D.detect(s, a, side=D.ACCUMULATION, min_turnover=0.0)
        assert result.verdict == D.VERDICT_CONFIRMED

    def test_accumulation_needs_rising_lows(self):
        s, a = self.bars([98.0, 94.0, 90.0], lower)
        result = D.detect(s, a, side=D.ACCUMULATION, min_turnover=0.0)
        assert result.verdict == D.VERDICT_NONE

    def test_upper_wicks_do_not_feed_the_mirror(self):
        s, a = self.bars([110.0, 106.0, 102.0], upper)
        assert D.detect(s, a, side=D.ACCUMULATION, min_turnover=0.0).count == 0

    def test_analyse_returns_both_sides(self):
        s, a = self.bars([110.0, 106.0, 102.0], upper)
        both = D.analyse(s, a, min_turnover=0.0)
        assert both.distribution.verdict == D.VERDICT_CONFIRMED
        assert both.accumulation.count == 0

    def test_unknown_side_is_refused(self):
        import pytest

        s, a = self.bars([110.0], upper)
        with pytest.raises(ValueError):
            D.detect(s, a, side="вверх")


class TestThinMarket:
    """§5.1: `none` — «проверено, признака нет», `n/a` — «не проверялось»."""

    def thin(self, volume):
        # Между событиями ДВЕ тихих свечи: одна схлопнула бы их в одно
        # событие по правилу §1.2.
        bars = [quiet(volume=volume)] * 40
        for top in (110.0, 106.0, 102.0):
            bars += [
                upper(top, volume=volume * 4),
                quiet(volume=volume), quiet(volume=volume),
            ]
        return D.detect(series(bars), atrs(len(bars)))

    def test_below_the_floor_is_na_with_a_reason(self):
        result = self.thin(1e5)
        assert result.verdict == D.VERDICT_NA
        assert "оборот" in result.reason
        assert result.count == 0

    def test_above_the_floor_is_measured(self):
        assert self.thin(1e7).verdict == D.VERDICT_CONFIRMED


class TestWindowComesFromTheTimeframe:
    """Окно берётся из `absorption_window`, а не константой в 30 свечей.

    На дневке обе величины равны 30, поэтому приёмочные тесты 1–5 от этого не
    меняются; на часовом ряду правило проекта даёт 72, и именно это позволяет
    зеркалу дотянуться до кластеров, которых ждёт приёмочный тест 6.
    """

    def test_daily_window_is_thirty(self):
        from cryptomcp.volume import absorption_window

        assert absorption_window("1d") == D.WINDOW

    def test_hourly_window_is_wider(self):
        from cryptomcp.volume import absorption_window

        assert absorption_window("1h") > D.WINDOW


class TestSpecCases:
    """Живой прогон приёмочных тестов ТЗ, 15.09.2026, спот, через `as_of_ms`.

    | # | вход | ждали | получили |
    |---|------|-------|----------|
    | 1 | OAX 1d, 13.07.2024 | confirmed | **confirmed**, три события |
    | 2 | OAX 1d, 30.05.2024 | forming | **forming**, 4 события против 5 в ТЗ |
    | 3 | OAX 1d, 20.10.2024 | confirmed | **none**, событий 2 |
    | 4 | ORN 1d, 05.06.2024 | none | **none**, событие 1 против 2 в ТЗ |
    | 5 | ORN 1d, 20.06.2024 | none | **none**, климакс 18.06 отсеян фитилём 19% |
    | 6 | ASTER 1h, 19.08.2026 | none + непустое зеркало | **none**, зеркало 4 события |

    Три расхождения — в самом ТЗ, и все замерены:

    - **тест 3 недостижим при пороге 2.0x.** В окне проходят ровно два бара:
      09.10 (41.74x, фитиль 69%) и 13.10 (47.50x, 51%). Ближайшие кандидаты —
      02.10 (1.58x, 65%) и 07.10 (0.95x, 69%), оба не проходят по объёму.
      Третье событие появилось бы при пороге около 1.6x.
    - **тест 4 и «через одну свечу» противоречат друг другу.** §1.2 требует
      схлопывать бары через свечу, §5.2 считает 01.06 и 03.06 двумя событиями,
      хотя между ними ровно одна свеча. Реализовано по §1.2; на вердикт не
      влияет — и одно событие, и два меньше трёх.
    - **смещение в тесте 1 — 2.43 ATR, а не «≈1.7».** Вердикт тот же.

    Отдельно: порог оборота 2M гасит тесты 1 и 6 — OAX на 12.07.2024 давал
    1.40M в сутки по окну, ASTER на 19.08.2026 — 1.66M. Тесты 2–5 через порог
    проходят. В проде он не мешает: архивный универсум начинается с 3M.

    Оборот считается средним по окну, а не медианой, потому что тем же
    средним раздел 2 печатает «M на свечу». С медианой AINUSDT 15.09.2026
    показывал 5.19M в разделе 2 и «0.66M ниже 2M» в разделе 7 — монета
    взорвалась в последние двое суток из тридцати, и две величины под одним
    словом стояли в одном сообщении.
    """

    def test_this_is_documentation(self):
        """Живые кейсы требуют сети; здесь зафиксирован их результат."""
