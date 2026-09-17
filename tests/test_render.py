"""Тесты рендера (PLAN §4.2, §4.10, §5).

Здесь проверяется не арифметика, а то, что выдача не вводит в заблуждение:
одна и та же свеча не должна получать разные числа в разных инструментах, а
у каждого числа должно быть видно, к какому моменту оно относится.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptomcp.analysis import TimeframeView
from cryptomcp.derivatives import Funding, OpenInterest
from cryptomcp.indicators import Metric
from cryptomcp.levels import VolumeProfile
from cryptomcp.render import (
    closed_through,
    render_derivatives,
    render_klines,
    render_screen,
    render_snapshot,
)
from cryptomcp.series import INTERVAL_MS, build_series
from cryptomcp.symbols import SymbolInfo
from cryptomcp.volume import VolumeContext, volume_context

H4 = INTERVAL_MS["4h"]
INFO = SymbolInfo("TESTUSDT", "TEST", "USDT", 0.0001, 4, "PERPETUAL", "TRADING")


def kline(open_time: int, step: int, quote_vol: float, close: float):
    return [
        open_time, f"{close:.8f}", f"{close + 1:.8f}", f"{close - 1:.8f}",
        f"{close:.8f}", "1.0", open_time + step - 1, f"{quote_vol:.8f}", 10,
        "0.5", f"{quote_vol * 0.5:.8f}", "0",
    ]


def series_4h(days: int = 40):
    """Ряд с суточной сезонностью: ночной слот втрое тише дневных."""
    volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 300.0]
    raw = [
        kline(((day * 6) + slot) * H4, H4, volumes[slot], 100.0 + day * 0.1)
        for day in range(days)
        for slot in range(6)
    ]
    return build_series(raw, "TESTUSDT", "4h", raw[-1][6] + 10_000, grace_ms=0)


def volume_cells(text: str) -> list[str]:
    """Колонка «объём» из строк со свечами.

    Ячейка ищется по виду, а не по позиции: колонок в таблице прибавляется
    (абсолютный оборот, куплено по рынку, число сделок), и счёт с конца
    ломался бы при каждой такой правке. Множитель — единственная ячейка,
    оканчивающаяся на «x».
    """
    cells = []
    for line in text.splitlines():
        if not line or not line[0].isdigit():
            continue
        found = [t for t in line.split() if t.rstrip("~").endswith("x")]
        cells.append(found[-1] if found else "n/a")
    return cells


def view(**overrides) -> TimeframeView:
    defaults = dict(
        interval="4h", price=100.0, atr_value=2.0, atr_pct=2.0, rsi_value=50.0,
        ema_state="above", structure="HH/HL", position_in_range=0.5,
        bbw=Metric("BBW", 0.04, pct_rank=10.0, n_obs=360, span_days=90,
                   threshold=20.0, threshold_side="below"),
        atr_metric=Metric("ATR", 2.0, pct_rank=10.0, n_obs=360, span_days=90),
        atr_declining_bars=10,
        change_24h=1.5,
        range_low=95.0, range_high=105.0, range_width=0.10, range_width_atr=5.0,
        range_threshold=0.06,
        range_metric=Metric("диапазон(20)", 10.0, unit="%", pct_rank=15.0,
                            n_obs=360, span_days=90), narrow_bars=0,
        narrow_days=0.0,
        narrow_metric=Metric("длительность сжатия", 0.0, unit=" св.",
                             pct_rank=30.0, n_obs=12, span_days=610),
        volume=VolumeContext(0.7, "медиана слота", 250, 0.6, 3, 0.55),
        profile=VolumeProfile(100.0, 95.0, 105.0, 1e6, 60),
        divergence=None,
        meta={"closed_through_ms": 4 * H4 - 1, "missing": 0},
    )
    defaults.update(overrides)
    return TimeframeView(**defaults)


class TestVolumeColumnIsOneMetric:
    """Одна свеча — одно число, независимо от инструмента и от limit.

    Регрессия: колонка «об./ср» считалась от среднего по показанному окну.
    У CAKE свеча 30.08 давала 4.10x при limit=50 и 2.25x при limit=14, а
    снапшот по ней же — третье число, и колонки выглядели сопоставимыми.
    """

    def test_last_candle_same_at_any_limit(self):
        s = series_4h()
        assert volume_cells(render_klines(s, INFO, 10))[-1] == (
            volume_cells(render_klines(s, INFO, 50))[-1]
        )

    def test_whole_overlap_matches_between_limits(self):
        s = series_4h()
        short = volume_cells(render_klines(s, INFO, 10))
        long_window = volume_cells(render_klines(s, INFO, 50))
        assert short == long_window[-len(short):]

    def test_matches_snapshot_column(self):
        s = series_4h()
        cell = volume_cells(render_klines(s, INFO, 10))[-1]
        ratio = volume_context(s, np.full(len(s), 1.0)).ratio
        assert float(cell.rstrip("x~")) == pytest.approx(round(ratio, 2))

    def test_quiet_slot_is_not_read_as_fading(self):
        """Ночная свеча сравнивается со своим слотом, а не с сутками."""
        s = series_4h()
        assert float(volume_cells(render_klines(s, INFO, 10))[-1].rstrip("x~")) == (
            pytest.approx(1.0, abs=0.1)
        )


class TestClosedThrough:
    """Каким закрытием заканчивается строка — должно быть написано."""

    def test_boundary_not_last_millisecond(self):
        assert closed_through(view()) == "1970-01-01 16:00"

    def test_short_form_drops_year(self):
        assert closed_through(view(), short=True) == "01-01 16:00"

    def test_missing_meta_is_na(self):
        assert closed_through(view(meta={})) == "n/a"

    def test_snapshot_lists_every_timeframe(self):
        text = render_snapshot(
            INFO, {"4h": view(), "1h": view(interval="1h")},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
        )
        assert "закрыты по (UTC): 4h 01-01 16:00 · 1h 01-01 16:00" in text


class TestNarrowBarsWording:
    """Ноль у широкого диапазона — верное значение, а не сломанный счётчик."""

    def test_range_line_names_the_percentile(self):
        """Ширина сама по себе ничего не значит — значим её перцентиль."""
        text = render_snapshot(
            INFO, {"4h": view()},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
        )
        assert "15-й перцентиль своей истории, узким был 0 св. подряд" in text
        assert "держится 0 св." not in text

    def test_range_line_explains_a_missing_base(self):
        from cryptomcp.indicators import Metric

        weak = Metric("диапазон(20)", 10.0, unit="%", pct_rank=None, n_obs=5,
                      span_days=1, base_note="n/a (наблюдений 5, нужно 60)")
        text = render_snapshot(
            INFO, {"4h": view(range_metric=weak)},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
        )
        assert "наблюдений 5" in text

    def test_squeeze_column_separates_two_criteria(self):
        text = render_snapshot(
            INFO, {"4h": view(narrow_bars=7)},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
        )
        assert "ДА · BBW 10 pct · узк 7" in text


class TestDerivativesRendering:
    """Посчитанное должно доезжать до выдачи."""

    def funding(self) -> Funding:
        return Funding(
            "TESTUSDT", 0.0001, 4, 0, mark_price=101.0, index_price=100.0,
            percentile=Metric("funding", 0.0001, pct_rank=65.0, n_obs=500,
                              span_days=83.0),
            history=((0, 0.0001), (H4, 0.0002)),
        )

    def open_interest(self) -> OpenInterest:
        return OpenInterest(
            "TESTUSDT", 1000.0, 2e6, {"1h": 0.02}, {"1h": 0.03},
            percentile=Metric("open_interest", 1000.0, pct_rank=88.0, n_obs=499,
                              span_days=21.0, base_note="история OI ограничена"),
            history=((0, 1000.0, 100.0), (3_600_000, 1020.0, 101.0)),
        )

    def test_basis_reaches_output(self):
        assert "базис +1.000%" in render_derivatives(self.funding(), None)

    def test_oi_percentile_reaches_output(self):
        text = render_derivatives(None, self.open_interest())
        assert "88 pct за 21 сут. (история OI ограничена)" in text

    def test_series_only_when_asked(self):
        """Снапшот обязан оставаться коротким, get_derivatives — нет."""
        short = render_derivatives(self.funding(), self.open_interest())
        full = render_derivatives(
            self.funding(), self.open_interest(), history=True, precision=2
        )
        assert "по часам" not in short
        assert "начисления" not in short
        assert "по часам" in full
        assert "+2.00" in full  # ΔOI% относительно начала окна


class TestSkippedTimeframes:
    """Недоступный ТФ печатается строкой, а не исчезает из лестницы.

    Пропавшая без объяснения строка читается как «на 1w сжатия нет», хотя
    означает «не считали»; молчание здесь хуже пометки.
    """

    def snapshot(self):
        from cryptomcp.errors import insufficient_history

        return render_snapshot(
            INFO, {"4h": view()},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
            skipped={"1w": insufficient_history("UAIUSDT", "1w", 43, 60)},
            order=("1w", "4h"),
        )

    def test_row_states_the_shortfall(self):
        assert "1w   — недостаточно истории (43/60)" in self.snapshot()

    def test_row_keeps_its_place_in_the_ladder(self):
        rows = [
            line for line in self.snapshot().splitlines()
            if line.startswith(("1w", "4h"))
        ]
        assert rows[0].startswith("1w")
        assert rows[1].startswith("4h")

    def test_full_reason_printed_below(self):
        text = self.snapshot()
        assert "недоступные ТФ — остальные посчитаны:" in text
        assert "свежий листинг — это не сбой" in text

    def test_closed_through_lists_only_computed(self):
        assert "закрыты по (UTC): 4h 01-01 16:00" in self.snapshot()

    def test_no_block_when_nothing_skipped(self):
        text = render_snapshot(
            INFO, {"4h": view()},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=4 * H4,
        )
        assert "недоступные ТФ" not in text


class TestVersionBumpIsExplained:
    """Час после подъёма версии выдача выглядит сломанной, но не сломана.

    Колонки скана пусты у ВСЕХ строк разом: записи текущего поколения ещё не
    написаны, а прежние с ним не сравниваются. Та же подсказка уже стоит в
    scan_pairs; без неё отличить переход от поломки снаружи нельзя.
    """

    NOW = 1_788_393_600_000

    def episode(self):
        return {
            "symbol": "BTCUSDT", "tf": "1d", "status": "active",
            "entered_at": self.NOW - 86_400_000, "entered_by": "scanner",
            "squeeze_index": 0.4, "accumulation_score": None,
            "rank_at_entry": 7, "price_at_entry": 90123.45,
            "last_rank": 3, "last_index": 0.47,
            "exited_at": None, "exit_reason": None,
        }

    def test_empty_scans_with_earlier_versions_are_explained(self):
        from cryptomcp.render import render_watchlist

        text = render_watchlist(
            [self.episode()], {}, now_ms=self.NOW,
            earlier=["v6"], version="v7",
        )
        assert "колонки скана пусты не из-за сбоя" in text
        assert "v6" in text and "v7" in text

    def test_nothing_is_said_when_scans_are_there(self):
        from cryptomcp.render import render_watchlist

        scan = {"symbol": "BTCUSDT", "price": 90123.45, "narrow_bars": 11,
                "closed_through_ms": self.NOW - 3_600_000}
        text = render_watchlist(
            [self.episode()], {("BTCUSDT", "1d"): scan}, now_ms=self.NOW,
            earlier=["v6"], version="v7",
        )
        assert "колонки скана пусты" not in text

    def test_trend_is_printed_as_a_mark_not_a_filter(self):
        """Пометка тренда: «ниже обеих EMA» печатается, но строку не убирает."""
        from cryptomcp.render import render_watchlist

        scan = {"symbol": "BTCUSDT", "price": 90123.45, "narrow_bars": 11,
                "ema_state": "below", "closed_through_ms": self.NOW - 3_600_000}
        text = render_watchlist(
            [self.episode()], {("BTCUSDT", "1d"): scan}, now_ms=self.NOW,
        )

        assert "тренд" in text.splitlines()[1]
        assert "ниже" in text
        assert "BTCUSDT" in text
        assert "ПОМЕТКА, а не фильтр" in text

    def test_twin_flow_quadrant_is_printed_short(self):
        """Поток соседнего рынка печатается справкой, одним словом."""
        from cryptomcp.render import render_watchlist

        scan = {"symbol": "BTCUSDT", "price": 90123.45, "narrow_bars": 11,
                "twin_delta_quadrant": "поглощение",
                "closed_through_ms": self.NOW - 3_600_000}
        text = render_watchlist(
            [self.episode()], {("BTCUSDT", "1d"): scan}, now_ms=self.NOW,
        )

        assert "поток²" in text.splitlines()[1]
        assert "погл" in text
        assert "в ранг не входит" in text

    def test_trend_without_a_scan_row_is_a_dash(self):
        from cryptomcp.render import render_watchlist

        text = render_watchlist([self.episode()], {}, now_ms=self.NOW)

        assert "тренд" in text.splitlines()[1]

    def test_nothing_is_said_without_earlier_generations(self):
        """Пустой журнал без прежних поколений — это другое, и лечится другим."""
        from cryptomcp.render import render_watchlist

        text = render_watchlist([self.episode()], {}, now_ms=self.NOW)
        assert "колонки скана пусты" not in text


class TestStaleArchiveIsNamed:
    """Замер 14.09.2026: STORJUSDT (спот) печатал живую цену и рядом
    «закрыты по 1d 09-03 03:00». Архив оборвался одиннадцатью сутками раньше,
    но понять это можно было только вычитанием даты из сегодняшней — а живая
    цена в шапке создавала ровно обратное впечатление."""

    DAY = 86_400_000

    def render(self, *, age_days, quote_volume_24h=0.9e6):
        now = 100 * self.DAY
        closed = int(now - age_days * self.DAY)
        return render_snapshot(
            INFO, {"1d": view(interval="1d", meta={"closed_through_ms": closed})},
            live_price=100.5, change_24h=1.0,
            quote_volume_24h=quote_volume_24h, now_ms=now,
        )

    def test_stale_series_is_called_out(self):
        text = self.render(age_days=11)
        assert "ДАННЫЕ НЕ ОБНОВЛЯЛИСЬ" in text
        assert "1d — 11.0 сут назад" in text

    def test_last_known_turnover_is_printed(self):
        """Оборот отвечает на вопрос «почему перестали»: ниже порога архива.

        Формат тот же, что в шапке: два разных написания одного числа в одном
        сообщении читаются как два разных числа.
        """
        assert "оборот за сутки сейчас 0.9M USDT" in self.render(age_days=11)

    def test_one_missed_candle_is_not_an_alarm(self):
        """Отставание на свечу — норма: строка пишется на закрытие."""
        assert "ДАННЫЕ НЕ ОБНОВЛЯЛИСЬ" not in self.render(age_days=1.5)

    def test_bound_is_the_timeframe_not_the_calendar(self):
        """Те же полтора суток на 4h — это девять пропущенных свечей."""
        now = 100 * self.DAY
        text = render_snapshot(
            INFO,
            {"4h": view(meta={"closed_through_ms": int(now - 1.5 * self.DAY)})},
            live_price=100.5, change_24h=1.0, quote_volume_24h=1e9, now_ms=now,
        )
        assert "ДАННЫЕ НЕ ОБНОВЛЯЛИСЬ" in text


class TestWatchlistRender:
    """Выдача списка наблюдения: чего в ней не должно быть видно неправильно."""

    NOW = 1_788_393_600_000
    H4 = 4 * 3_600_000

    def episode(self, **over):
        row = {
            "symbol": "BTCUSDT", "tf": "4h", "status": "active",
            "entered_at": self.NOW - 2 * 86_400_000, "entered_by": "scanner",
            "squeeze_index": 0.41, "accumulation_score": None,
            "rank_at_entry": 7, "price_at_entry": 90123.45,
            "last_rank": 3, "last_index": 0.47,
            "exited_at": None, "exit_reason": None,
        }
        row.update(over)
        return row

    def render(self, rows, scans=None):
        from cryptomcp.render import render_watchlist

        return render_watchlist(rows, scans or {}, now_ms=self.NOW)

    def test_empty_says_so(self):
        assert self.render([]) == "список наблюдения пуст"

    def test_high_price_keeps_its_order_of_magnitude(self):
        """Формат %.6g обрезал бы 90123.45 до 90123.4 — разряд важнее знака."""
        assert "90123.45" in self.render([self.episode()])

    def test_rank_shows_entry_and_current(self):
        assert "7→3" in self.render([self.episode()])

    def test_missing_rank_is_dash_not_zero(self):
        """У ручной записи ранга при входе не было; ноль читался бы как «первый»."""
        text = self.render([self.episode(rank_at_entry=None, entered_by="manual")])
        assert "—→3" in text

    def test_a_frozen_scan_is_named_in_the_row(self):
        """Строка от 08.09 не должна читаться как сегодняшняя (замер 14.09.2026)."""
        scan = {
            "symbol": "BTCUSDT", "price": 90123.45, "narrow_bars": 18,
            "closed_through_ms": self.NOW - 6 * 86_400_000,
        }
        text = self.render(
            [self.episode(tf="1d")], {("BTCUSDT", "1d"): scan}
        )

        assert "данные от" in text
        assert "6.0 сут назад" in text

    def test_a_frozen_scan_does_not_fake_a_flat_move(self):
        """У AIOTUSDT цена скана равнялась цене входа, и выдача печатала +0.0%.

        Фактически монета прошла +19.7%: цена скана замёрзла вместе со
        строкой, а колонка «сейчас» выдавала её за текущую.
        """
        scan = {
            "symbol": "BTCUSDT", "price": 90123.45,
            "closed_through_ms": self.NOW - 6 * 86_400_000,
        }
        text = self.render(
            [self.episode(tf="1d")], {("BTCUSDT", "1d"): scan}
        )

        assert "+0.0%" not in text
        assert "n/a" in text

    def test_a_fresh_scan_is_not_marked(self):
        scan = {
            "symbol": "BTCUSDT", "price": 90123.45, "narrow_bars": 18,
            "closed_through_ms": self.NOW - self.H4,
        }
        text = self.render([self.episode()], {("BTCUSDT", "4h"): scan})

        assert "данные от" not in text

    def test_accumulation_prints_na_while_not_measured(self):
        """Пусто в колонке — «не измерено», а не «признака нет»."""
        assert "n/a" in self.render([self.episode()])

    def test_closed_episode_stops_aging_and_shows_reason(self):
        row = self.episode(
            status="broken_out", exited_at=self.NOW - 86_400_000,
            exit_reason="пробой диапазона входа",
        )
        text = self.render([row])
        assert "пробой диапазона входа" in text
        # Вошёл двое суток назад, вышел сутки назад — эпизод прожил ровно сутки.
        assert " 1.0  scanner" in text

    def test_move_counted_from_entry_price(self):
        scans = {("BTCUSDT", "4h"): {"price": 99135.795, "narrow_bars": 4}}
        text = self.render([self.episode()], scans)
        assert "+10.0%" in text
        assert "    4" in text


class TestScanHistoryRender:
    NOW = 1_788_393_600_000

    def row(self, **over):
        row = {
            "ts_ms": self.NOW + 900_000, "symbol": "HOMEUSDT", "tf": "4h",
            "squeeze_index": 0.64,
            "components": (
                '{"volatility": 0.59, "range": 0.62, "volume": 0.59, '
                '"divergence": 1.0}'
            ),
            "excluded": '["value_area"]', "price": 0.005929,
            "range_width_pct": 12.86, "narrow_bars": 26,
            "bbw_pct_rank": 69.0, "volume_ratio": 0.83,
            "closed_through_ms": self.NOW - 1,
        }
        row.update(over)
        return row

    def render(self, rows):
        from cryptomcp.render import render_scan_history

        return render_scan_history("HOMEUSDT", "4h", rows, "v3")

    def test_empty_explains_why(self):
        assert "записей скана нет" in self.render([])

    def test_candle_boundary_not_last_millisecond(self):
        """closed_through_ms — конец свечи: печатать надо границу, а не 23:59."""
        import datetime as dt

        expected = dt.datetime.fromtimestamp(self.NOW / 1000, dt.UTC)
        text = self.render([self.row()])
        assert expected.strftime("%Y-%m-%d %H:%M") in text
        assert "23:59" not in text

    def test_excluded_group_is_dash_not_zero(self):
        """Ноль в группе означал бы «признака нет», а не «не измерили»."""
        line = self.render([self.row()]).splitlines()[2]
        groups = line[24:59]  # пять колонок по 7 знаков после времени и индекса
        assert groups.count("—") == 1
        assert "0.00" not in groups


class TestEmptyScreenReason:
    """Пустая выдача обязана называть СВОЮ причину.

    Их три, и лечатся они разным: журнала по этому ТФ нет вовсе, фильтры
    отсеяли всех, и — сразу после подъёма версии формулы — записи есть, но
    прежнего поколения. Последняя причина временная и проходит сама с
    ближайшим прогоном; без неё выдача винила таймфрейм, и это читалось как
    поломка сканера.
    """

    def test_version_bump_is_named(self):
        text = render_screen(
            "4h", [], version="v4", sort_by="squeeze",
            filtered=122, logged=0, matched=0, earlier=["v3"],
        )
        assert "формула поднята до v4" in text
        assert "v3" in text
        assert "ближайшим часовым прогоном" in text

    def test_no_journal_at_all(self):
        text = render_screen(
            "2h", [], version="v4", sort_by="squeeze",
            filtered=122, logged=0, matched=0,
        )
        assert "записей скана нет вовсе" in text

    def test_filters_rejected_everyone(self):
        text = render_screen(
            "4h", [], version="v4", sort_by="squeeze",
            filtered=122, logged=130, matched=0,
        )
        assert "ни одна не прошла фильтры" in text


class TestLadderColumns:
    """Лестница снапшота показывает то, что видно только в сравнении строк.

    Заготовка строится здесь, а не импортируется из соседнего тест-модуля:
    кросс-импорты между тестами в проекте уже убирали однажды, и они ломаются
    ровно там, где их не видно локально (в CI нет пакета ``tests``).
    """

    def view(self, interval="4h", *, ma_ratio=0.6, shock=None):
        from cryptomcp.analysis import TimeframeView
        from cryptomcp.indicators import Metric
        from cryptomcp.volume import VolumeContext

        return TimeframeView(
            interval=interval, price=100.0, atr_value=2.0, atr_pct=2.0,
            rsi_value=50.0, ema_state="above", structure="HH/HL",
            position_in_range=0.5,
            bbw=Metric("BBW", 0.04, pct_rank=10.0, n_obs=360, span_days=90),
            atr_metric=Metric("ATR", 2.0, pct_rank=10.0, n_obs=360, span_days=90),
            atr_declining_bars=10,
            change_24h=1.5,
            range_low=95.0, range_high=105.0, range_width=0.03,
            range_width_atr=1.5, range_threshold=0.06,
            range_metric=Metric("диапазон(20)", 10.0, unit="%", pct_rank=15.0,
                                n_obs=360, span_days=90),
            narrow_bars=20,
            narrow_days=3.3,
            narrow_metric=Metric("длительность сжатия", 20.0, unit=" св.",
                                 pct_rank=80.0, n_obs=12, span_days=610),
            volume=VolumeContext(1.0, "сезонный слот", 250, ma_ratio, 0, 0.5),
            profile=None,
            divergence=None,
            shock=shock,
        )

    def ladder(self, views):
        from cryptomcp.render import render_snapshot
        from cryptomcp.symbols import SymbolInfo

        info = SymbolInfo(
            symbol="TESTUSDT", base="TEST", quote="USDT", tick_size=0.001,
            price_precision=3, contract_type="PERPETUAL", status="TRADING",
        )
        return render_snapshot(
            info, views, live_price=100.0, change_24h=1.0,
            quote_volume_24h=1e7, now_ms=1_788_000_000_000,
            order=tuple(views),
        )

    def test_ma_ratio_shows_the_divergence_between_timeframes(self):
        """Кейс ASTER 19.08.2026: дневная 0.24x при часовой 1.45x."""
        text = self.ladder({
            "1d": self.view("1d", ma_ratio=0.24),
            "1h": self.view("1h", ma_ratio=1.45),
        })

        assert "MA20/100" in text
        assert "0.24x" in text
        assert "1.45x" in text

    def test_shock_marks_the_squeeze_cell(self):
        from cryptomcp.analysis import Shock

        loud = Shock(bars_ago=9, range_atr=4.0, range_share=0.94, volume_ratio=5.6)
        quiet = Shock(bars_ago=2, range_atr=1.3, range_share=0.34, volume_ratio=1.5)

        assert "ШОК 4.0 ATR" in self.ladder({"4h": self.view(shock=loud)})

        text = self.ladder({"4h": self.view(shock=quiet)})
        # «ШОК» есть и в легенде под лестницей, поэтому проверяется ячейка.
        assert "ШОК 1.3" not in text
        ladder_row = next(line for line in text.splitlines() if line.startswith("4h "))
        assert "ШОК" not in ladder_row


class TestShockCell:
    """В таблице скана печатается только событие, а не самый широкий бар."""

    def test_event_printed(self):
        from cryptomcp.render import _shock

        assert _shock({"shock_atr": 4.0, "shock_volume": 5.6}) == "4.0"

    def test_wide_but_quiet_is_a_dash(self):
        from cryptomcp.render import _shock

        assert _shock({"shock_atr": 4.0, "shock_volume": 1.2}) == "—"
        assert _shock({"shock_atr": 1.3, "shock_volume": 5.6}) == "—"

    def test_missing_is_a_dash(self):
        from cryptomcp.render import _shock

        assert _shock({}) == "—"


class TestTwinMarketColumns:
    """Величины соседнего рынка: печатаются, когда есть, и не выдумываются."""

    @staticmethod
    def render(rows):
        from cryptomcp.render import render_scan_history

        return render_scan_history("HOMEUSDT", "4h", rows, "v4")

    def row(self, **extra):
        base = {
            "ts_ms": 1_788_480_000_000,
            "closed_through_ms": 1_788_479_999_999,
            "symbol": "HOMEUSDT",
            "source": "spot",
            "squeeze_index": 0.68,
            "components": "{}",
            "range_width_pct": 10.24,
            "narrow_bars": 17,
            "price": 0.00615,
        }
        base.update(extra)
        return base

    def test_columns_appear_with_the_second_market(self):
        text = self.render([self.row(
            twin_market="futures", twin_index=0.66,
            twin_range_width_pct=8.85, twin_narrow_bars=36,
        )])

        assert "инд²" in text and "диап²" in text and "узк²" in text
        assert "8.85%" in text
        assert "соседнем рынке (перп)" in text

    def test_without_the_second_market_columns_do_not_appear(self):
        """Столбец прочерков сообщал бы «совпало», а верно «не мерили»."""
        text = self.render([self.row()])

        assert "узк²" not in text

    def test_market_of_the_row_is_in_the_header(self):
        text = self.render([self.row()])

        assert "рынок: спот" in text


class TestAbsoluteTurnover:
    """Множитель отвечает «много это или мало», абсолют — «сколько».

    До этой правки абсолютных сумм не печатал ни один инструмент, хотя
    заголовок колонки в get_klines говорил «объём в USDT». Из-за этого
    неразличимы «объём ниже среднего» и «оборота нет вовсе»: затухание до
    0.35x выглядит одинаково у монеты с 30K оборота и у монеты с 30M.
    """

    def test_short_form_by_scale(self):
        from cryptomcp.render import usdt

        assert usdt(2_100_000_000) == "2.10B"
        assert usdt(341_000_000) == "341.00M"
        assert usdt(12_400) == "12.40K"
        assert usdt(920) == "920"

    def test_missing_value_is_not_a_zero(self):
        from cryptomcp.render import usdt

        assert usdt(float("nan")) == "n/a"

    def test_klines_print_absolute_columns(self):
        text = render_klines(series_4h(), INFO, 5)

        assert "оборот" in text and "куплено" in text and "сделок" in text

    def test_multiplier_column_survives(self):
        """Абсолют добавлен рядом с множителем, а не вместо него."""
        assert volume_cells(render_klines(series_4h(), INFO, 5))[-1].endswith("x")

    def test_metrics_print_window_turnover(self):
        from cryptomcp.render import render_squeeze_metrics
        from cryptomcp.volume import VolumeContext

        volume = VolumeContext(
            1.0, "медиана последних 20", 250, 0.6, 3, 0.55,
            bars_window=30, quote_total=110_270_000.0,
            taker_buy_quote_total=52_520_000.0,
        )
        text = render_squeeze_metrics(view(volume=volume))

        assert "оборот окна" in text
        assert "110.27M USDT за 30 свечей" in text
        assert "3.68M на свечу" in text
        assert "куплено по рынку 52.52M (48%)" in text


class TestAccumulationList:
    """Второй список печатает кластер первым: он основание отбора."""

    NOW = 1_788_400_000_000

    def entry(self, **fields):
        row = {
            "symbol": "IDUSDT", "tf": "1d", "market": "spot", "status": "active",
            "entered_at": self.NOW - 86_400_000, "entered_index": 0.80,
            "entered_clusters": 2, "entered_bars": 3, "price_at_entry": 0.0324,
            "rank_at_entry": 3, "last_rank": 1, "last_index": 0.85,
            "last_clusters": 2, "exited_at": None, "exit_reason": None,
        }
        row.update(fields)
        return row

    def test_empty_list_says_why(self):
        from cryptomcp.render import render_accumulation

        text = render_accumulation([], {}, now_ms=self.NOW)

        assert "кластером набора" in text

    def test_columns_and_move_from_entry(self):
        from cryptomcp.render import render_accumulation

        scan = {"symbol": "IDUSDT", "price": 0.0356, "ema_state": "mixed",
                "twin_delta_quadrant": "поглощение"}
        text = render_accumulation(
            [self.entry()], {("IDUSDT", "1d"): scan}, now_ms=self.NOW, version="v9",
        )

        header = text.splitlines()[1]
        assert "клст" in header and "бары" in header and "поток²" in header
        assert "+9.9%" in text, "ход от цены входа"
        assert "3→1" in text
        assert "погл" in text and "смеш" in text
        assert "outcomes" in text, "в подписи сказано, ради чего список ведётся"

    def test_closed_entry_prints_the_reason(self):
        from cryptomcp.render import render_accumulation

        text = render_accumulation(
            [self.entry(exited_at=self.NOW, exit_reason="кластера набора больше нет",
                        status="exited")],
            {}, now_ms=self.NOW,
        )

        assert "чем кончилось" in text.splitlines()[1]
        assert "кластера набора больше нет" in text
