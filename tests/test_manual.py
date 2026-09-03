"""Тесты ручных записей списка наблюдения (PLAN §4.28).

Проверяется главное свойство решения: ручная запись живёт в СВОЁМ файле и
никак не пересекается с рабочей структурой сборщика — ни таблицей, ни
логикой выхода по рангу.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from cryptomcp import manual

DAY = 86_400_000
NOW = 1_788_500_000_000


@pytest.fixture
def con(tmp_path):
    connection = manual.connect(str(tmp_path / "manual.sqlite"))
    yield connection
    connection.close()


class TestAddRemove:
    def test_entry_keeps_price_and_note(self, con):
        """Цена на входе обязательна: без неё ручной отбор не сравнить со
        сканерным, а ради этого сравнения признак и заводится."""
        assert manual.add(
            con, "ASTERUSDT", "4h", entered_at=NOW, note="фитили", price=0.6,
        )

        row = manual.entries(con, now_ms=NOW)[0]
        assert row["symbol"] == "ASTERUSDT"
        assert row["price_at_entry"] == pytest.approx(0.6)
        assert row["note"] == "фитили"
        assert row["entered_by"] == "manual"
        assert row["status"] == "manual"

    def test_second_add_does_not_duplicate(self, con):
        assert manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)
        assert manual.add(con, "ASTERUSDT", "4h", entered_at=NOW + 1000) == 0
        assert len(manual.entries(con, now_ms=NOW)) == 1

    def test_same_pair_on_two_timeframes_is_two_entries(self, con):
        manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)
        manual.add(con, "ASTERUSDT", "1d", entered_at=NOW)
        assert len(manual.entries(con, now_ms=NOW)) == 2

    def test_remove_takes_the_coin_not_the_row(self, con):
        """Снимают монету, а не строку: иначе половина осталась бы молча."""
        manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)
        manual.add(con, "ASTERUSDT", "1d", entered_at=NOW)

        assert manual.remove(con, "ASTERUSDT", removed_at=NOW + DAY, reason="пробой") == 2
        assert manual.entries(con, now_ms=NOW + DAY) == []

        closed = manual.entries(con, now_ms=NOW + DAY, status="closed")
        assert [row["status"] for row in closed] == ["removed", "removed"]
        assert {row["remove_reason"] for row in closed} == {"пробой"}

    def test_remove_of_unknown_pair_reports_zero(self, con):
        assert manual.remove(con, "NOPEUSDT", removed_at=NOW) == 0

    def test_reentry_after_removal_is_allowed(self, con):
        manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)
        manual.remove(con, "ASTERUSDT", removed_at=NOW + DAY)
        assert manual.add(con, "ASTERUSDT", "4h", entered_at=NOW + 2 * DAY)
        assert len(manual.entries(con, now_ms=NOW + 2 * DAY)) == 1
        assert len(manual.entries(con, now_ms=NOW + 2 * DAY, status="all")) == 2


class TestExpiry:
    """Срок — предикат, а не событие: строка не переписывается и переживает
    своё истечение, иначе сравнивать ручной отбор со сканерным будет не с чем."""

    def test_entry_expires_after_thirty_days(self, con):
        manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)

        assert manual.entries(con, now_ms=NOW + 29 * DAY)[0]["status"] == "manual"
        assert manual.entries(con, now_ms=NOW + 31 * DAY) == []

        expired = manual.entries(con, now_ms=NOW + 31 * DAY, status="expired")
        assert [row["symbol"] for row in expired] == ["ASTERUSDT"]
        assert expired[0]["removed_at"] is None      # строка не тронута

    def test_expired_falls_into_closed(self, con):
        manual.add(con, "ASTERUSDT", "4h", entered_at=NOW)
        closed = manual.entries(con, now_ms=NOW + 31 * DAY, status="closed")
        assert [row["status"] for row in closed] == ["expired"]


class TestReadOnly:
    def test_missing_file_is_not_created_on_read(self, tmp_path):
        """Пока никто ничего не добавил, файла нет — и читатель его не заводит.

        Иначе сборщик, читающий список для CLI, стал бы автором файла,
        писателем которого он не является.
        """
        path = str(tmp_path / "none.sqlite")
        assert manual.read_only(path) is None
        assert not os.path.exists(path)
        assert manual.entries(None, now_ms=NOW) == []

    def test_read_only_connection_cannot_write(self, tmp_path):
        path = str(tmp_path / "manual.sqlite")
        writer = manual.connect(path)
        manual.add(writer, "ASTERUSDT", "4h", entered_at=NOW)
        writer.close()

        reader = manual.read_only(path)
        try:
            assert len(manual.entries(reader, now_ms=NOW)) == 1
            with pytest.raises(sqlite3.OperationalError):
                reader.execute("DELETE FROM manual_entries")
        finally:
            reader.close()


class TestMerge:
    """Слияние двух баз при чтении."""

    def scanner(self, symbol="ASTERUSDT", tf="4h", **kw):
        row = dict(
            symbol=symbol, tf=tf, status="active", entered_at=NOW,
            entered_by="scanner", exited_at=None, price_at_entry=0.6,
            rank_at_entry=3, last_rank=2, squeeze_index=0.6, last_index=0.7,
            accumulation_score=None,
        )
        row.update(kw)
        return row

    def manual_row(self, symbol="ASTERUSDT", tf="4h", **kw):
        row = dict(
            symbol=symbol, tf=tf, note="фитили", entered_at=NOW,
            entered_by="manual", price_at_entry=0.6, removed_at=None,
            remove_reason=None, source="manual", status="manual",
        )
        row.update(kw)
        return row

    def test_pair_in_both_bases_is_one_row(self):
        from cryptomcp.collector import merge_manual

        merged = merge_manual([self.scanner()], [self.manual_row()])

        assert len(merged) == 1
        assert merged[0]["entered_by"] == "scanner+manual"
        assert merged[0]["note"] == "фитили"
        assert merged[0]["last_rank"] == 2          # ранг остаётся сканерным

    def test_manual_only_pair_has_no_rank(self):
        from cryptomcp.collector import merge_manual

        merged = merge_manual([], [self.manual_row(symbol="HEIUSDT")])

        assert len(merged) == 1
        assert merged[0]["entered_by"] == "manual"
        assert merged[0]["last_rank"] is None
        assert merged[0]["squeeze_index"] is None

    def test_closed_scanner_episode_does_not_absorb_manual_entry(self):
        """Закрытый эпизод — это прошлое, а ручная запись открыта сейчас."""
        from cryptomcp.collector import merge_manual

        merged = merge_manual(
            [self.scanner(status="broken_out", exited_at=NOW + DAY)],
            [self.manual_row()],
        )

        assert len(merged) == 2
        assert {row["entered_by"] for row in merged} == {"scanner", "manual"}


class TestRenderedManualRow:
    def test_index_of_manual_row_is_a_dash_not_zero(self):
        """Ноль в индексе читался бы как «сжатия нет», а его не мерили."""
        from cryptomcp.render import render_watchlist

        row = dict(
            symbol="HEIUSDT", tf="4h", status="manual", entered_at=NOW,
            entered_by="manual", exited_at=None, price_at_entry=1.0,
            note="упёрлась в уровень", rank_at_entry=None, last_rank=None,
            squeeze_index=None, last_index=None, accumulation_score=None,
        )
        text = render_watchlist([row], {}, now_ms=NOW + DAY)

        line = next(x for x in text.splitlines() if x.startswith("HEIUSDT"))
        assert "0.00" not in line
        assert "manual" in line
        assert "упёрлась в уровень" in text
