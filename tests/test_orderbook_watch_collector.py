"""Уборка стакана не имеет права ломать сбор невосстановимых деривативов."""

from __future__ import annotations

from cryptomcp.collector import cleanup_order_book_watches


def test_order_book_cleanup_is_noop_when_third_database_does_not_exist(monkeypatch):
    """Р3: до первого start отсутствие order_book_watch.sqlite штатно и тихо."""
    monkeypatch.setattr(
        "cryptomcp.collector.orderbook_watch.cleanup",
        lambda: {"expired": 0, "deleted": 0},
    )

    assert cleanup_order_book_watches() == {"expired": 0, "deleted": 0}


def test_order_book_cleanup_failure_is_logged_but_not_raised(monkeypatch, caplog):
    """Р3: блокировка третьей БД не должна стоить часа market.sqlite-данных."""
    def broken_cleanup():
        raise OSError("database is locked")

    monkeypatch.setattr("cryptomcp.collector.orderbook_watch.cleanup", broken_cleanup)

    assert cleanup_order_book_watches() == {"expired": 0, "deleted": 0}
    assert "уборка сессий стакана не выполнена" in caplog.text
