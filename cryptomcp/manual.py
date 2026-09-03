"""Ручные записи в список наблюдения — ОТДЕЛЬНЫЙ файл базы (PLAN §4.28).

Часть кандидатов находится глазами, а не рангом, и до сих пор положить их в
список было нельзя: `watchlist` ведёт сборщик, а сервер открывает базу
`read_only`.

Инвариант «сервер не пишет» защищает невосстановимые деривативы: сборщик,
упавший из-за чужой ошибки, теряет час открытого интереса навсегда. К ручной
записи в список это отношения не имеет — инвариант там шире задачи, которую
решает. Поэтому базы разведены физически, а не по таблицам одного файла:

- `market.sqlite` — пишет только сборщик, сервер читает `read_only`. Инвариант
  остаётся буквально нетронутым, а не «морально соблюдённым»;
- `watchlist_manual.sqlite` — пишет только сервер. Конкуренции за запись нет в
  принципе, поэтому не нужны ни `busy_timeout`, ни согласование с часовым
  прогоном. Строк здесь десятки.

Почему не колонка `entered_by='manual'` в самой `watchlist`: та таблица —
рабочая структура сборщика, он считает по ней ранги, применяет гистерезис
15/40 и помечает истёкшие. Ручная запись без ранга попала бы в логику, целиком
построенную на ранге, и каждое будущее изменение правил выхода обязано было бы
помнить про этот случай.

Читать обе базы одним запросом, если понадобится, позволяет `ATTACH` — файлы
для этого переносить не нужно.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from . import storage

#: Файл рядом с архивом: тот же том, та же судьба при деплое. Отдельный
#: `CRYPTOMCP_MANUAL_DB` нужен разве что тестам.
DEFAULT_PATH = os.environ.get("CRYPTOMCP_MANUAL_DB") or os.path.join(
    os.path.dirname(storage.DEFAULT_PATH) or ".", "watchlist_manual.sqlite"
)

#: Через сколько суток ручная запись считается истёкшей.
#:
#: Сборщик её не снимает и снять не может — он этой базы не видит. Снимается
#: она руками или по сроку, иначе список, который ведут глазами, за месяц
#: превращается в кладбище.
EXPIRY_DAYS = 30

DAY_MS = 86_400_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_entries (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT    NOT NULL,
    tf            TEXT    NOT NULL,
    note          TEXT,
    entered_at    INTEGER NOT NULL,
    entered_by    TEXT    NOT NULL DEFAULT 'manual',
    -- Цена на момент добавления. Без неё ручную запись нельзя сравнить со
    -- сканерной: смысл ручного списка не только в удобстве, но и в ответе на
    -- вопрос, находит ли глаз то, чего не находит формула.
    price_at_entry REAL,
    removed_at    INTEGER,
    remove_reason TEXT
);

-- Открытая запись на пару и таймфрейм может быть только одна; снятых —
-- сколько угодно. Тот же приём, что у эпизодов сканера.
CREATE UNIQUE INDEX IF NOT EXISTS manual_open
    ON manual_entries (symbol, tf) WHERE removed_at IS NULL;
"""


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Открыть базу ручных записей на запись, создав файл при первом обращении."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    con = sqlite3.connect(path, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    con.commit()
    return con


def read_only(path: str = DEFAULT_PATH) -> sqlite3.Connection | None:
    """Открыть для чтения, НЕ создавая файл.

    Пока никто ничего не добавил, файла нет — и это штатное состояние, а не
    ошибка. Создавать его на чтении нельзя: тогда сборщик, читающий список для
    CLI, стал бы автором файла, писателем которого он не является.
    """
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    con.row_factory = sqlite3.Row
    return con


def status_of(row: dict[str, Any], now_ms: int) -> str:
    """Состояние записи: снята руками, истекла по сроку или открыта."""
    if row.get("removed_at"):
        return "removed"
    if now_ms - int(row["entered_at"]) >= EXPIRY_DAYS * DAY_MS:
        return "expired"
    return "manual"


def add(
    con: sqlite3.Connection,
    symbol: str,
    tf: str,
    *,
    entered_at: int,
    note: str | None = None,
    price: float | None = None,
) -> int:
    """Завести ручную запись. Возвращает id, либо 0, если такая уже открыта."""
    cursor = con.execute(
        "INSERT OR IGNORE INTO manual_entries "
        "(symbol, tf, note, entered_at, entered_by, price_at_entry) "
        "VALUES (?, ?, ?, ?, 'manual', ?)",
        (symbol, tf, note, entered_at, price),
    )
    con.commit()
    return int(cursor.lastrowid or 0) if cursor.rowcount else 0


def remove(
    con: sqlite3.Connection,
    symbol: str,
    *,
    removed_at: int,
    reason: str | None = None,
) -> int:
    """Снять все открытые ручные записи по паре. Возвращает число снятых.

    Таймфрейм не спрашивается сознательно: снимают монету, а не строку. Если
    она стоит на двух таймфреймах, снимаются обе — иначе половина осталась бы
    в списке молча.
    """
    cursor = con.execute(
        "UPDATE manual_entries SET removed_at = ?, remove_reason = ? "
        "WHERE symbol = ? AND removed_at IS NULL",
        (removed_at, reason, symbol),
    )
    con.commit()
    return int(cursor.rowcount)


def entries(
    con: sqlite3.Connection | None,
    *,
    now_ms: int,
    status: str | None = None,
    tf: str | None = None,
) -> list[dict[str, Any]]:
    """Ручные записи с посчитанным состоянием.

    Срок не мутирует строку: истечение — это предикат, а не событие. Так
    чтение остаётся чтением, а запись переживает свой срок и попадает потом в
    сравнение ручного отбора со сканерным.
    """
    if con is None:
        return []
    rows = [dict(row) for row in con.execute(
        "SELECT * FROM manual_entries ORDER BY entered_at DESC, id DESC"
    )]
    for row in rows:
        row["source"] = "manual"
        row["status"] = status_of(row, now_ms)
    if tf is not None:
        rows = [row for row in rows if row["tf"] == tf]
    return [row for row in rows if _matches(row, status)]


def _matches(row: dict[str, Any], status: str | None) -> bool:
    """Как фильтр статуса сканера ложится на ручные записи.

    Умолчание — открытые: это рабочий список, ради которого всё и делалось.
    `closed` собирает и снятые руками, и истёкшие по сроку: для читающего это
    одно и то же состояние «больше не смотрим».
    """
    if status is None:
        return row["status"] == "manual"
    if status == "all":
        return True
    if status == "closed":
        return row["status"] in ("removed", "expired")
    return row["status"] == status
