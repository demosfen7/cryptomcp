"""Журнал расчётов (PLAN §4.11).

ТЗ §1.2 называет проверку собственных срабатываний основой проекта, но полный
трекер отложен. Без журнала с первого дня через три месяца был бы рабочий
сервер и ноль истории, а вопрос «работает ли индекс» остался бы открытым ещё
на три месяца.

Поле версии формулы обязательно: без него после первой калибровки весов
история станет несопоставимой и бесполезной.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from . import SQUEEZE_FORMULA_VERSION


class Journal:
    """Дописывает по строке JSONL на каждый расчёт индекса."""

    def __init__(self, path: str, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = threading.Lock()

    def record(
        self,
        symbol: str,
        view: Any,
        *,
        as_of_ms: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Записать результат расчёта.

        Журнал не должен ронять инструмент: если запись невозможна (нет прав,
        нет места), ошибка проглатывается. Потеря строки лога — меньшее зло,
        чем отказ вернуть модели готовый анализ.
        """
        if not self.enabled or view.squeeze_index is None:
            return

        entry = {
            "ts_ms": int(time.time() * 1000),
            "symbol": symbol,
            "interval": view.interval,
            "formula_version": SQUEEZE_FORMULA_VERSION,
            "squeeze_index": view.squeeze_index,
            "components": {k: round(v, 4) for k, v in view.components.items()},
            "excluded": view.excluded,
            "price": view.price,
            "range_low": view.range_low,
            "range_high": view.range_high,
            "atr_pct": round(view.atr_pct, 4),
            "rsi": round(view.rsi_value, 2),
            "bbw_pct_rank": view.bbw.pct_rank,
            "closed_through_ms": view.meta.get("closed_through_ms"),
        }
        if as_of_ms is not None:
            # Ретроспективные расчёты помечаются, иначе они смешаются с живыми
            # и испортят статистику отработки.
            entry["as_of_ms"] = as_of_ms
        if extra:
            entry.update(extra)

        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock, open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
