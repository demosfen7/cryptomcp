"""Binance Futures Market MCP — рыночный контекст для ИИ-анализа.

Реализует модули A1 (data_fetcher) и A2 (metrics) из ТЗ
`crypto-scanner-architecture.md`, плюс расширение §12 (фандинг и открытый
интерес). Подробности — в PLAN.md.

Принцип: агрегаты считает код, структуру читает модель.
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Версия формулы squeeze_index. Пишется в журнал при каждом расчёте: без неё
#: после первой калибровки весов история станет несопоставимой (PLAN §4.11).
SQUEEZE_FORMULA_VERSION = "v1"

from .client import BinanceFuturesClient  # noqa: E402
from .errors import ErrorKind, ToolError  # noqa: E402
from .fetcher import CandleFetcher  # noqa: E402
from .series import Series, build_series, interval_ms  # noqa: E402

__all__ = [
    "BinanceFuturesClient",
    "CandleFetcher",
    "ErrorKind",
    "SQUEEZE_FORMULA_VERSION",
    "Series",
    "ToolError",
    "build_series",
    "interval_ms",
]
