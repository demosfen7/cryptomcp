"""Структурированные ошибки инструментов (PLAN §6.5).

Инструмент MCP не должен выбрасывать исключение наружу: модель получит сырой
traceback вместо понятного ответа и, скорее всего, зациклится на повторах.
Вместо этого каждый инструмент возвращает payload с полями kind / message /
retryable, по которым модель может принять осмысленное решение.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorKind(StrEnum):
    """Типы ошибок, различимые моделью."""

    UNKNOWN_SYMBOL = "unknown_symbol"
    BAD_PARAMS = "bad_params"
    RATE_LIMITED = "rate_limited"
    IP_BANNED = "ip_banned"
    #: Не сбой, а нормальный ответ для свежего листинга.
    INSUFFICIENT_HISTORY = "insufficient_history"
    DATA_GAP = "data_gap"
    UPSTREAM_ERROR = "upstream_error"


#: Ошибки, при которых повтор осмыслен. IP_BANNED сюда сознательно не входит:
#: ретраи во время бана продлевают его (PLAN §6.4).
_RETRYABLE = frozenset({ErrorKind.RATE_LIMITED, ErrorKind.UPSTREAM_ERROR})


@dataclass
class ToolError(Exception):
    """Ошибка, которую инструмент отдаёт модели вместо исключения."""

    kind: ErrorKind
    message: str
    retry_after_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "kind": self.kind.value,
                "message": self.message,
                "retryable": self.retryable,
            },
        }
        if self.retry_after_s is not None:
            payload["error"]["retry_after_s"] = round(self.retry_after_s, 3)
        if self.details:
            payload["error"]["details"] = self.details
        return payload


def unknown_symbol(symbol: str, market: str = "futures") -> ToolError:
    """Символа нет на этом рынке — но, возможно, есть на соседнем.

    Списки не совпадают в обе стороны: UAIUSDT торгуется перпетуалом и не
    торгуется на споте, а спотовых пар без фьючерса ещё больше. Подсказка про
    второй рынок избавляет от вывода «монеты нет на Binance».
    """
    other = "spot" if market == "futures" else "futures"
    return ToolError(
        ErrorKind.UNKNOWN_SYMBOL,
        f"Символ {symbol!r} не найден на рынке {market} Binance. "
        f"Часть монет есть только на одном из рынков — попробовать "
        f"market={other!r}.",
        details={"symbol": symbol, "market": market},
    )


def bad_params(message: str, **details: Any) -> ToolError:
    return ToolError(ErrorKind.BAD_PARAMS, message, details=details)


def insufficient_history(
    symbol: str, interval: str, have: int, need: int
) -> ToolError:
    return ToolError(
        ErrorKind.INSUFFICIENT_HISTORY,
        f"{symbol} {interval}: доступно {have} закрытых свечей, требуется {need}. "
        f"Вероятно, свежий листинг — это не сбой.",
        details={"symbol": symbol, "interval": interval, "have": have, "need": need},
    )
