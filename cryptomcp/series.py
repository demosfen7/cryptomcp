"""Нормализация свечей и отсечение незакрытой (PLAN §6.1, §6.2).

Незакрытая свеча имеет неполные объём и диапазон, поэтому любая метрика поверх
неё смещена. ТЗ называет это дважды (§4.1 и §10) как главную причину, по которой
подобные системы врут. Отсечение сделано здесь, на входе, а не в индикаторах:
так ни один расчёт физически не может увидеть незакрытую свечу.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MINUTE_MS = 60_000

#: Запас после закрытия свечи, прежде чем считать её пригодной.
#:
#: Закрытость определяется по часам БИРЖИ, а не по локальным — расхождение
#: измерялось в 1552 мс. Но одной синхронизации мало: биржа иногда доправляет
#: последнюю свечу в первые мгновения после закрытия, а смещение часов между
#: пересинхронизациями плывёт. Пять секунд закрывают оба случая и ничего не
#: стоят: задача не высокочастотная, свеча 1m обновляется раз в минуту.
CLOSE_GRACE_MS = 5_000

#: Длительность интервала в миллисекундах. 1M намеренно отсутствует: длина
#: месяца непостоянна, и проверка непрерывности по фиксированному шагу для него
#: неприменима.
INTERVAL_MS: dict[str, int] = {
    "1m": MINUTE_MS,
    "3m": 3 * MINUTE_MS,
    "5m": 5 * MINUTE_MS,
    "15m": 15 * MINUTE_MS,
    "30m": 30 * MINUTE_MS,
    "1h": 60 * MINUTE_MS,
    "2h": 120 * MINUTE_MS,
    "4h": 240 * MINUTE_MS,
    "6h": 360 * MINUTE_MS,
    "8h": 480 * MINUTE_MS,
    "12h": 720 * MINUTE_MS,
    "1d": 1440 * MINUTE_MS,
    "3d": 3 * 1440 * MINUTE_MS,
    "1w": 7 * 1440 * MINUTE_MS,
}

#: Порядок колонок в ответе /fapi/v1/klines.
_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "_ignore",
]

_NUMERIC = [
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]


def interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(
            f"Неподдерживаемый интервал {interval!r}. "
            f"Доступны: {', '.join(INTERVAL_MS)}"
        )
    return INTERVAL_MS[interval]


@dataclass(frozen=True)
class Series:
    """Ряд ЗАКРЫТЫХ свечей одного символа и таймфрейма."""

    symbol: str
    interval: str
    df: pd.DataFrame
    #: Время закрытия последней свечи (мс, UTC).
    closed_through_ms: int
    #: Сколько незакрытых свечей отброшено на входе. Обычно 0 или 1.
    dropped_unclosed: int
    #: Число пропущенных свечей — сумма недостающих шагов между соседями.
    missing: int

    def __len__(self) -> int:
        return len(self.df)

    @property
    def span_days(self) -> float:
        """Календарный охват ряда в сутках — для правила §4.2."""
        if len(self.df) < 2:
            return 0.0
        first = int(self.df["open_time"].iloc[0])
        last = int(self.df["close_time"].iloc[-1])
        return (last - first) / 86_400_000

    @property
    def has_gaps(self) -> bool:
        return self.missing > 0

    def col(self, name: str) -> np.ndarray:
        return self.df[name].to_numpy(dtype=float)

    @property
    def close(self) -> np.ndarray:
        return self.col("close")

    @property
    def high(self) -> np.ndarray:
        return self.col("high")

    @property
    def low(self) -> np.ndarray:
        return self.col("low")

    @property
    def quote_volume(self) -> np.ndarray:
        return self.col("quote_volume")

    @property
    def taker_buy_ratio(self) -> np.ndarray:
        """Доля агрессивных покупок в quote-объёме (PLAN §4.5)."""
        total = self.quote_volume
        taker = self.col("taker_buy_quote")
        return np.divide(
            taker, total, out=np.full_like(total, np.nan), where=total > 0
        )

    def tail(self, n: int) -> Series:
        if n >= len(self.df):
            return self
        return Series(
            symbol=self.symbol,
            interval=self.interval,
            df=self.df.iloc[-n:].reset_index(drop=True),
            closed_through_ms=self.closed_through_ms,
            dropped_unclosed=self.dropped_unclosed,
            missing=count_missing(self.df.iloc[-n:], self.interval),
        )

    def meta(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "candles": len(self.df),
            "span_days": round(self.span_days, 2),
            "closed_through_ms": self.closed_through_ms,
            "dropped_unclosed": self.dropped_unclosed,
            "missing": self.missing,
        }


def count_missing(df: pd.DataFrame, interval: str) -> int:
    """Сколько свечей отсутствует внутри ряда.

    Приостановки торгов и делистинги оставляют дыры, на которых индикаторы
    молча считают чушь (PLAN §6.2).
    """
    if len(df) < 2:
        return 0
    step = interval_ms(interval)
    diffs = np.diff(df["open_time"].to_numpy(dtype="int64"))
    # Каждый шаг длиной k*step вместо step означает k-1 пропущенных свечей.
    steps = np.rint(diffs / step).astype("int64")
    return int(np.clip(steps - 1, 0, None).sum())


def series_from_records(
    records: Sequence[Sequence[Any]], symbol: str, interval: str
) -> Series:
    """Ряд из архива: (ts, o, h, l, c, volume, quote_volume, trades, taker×2).

    Отсечения незакрытой свечи здесь нет и не нужно: в архив она не попадает
    по построению — загрузчик пропускает её тем же кодом, что и онлайновый
    путь. Проверка непрерывности остаётся: дыры в архиве возможны, если
    символ не торговался.
    """
    interval_ms(interval)
    columns = [
        "open_time", "open", "high", "low", "close", "volume_base",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    ]
    if not records:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
        return Series(symbol.upper(), interval, empty, 0, 0, 0)

    df = pd.DataFrame(list(records), columns=columns)
    df["open_time"] = df["open_time"].astype("int64")
    df["trades"] = df["trades"].fillna(0).astype("int64")
    df["close_time"] = df["open_time"] + interval_ms(interval) - 1
    for column in ("open", "high", "low", "close", "volume_base",
                   "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    return Series(
        symbol=symbol.upper(),
        interval=interval,
        df=df,
        closed_through_ms=int(df["close_time"].iloc[-1]),
        dropped_unclosed=0,
        missing=count_missing(df, interval),
    )


def build_series(
    raw: Sequence[Sequence[Any]],
    symbol: str,
    interval: str,
    now_ms: int,
    *,
    grace_ms: int = CLOSE_GRACE_MS,
) -> Series:
    """Собрать ряд закрытых свечей из ответа /fapi/v1/klines.

    ``now_ms`` обязателен и должен приходить с часов биржи: определять
    закрытость по локальным часам нельзя, расхождение в минуту меняет результат.

    ``grace_ms`` — запас после закрытия, см. CLOSE_GRACE_MS.
    """
    interval_ms(interval)  # валидация интервала до любой работы

    if not raw:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in _COLUMNS[:-1]})
        return Series(symbol.upper(), interval, empty, 0, 0, 0)

    df = pd.DataFrame(raw, columns=_COLUMNS).drop(columns=["_ignore"])
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["trades"] = df["trades"].astype("int64")
    for column in _NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    before = len(df)
    # Свеча закрыта, когда время биржи прошло её close_time плюс запас.
    # close_time = open_time + step - 1, поэтому сравнение строгое.
    df = df[df["close_time"] < now_ms - grace_ms]
    dropped = before - len(df)

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    if df.empty:
        return Series(symbol.upper(), interval, df, 0, dropped, 0)

    return Series(
        symbol=symbol.upper(),
        interval=interval,
        df=df,
        closed_through_ms=int(df["close_time"].iloc[-1]),
        dropped_unclosed=dropped,
        missing=count_missing(df, interval),
    )
