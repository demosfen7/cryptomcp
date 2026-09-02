"""Чтение рядов из архива с догрузкой свежего хвоста (PLAN §4.15).

Смысл не в том, чтобы перестать ходить в биржу, а в том, чтобы не выкачивать
одну и ту же историю по десять раз в день. Снапшот стоил ~126 единиц веса, из
которых почти всё — постраничная догрузка шестидесяти суток ради перцентилей.
Эти шестьдесят суток уже лежат в базе и меняются раз в час.

Два правила, нарушение которых обесценивает всю затею:

1. **Архив даёт историю, биржа — хвост.** Сборщик ходит раз в час, а свеча 1h
   закрывается тоже раз в час, но не в ту же минуту. Отдавать выдачу прямо из
   базы значило бы иногда отставать на час — то есть менять качество ответа на
   экономию запросов. Поэтому хвост после последней архивной свечи всегда
   догружается: это один запрос вместо десятка страниц.

2. **Источник обязан совпадать.** В архиве у символа один рынок (спот там, где
   он есть, иначе фьючерс). Запрос спотового ряда по монете, заархивированной с
   фьючерса, обязан идти в биржу, а не склеивать два рынка в один ряд.

Ретроспектива (``as_of_ms``) — тот случай, ради которого архив и заводился:
хвост не нужен вовсе, и запрос не стоит ничего. Массовый прогон по истории для
калибровки становится вопросом процессорного времени, а не недель ожидания.
"""

from __future__ import annotations

import logging
import os
import sqlite3

from . import storage
from .analysis import required_candles
from .errors import insufficient_history
from .fetcher import CandleFetcher
from .series import Series, build_series, series_from_records

log = logging.getLogger("cryptomcp.reader")

#: Сколько свечей просить у биржи на хвост. С запасом: между прогонами
#: сборщика набирается одна-две, но после его простоя может и десяток.
TAIL_LIMIT = 200

#: Запас к каноническому окну: индикаторам нужен разогрев, а EMA200 — двести
#: свечей до первого значения.
WARMUP = 250


def archive_path() -> str | None:
    """Путь к базе, если она вообще есть.

    Локально сборщика может не быть вовсе, и тогда сервер обязан работать как
    работал — просто из биржи, без единого упоминания архива в выдаче.
    """
    if os.environ.get("CRYPTOMCP_USE_ARCHIVE", "1") == "0":
        return None
    path = storage.DEFAULT_PATH
    return path if os.path.exists(path) else None


class ArchiveReader:
    """Ряды из архива, дополненные свежим хвостом с биржи."""

    def __init__(self, fetcher: CandleFetcher, market: str, path: str | None = None):
        self._fetcher = fetcher
        self._market = market
        self._path = path if path is not None else archive_path()

    async def get(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        as_of_ms: int | None = None,
        min_candles: int | None = None,
        target_span_days: float | None = None,
        max_pages: int = 12,
    ) -> Series:
        """Тот же контракт, что у CandleFetcher.get.

        Возвращается КАНОНИЧЕСКОЕ окно — столько свечей, сколько нужно
        метрикам, независимо от того, сколько их пришло. Это и делает два
        источника взаимозаменяемыми: без обрезки архив отдавал бы на несколько
        свечей больше биржи, и сезонная база объёма расходилась бы в третьем
        знаке — ровно тот сорт расхождения, который уже ловили в колонке
        объёма get_klines.
        """
        window = required_candles(interval) + WARMUP
        series = await self._from_archive(symbol, interval, window, as_of_ms)
        if series is None:
            series = await self._fetcher.get(
                symbol, interval, limit=max(limit, window), as_of_ms=as_of_ms,
                target_span_days=target_span_days, max_pages=max_pages,
            )
        series = series.tail(window)
        if min_candles is not None and len(series) < min_candles:
            raise insufficient_history(symbol, interval, len(series), min_candles)
        return series

    async def _from_archive(
        self, symbol: str, interval: str, window: int, as_of_ms: int | None
    ) -> Series | None:
        if self._path is None:
            return None
        try:
            con = storage.connect(self._path, read_only=True)
        except sqlite3.Error as error:  # база занята или повреждена — не беда
            log.warning("архив недоступен: %s", error)
            return None
        try:
            if storage.archive_source(con, symbol) != self._market:
                return None
            records = storage.load_candles(
                con, symbol, interval, window, before_ms=as_of_ms
            )
        except sqlite3.Error as error:
            log.warning("архив не прочитан (%s %s): %s", symbol, interval, error)
            return None
        finally:
            con.close()

        # Неполное окно — это уже другой ряд, и брать его вместо биржевого
        # нельзя: метрики посчитаются по более короткой базе и разойдутся.
        if len(records) < window:
            return None
        if as_of_ms is not None:
            # Хвост не нужен: всё, что после запрошенного момента, в расчёт и
            # не должно попадать. Ретроспектива не стоит ни одного запроса.
            return series_from_records(records, symbol, interval)

        return await self._with_tail(symbol, interval, records)

    async def _with_tail(
        self, symbol: str, interval: str, records: list[tuple]
    ) -> Series:
        """Дописать свечи, закрывшиеся после последней архивной."""
        last_open = int(records[-1][0])
        fresh: list[tuple] = []
        try:
            client = self._fetcher.client
            raw = await client.klines(
                symbol, interval, limit=TAIL_LIMIT, start_time=last_open
            )
            now_ms = await client.now_ms()
            tail = build_series(raw, symbol, interval, now_ms)
            if len(tail):
                frame = tail.df
                columns = (
                    "open_time", "open", "high", "low", "close", "volume_base",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
                )
                fresh = list(zip(*(frame[c].tolist() for c in columns), strict=True))
        except Exception as error:  # noqa: BLE001 — хвост не критичен
            # Архив без хвоста лучше, чем отказ: он отстанет максимум на час,
            # и это видно по штампу «закрыты по» в самой выдаче.
            log.warning("хвост не догружен (%s %s): %s", symbol, interval, error)

        merged = {int(row[0]): row for row in records}
        # Свежая свеча важнее архивной: биржа правит последнюю после закрытия.
        merged.update({int(row[0]): row for row in fresh})
        ordered = [merged[key] for key in sorted(merged)]
        return series_from_records(ordered, symbol, interval)


