"""Транспорт уведомлений в Telegram (PLAN §4.24).

Уведомляется дельта списка наблюдения — вошедшие, вышедшие и подтверждённые
монеты. Сам список ведёт `collector.update_watchlist`, здесь только доставка.

Два правила определяют весь модуль.

1. **Отказ доставки не должен ронять прогон.** Сборщик, потерявший час,
   теряет час открытого интереса навсегда: раздел `/futures/data/` отдаёт
   только последние 30 суток. Недоставленное уведомление не стоит ничего —
   дельта останется в базе и в логе. Поэтому `send` не поднимает исключений
   вообще, а вызывается уже ПОСЛЕ коммита, когда данные на диске.
2. **Молчание, когда сказать нечего.** Прогон идёт каждый час, а состав
   списка меняется куда реже. Ежечасное «изменений нет» превратило бы канал
   в шум, который перестают читать, — и настоящий вход в нём потерялся бы.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import html
import logging
import os
from urllib.parse import quote

import httpx

from .markets import MARKETS, market_short
from .series import interval_ms

log = logging.getLogger("cryptomcp.notify")

#: Потолок длины сообщения у Telegram — 4096 символов. Берём с запасом под
#: хвостовую строку об усечении.
MAX_TEXT = 3900

#: Попыток доставки. Две, а не одна: сеть моргает чаще, чем Telegram лежит.
#: Больше смысла нет — уведомление ценно, пока свежее, а следующий прогон
#: через час всё равно принесёт актуальный состав.
ATTEMPTS = 2

#: Пауза между попытками. Короткая: прогон сборщика ждать не должен.
RETRY_DELAY_S = 2.0

TIMEOUT_S = 10.0

#: Разметка сообщений. HTML, а не MarkdownV2: у MarkdownV2 экранировать нужно
#: полтора десятка символов в любом месте текста, включая «·», «−» и точку в
#: числах, — цена ошибки там 400 и полностью потерянное уведомление.
PARSE_MODE = "HTML"

#: Как символ Binance называется на TradingView. Суффикс «.P» — перпетуал,
#: без суффикса — спот: BINANCE:ONDOUSDT и BINANCE:ONDOUSDT.P это два разных
#: графика с разными экстремумами. Рынок берётся из самой записи, а не
#: назначается константой: архив предпочитает спот (там глубже история), и на
#: HOMEUSDT 4h ряды разошлись вдвое — узк 17 по споту против 36 по фьючерсу на
#: одной и той же свече. Ссылка обязана вести туда, где монету отобрали.
TV_EXCHANGE = "BINANCE"

#: Суффикс тикера берётся из описания рынка, а не задаётся здесь второй раз:
#: `.P` уже объявлен в markets.FUTURES. Неизвестный рынок (пустое поле у
#: старых эпизодов) ведёт на спотовый тикер — он существует у любой пары, а
#: перпетуала может не быть вовсе.

#: Таймфрейм в параметрах TradingView: минуты числом, дневки и старше — буквой.
#: Чего нет в таблице, то и не передаём — график откроется на своём умолчании,
#: это лучше отвергнутой ссылки.
TV_INTERVALS = {"1h": "60", "4h": "240", "1d": "D", "1w": "W"}


def tradingview_url(
    symbol: str, tf: str | None = None, market: str | None = None
) -> str:
    """Ссылка на график того рынка, по которому монета отобрана."""
    described = MARKETS.get(market or "")
    suffix = described.suffix if described else ""
    query = f"symbol={quote(f'{TV_EXCHANGE}:{symbol}{suffix}')}"
    interval = TV_INTERVALS.get((tf or "").lower())
    if interval:
        query += f"&interval={interval}"
    return f"https://www.tradingview.com/chart/?{query}"


def _head(entry: dict) -> str:
    """Начало строки: тикер хэштегом, таймфрейм, рынок.

    Хэштег, а не ссылка: нажатие на него в Telegram показывает ВСЕ сообщения
    канала с этой монетой, то есть всю её историю входов, подтверждений и
    выходов одним касанием. Ссылка на график из строки не исчезает — она
    уезжает в конец отдельным словом, потому что тикер может быть либо
    хэштегом, либо ссылкой, но не тем и другим сразу: обёрнутый в тег <a>
    хэштег перестаёт быть хэштегом.
    """
    tf, market = entry.get("tf"), entry.get("market")
    return f"{_tag(entry['symbol'])} {tf} {market_short(market)}"


def _tag(symbol: str) -> str:
    """Тикер хэштегом для поиска по каналу."""
    return f"#{html.escape(str(symbol))}"


def _chart(entry: dict) -> str:
    """Ссылка на график отдельным словом в конце строки."""
    return " · " + _link(
        entry["symbol"], entry.get("tf"), entry.get("market"), label="график"
    )


def _squeeze(entry: dict) -> str:
    """Длительность сжатия свечами и календарём.

    Календарь обязателен: «узк 19» на 4h выглядит внушительнее, чем «узк 7»
    на 1d, хотя это 3.2 суток против семи, а в одном сообщении строки обоих
    таймфреймов стоят рядом.
    """
    bars = entry.get("narrow_bars")
    if bars is None:
        return ""
    days = int(bars) * interval_ms(entry["tf"]) / 86_400_000
    return f" · узк {bars} ({days:.1f} сут)"


def _link(
    symbol: str,
    tf: str | None = None,
    market: str | None = None,
    *,
    label: str | None = None,
) -> str:
    """Ссылка на график. Экранируется и текст, и адрес."""
    return (
        f'<a href="{html.escape(tradingview_url(symbol, tf, market), quote=True)}">'
        f"{html.escape(str(label or symbol))}</a>"
    )


def _trim(text: str) -> str:
    """Укоротить до лимита Telegram, не разрезая строку и не разрывая тег.

    Обрезка по символу поделила бы пополам `<a href="…">` — на такое Telegram
    отвечает 400, и уведомление теряется целиком вместо того, чтобы прийти
    неполным. Поэтому режем по строкам: каждая строка — отдельная монета,
    хвост списка потерять не так дорого, как всё сообщение.
    """
    if len(text) <= MAX_TEXT:
        return text
    tail = "\n…список обрезан"
    kept: list[str] = []
    used = 0
    for line in text.split("\n"):
        if used + len(line) + 1 > MAX_TEXT - len(tail):
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept) + tail


class Telegram:
    """Отправитель. Создаётся из окружения; без секретов — не создаётся."""

    def __init__(self, token: str, chat_id: str, *, timeout: float = TIMEOUT_S) -> None:
        self._token = token
        self._chat_id = chat_id
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> Telegram | None:
        """Собрать из переменных окружения или вернуть None.

        Отсутствие секретов — штатный режим, а не ошибка: так сборщик
        запускается локально и на машине разработчика, где бота нет. Ровно
        тем же гейтом устроен шаг деплоя в `.github/workflows/deploy.yml`.
        """
        token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
        if not token or not chat_id:
            return None
        return cls(token, chat_id)

    async def send(self, text: str) -> bool:
        """Отправить сообщение. Никогда не поднимает исключение.

        Текст трактуется как HTML: всё, что приходит извне — символ, причина
        выхода, — вызывающий код обязан пропустить через `html.escape`, иначе
        Telegram ответит 400 и сообщение пропадёт целиком. Разметку собирает
        `render_watchlist_delta`.

        Возвращает признак успеха — он нужен тестам и логу, а вызывающему
        коду решать по нему нечего: повторять на следующем прогоне нельзя,
        уведомление к тому времени устареет.
        """
        if not text:
            return False
        text = _trim(text)

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": PARSE_MODE,
            "disable_web_page_preview": True,
        }
        for attempt in range(1, ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as http:
                    response = await http.post(url, json=payload)
                if response.status_code == 200:
                    return True
                # Токен и chat id повторять бессмысленно: 401/403/400 —
                # это конфигурация, а не сеть. Пишем в лог ответ биржи
                # целиком: без него «не отправилось» неотличимо от «бот не
                # добавлен в группу».
                log.warning(
                    "telegram ответил %s: %s", response.status_code, response.text[:300]
                )
                if response.status_code < 500:
                    return False
            except Exception as error:  # noqa: BLE001 — доставка не роняет прогон
                log.warning("telegram недоступен (попытка %d): %s", attempt, error)
            if attempt < ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY_S)
        return False


def render_watchlist_delta(
    changes: dict[str, list], now_ms: int | None = None
) -> str | None:
    """Собрать текст по дельте или вернуть None, если писать не о чем.

    Порядок разделов — по убыванию новизны: вход это новость, подтверждение
    ранга — уточнение, выход — закрытие темы.

    Записи дельты — словари: полей стало пять, и позиционное чтение кортежа
    на пятом поле ошибается молча.

    Тикер — хэштег: нажатие собирает по каналу все сообщения об этой монете,
    то есть её историю входов и выходов. Ссылка на график TradingView в том же
    таймфрейме и на том же рынке стоит в конце строки словом «график»: иначе
    между «пришло уведомление» и «вижу свечи» стоит ручной поиск тикера, а в
    момент входа ценна как раз скорость. Одним элементом обе роли не
    закрываются — хэштег внутри тега <a> перестаёт быть хэштегом.

    Рынок печатается и словом — «перп» или «спот»: на HOMEUSDT ряды
    разошлись вдвое (узк 17 против 36), и молча выдавать один за другой
    нельзя. Результат — HTML, и уходит он только через `Telegram.send` с
    parse_mode=HTML.
    """
    entered = changes.get("entered") or []
    promoted = changes.get("promoted") or []
    exited = changes.get("exited") or []
    if not (entered or promoted or exited):
        return None

    now_ms = now_ms or int(dt.datetime.now(dt.UTC).timestamp() * 1000)
    stamp = dt.datetime.fromtimestamp(now_ms / 1000, dt.UTC).strftime("%d.%m %H:%M")

    lines = [f"Список наблюдения · {stamp} UTC"]
    if entered:
        lines.append(f"\nВошли ({len(entered)})")
        lines += [
            f"  + {_head(e)} · ранг {e['rank']}{_squeeze(e)}{_chart(e)}"
            for e in entered
        ]
    if promoted:
        lines.append(f"\nПодтверждены ({len(promoted)})")
        lines += [
            f"  ↑ {_head(e)} · ранг {e['rank']}{_squeeze(e)}{_chart(e)}"
            for e in promoted
        ]
    if exited:
        lines.append(f"\nВышли ({len(exited)})")
        lines += [
            f"  − {_head(e)} · {html.escape(str(e['reason']))}{_chart(e)}"
            for e in exited
        ]
    return "\n".join(lines)


async def notify_watchlist(
    changes: dict[str, list],
    now_ms: int | None = None,
    sender: Telegram | None = None,
) -> bool:
    """Уведомить о дельте, если есть о чём и есть куда."""
    text = render_watchlist_delta(changes, now_ms)
    if text is None:
        return False
    sender = sender or Telegram.from_env()
    if sender is None:
        log.debug("telegram не настроен: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID пусты")
        return False
    return await sender.send(text)
