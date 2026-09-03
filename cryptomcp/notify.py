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
import logging
import os

import httpx

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

        Возвращает признак успеха — он нужен тестам и логу, а вызывающему
        коду решать по нему нечего: повторять на следующем прогоне нельзя,
        уведомление к тому времени устареет.
        """
        if not text:
            return False
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "\n…список обрезан"

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
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
        lines += [f"  + {s} {tf} · ранг {rank}" for s, tf, rank in entered]
    if promoted:
        lines.append(f"\nПодтверждены ({len(promoted)})")
        lines += [f"  ↑ {s} {tf} · ранг {rank}" for s, tf, rank in promoted]
    if exited:
        lines.append(f"\nВышли ({len(exited)})")
        lines += [f"  − {s} {tf} · {reason}" for s, tf, reason in exited]
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
