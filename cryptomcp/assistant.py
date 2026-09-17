"""Сценарии бота через Claude: инструменты в процессе, учёт денег (PLAN §4.44).

**Почему не SDK.** `httpx` в проекте уже есть, а обращение к Messages API —
это один POST. Официальный SDK добавил бы зависимость ради удобства, которого
здесь нет: кеширование префикса и разбор `usage` всё равно пишутся руками,
потому что от них зависят деньги.

**Почему инструменты вызываются прямо здесь.** Бот живёт в том же процессе,
что и функции сервера, и MCP-круг по HTTP с OAuth стоил бы сетевого времени
ради вызова той же функции. Клиентские инструменты Claude — это имя, описание
и схема, а описания и схемы у MCP-инструментов уже написаны: берём их из
реестра сервера, чтобы описание в двух местах не разъезжалось.

**Каждой кнопке — только её инструменты.** Описания всех восемнадцати
инструментов занимают около 16.5 тыс. символов, и они оплачиваются в каждом
запросе сценария. Кнопке «стакан у уровня» незачем платить за описание
списка наблюдения.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("cryptomcp.assistant")

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

#: Модель выбрана владельцем 17.09.2026: разборы Haiku получались пересказом
#: цифр, Opus стоил бы втрое дороже при том же объёме ответа.
MODEL = os.environ.get("BOT_CLAUDE_MODEL", "claude-sonnet-5")

#: Ответ в одно сообщение Telegram — это около 2000 символов; дальше модель
#: начинает пересказывать сама себя.
MAX_TOKENS = 2000

#: Потолок кругов «модель просит инструмент — мы отвечаем». Шести хватает на
#: самый длинный сценарий (проверка перед сделкой — пять инструментов), а
#: зацикливание на инструменте перестаёт стоить денег.
MAX_ROUNDS = 6

#: Цены Sonnet 5, $ за миллион токенов. Сняты со страницы
#: platform.claude.com/docs/en/about-claude/pricing 17.09.2026. Стоимость
#: считается по фактическому `usage` ответа, а не по этой оценке, — таблица
#: нужна только чтобы перевести токены в деньги.
PRICE_INPUT = 2.0
PRICE_CACHE_WRITE = 2.5
PRICE_CACHE_READ = 0.2
PRICE_OUTPUT = 10.0

#: Столько стоил самый дорогой сценарий на замере 17.09.2026 (4–7¢) с запасом.
#: Нужна до запроса: фактическую цену узнаём только после ответа, а решать,
#: пускать ли сценарий в дневной лимит, надо заранее.
SCENARIO_ESTIMATE_USD = 0.10

SYSTEM_PROMPT = """Ты — аналитик рынка криптовалют в Telegram-боте. Тебе дают инструменты с
данными Binance. Отвечаешь человеку, который не хочет тонуть в цифрах.

Правила ответа:
1. Первая строка после заголовка — «Коротко:» и вывод одной-двумя фразами.
2. Каждое число сравнивай человеческими словами: «уже, чем в 95% дней за год»,
   «в 5 раз ниже среднего за 100 дней», «держал 5 раз».
3. Не используй слова: перцентиль, pct, ATR, BBW, EMA, RSI, POC, Value Area,
   имбаланс, OI, squeeze, дельта, квадрант, n/a. Объясняй смысл, а не название
   индикатора.
4. Не больше 7–8 чисел на весь ответ. Цены уровней — да, промежуточные расчёты
   — нет.
5. Тикеры без USDT. Время данных — в заголовке, UTC.
6. Никогда не пиши «купи», «продай», «вход», «цель», «стоп», не давай
   вероятностей роста. Говори «готовность к движению», «что смотреть»,
   «уровни выхода из коридора». Направление называй, только если данные его
   действительно показывают, и прямо говори, насколько слабый это намёк.
7. Если данных не хватает — так и скажи словами, не выдумывай.
8. Ответ до 2000 символов. Разметка Telegram HTML: только <b>. Эмодзи — только
   метки разделов из шаблона.
9. Когда два признака противоречат друг другу, скажи об этом, а не выбирай
   удобный."""


@dataclass(frozen=True)
class Scenario:
    """Кнопка: свой набор инструментов и своя задача модели."""

    name: str
    title: str
    tools: tuple[str, ...]
    instruction: str
    needs_symbol: bool = False


SCENARIOS: dict[str, Scenario] = {
    "analyse": Scenario(
        name="analyse",
        title="разбор",
        tools=(
            "get_market_snapshot", "get_squeeze_metrics",
            "get_key_levels", "get_derivatives", "get_scan_history",
        ),
        needs_symbol=True,
        instruction=(
            "Разбери монету {symbol}. Разделы строго в таком порядке и с этими "
            "заголовками: первая строка «🔍 {short} · разбор · {stamp}», затем "
            "«Коротко:», «📍 Где цена», «😴 Насколько тихо», «💰 Что делают "
            "деньги», «👀 Что смотреть». В разделе про цену назови ближайшие "
            "уровни сверху и снизу с ценами и тем, сколько раз они держали."
        ),
    ),
    "before": Scenario(
        name="before",
        title="проверка перед сделкой",
        tools=(
            "get_market_snapshot", "get_key_levels",
            "get_order_book", "get_derivatives",
        ),
        needs_symbol=True,
        instruction=(
            "Проверь {symbol} перед сделкой: не поздно ли и где подвох. Первая "
            "строка «✅ {short} · проверка · {stamp}», затем «Коротко:» и "
            "светофор 🟢🟡🔴 по пяти пунктам, каждый одной строкой: цена уже "
            "ушла? · ближайшие уровни · стакан у этих уровней · фандинг · "
            "подтверждает ли спот. Для спота вызови снимок с market='spot'."
        ),
    ),
    "morning": Scenario(
        name="morning",
        title="утренний обзор",
        tools=("get_watchlist", "get_accumulation", "get_scan_history"),
        instruction=(
            "Сделай утренний обзор: на что смотреть сегодня. Первая строка "
            "«🌅 Утренний обзор · {stamp}», затем «Коротко:», список из 3–5 "
            "монет по одной строке «почему» и раздел «Что изменилось за ночь». "
            "Бери монеты из обоих списков — наблюдения и накопления, — и "
            "скажи, из какого каждая."
        ),
    ),
    "quiet": Scenario(
        name="quiet",
        title="тихий набор",
        tools=("get_accumulation", "scan_pairs", "get_derivatives"),
        instruction=(
            "Найди монеты, где кто-то набирает позицию без движения цены. "
            "Первая строка «🕵 Тихий набор · {stamp}», затем «Коротко:», до "
            "пяти монет с объяснением, чем это видно, и честная строка, у "
            "скольких из них деньги в контрактах подтверждают набор. Начни со "
            "второго списка сканера (get_accumulation), при нехватке добери "
            "через scan_pairs с timeframe='4h', min_vol_ratio=1.5, "
            "max_abs_change_window=3."
        ),
    ),
    "history": Scenario(
        name="history",
        title="разбор задним числом",
        tools=(
            "get_market_snapshot", "get_squeeze_metrics",
            "get_derivatives", "get_klines", "get_scan_history",
        ),
        needs_symbol=True,
        instruction=(
            "Разбери {symbol} задним числом: как монета выглядела ПЕРЕД "
            "движением. Момент прошлого передай инструментам параметром "
            "as_of_ms = {as_of_ms}. Первая строка «⏪ {short} · задним числом "
            "· {as_of_stamp}», затем «Коротко:», «Было ли затишье», «Что "
            "делали деньги», «Видел ли это сканер»."
        ),
    ),
    "weekly": Scenario(
        name="weekly",
        title="итоги недели",
        tools=("get_watchlist", "get_accumulation"),
        instruction=(
            "Подведи итоги недели по спискам сканера. Первая строка "
            "«📈 Итоги недели · {stamp}», затем «Коротко:», сколько монет вышло "
            "из коридора и в какую сторону, сколько просто выпало, лучшая и "
            "худшая монета. Списки бери со status='all'."
        ),
    ),
    "explain": Scenario(
        name="explain",
        title="объяснение",
        tools=(),
        instruction=(
            "Объясни человеку, что означают эти числа, за 3–5 строк. Первая "
            "строка «🧠 Что это значит». Не выдумывай того, чего в них нет, и "
            "не делай вывода о намерениях участников.\n\n{payload}"
        ),
    ),
}


@dataclass
class Usage:
    """Токены ответа по видам — из них считаются деньги."""

    input_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0

    def add(self, payload: dict[str, Any]) -> None:
        self.input_tokens += int(payload.get("input_tokens") or 0)
        self.cache_creation_tokens += int(payload.get("cache_creation_input_tokens") or 0)
        self.cache_read_tokens += int(payload.get("cache_read_input_tokens") or 0)
        self.output_tokens += int(payload.get("output_tokens") or 0)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT
            + self.cache_creation_tokens * PRICE_CACHE_WRITE
            + self.cache_read_tokens * PRICE_CACHE_READ
            + self.output_tokens * PRICE_OUTPUT
        ) / 1_000_000


@dataclass
class Answer:
    """Ответ сценария: текст человеку и цена этого текста."""

    text: str
    usage: Usage = field(default_factory=Usage)
    rounds: int = 0
    truncated: bool = False

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd


class AssistantError(RuntimeError):
    """Claude не ответил. Сообщение уже пригодно для показа человеку."""


async def tool_definitions(names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Описания инструментов берутся из реестра сервера, а не пишутся заново.

    Иначе описание инструмента существовало бы в двух местах и разъехалось на
    первой же правке — а именно оно объясняет модели, что означают числа.
    """
    if not names:
        return []
    from .server import server as mcp_server

    wanted = set(names)
    tools = []
    for tool in await mcp_server.list_tools():
        if tool.name in wanted:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
    missing = wanted - {tool["name"] for tool in tools}
    if missing:
        raise AssistantError(f"инструменты не найдены: {', '.join(sorted(missing))}")
    return tools


async def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Вызвать инструмент в этом же процессе.

    Ошибку инструмента отдаём модели текстом, а не исключением: инструменты
    проекта и так возвращают структурированный отказ вместо трейсбека (§6.5),
    и модель обязана увидеть причину, чтобы сказать о ней человеку.
    """
    from . import server

    function = getattr(server, name, None)
    if function is None:
        return f"инструмент {name} недоступен"
    try:
        return str(await function(**arguments))
    except Exception as error:  # noqa: BLE001 - модель должна узнать причину
        log.warning("инструмент %s не отработал: %s", name, error)
        return f"инструмент {name} не отработал: {error}"


class Assistant:
    """Один сценарий — один цикл вызовов инструментов и один счёт за него."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = MODEL,
        http: Any | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._http = http
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> Assistant | None:
        """Без ключа ассистента нет, и это штатный режим, а не ошибка."""
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        return cls(key) if key else None

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        if self._http is not None:
            return await self._http(payload, headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            raise AssistantError(
                f"Claude ответил {response.status_code}: {response.text[:200]}"
            )
        return dict(response.json())

    async def run(self, scenario: Scenario, **params: Any) -> Answer:
        """Провести сценарий: инструменты, ответ и цена по фактическому usage."""
        tools = await tool_definitions(scenario.tools)
        # Кешируется постоянная часть запроса: инструкция и описания
        # инструментов. Внутри сценария идут 2–3 обращения, и повтор префикса
        # стоит десятую часть цены.
        system = [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]
        if tools:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": scenario.instruction.format(**params)}
        ]
        usage = Usage()
        for round_number in range(1, MAX_ROUNDS + 1):
            payload: dict[str, Any] = {
                "model": self._model,
                "max_tokens": MAX_TOKENS,
                "system": system,
                # Копия, а не сам список: он растёт по ходу сценария, и
                # отправленный запрос не должен задним числом меняться.
                "messages": list(messages),
            }
            if tools:
                payload["tools"] = tools
            body = await self._post(payload)
            usage.add(dict(body.get("usage") or {}))
            content = list(body.get("content") or [])
            messages.append({"role": "assistant", "content": content})
            calls = [item for item in content if item.get("type") == "tool_use"]
            if not calls:
                text = "\n".join(
                    str(item.get("text", "")) for item in content
                    if item.get("type") == "text"
                ).strip()
                return Answer(text=text, usage=usage, rounds=round_number)
            results = await asyncio.gather(*(
                call_tool(str(call.get("name")), dict(call.get("input") or {}))
                for call in calls
            ))
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.get("id"),
                        "content": result,
                    }
                    for call, result in zip(calls, results, strict=True)
                ],
            })
        # Круги кончились: честно говорим об этом, а не выдаём обрывок за ответ.
        return Answer(
            text=(
                "Не успел собрать ответ: данных запрошено больше, чем "
                "помещается в один разбор. Попробуйте ещё раз или спросите "
                "конкретнее."
            ),
            usage=usage,
            rounds=MAX_ROUNDS,
            truncated=True,
        )


def cost_words(value: float) -> str:
    """Стоимость в центах; ниже цента — «<1¢», иначе сотые доллара не читаются."""
    cents = value * 100
    if cents < 1:
        return "<1¢"
    return f"{cents:.1f}¢"


def scenario_payload(text: str, limit: int = 4000) -> str:
    """Данные шаблона для кнопки «Объяснить», обрезанные по здравому смыслу."""
    return text if len(text) <= limit else text[:limit] + "…"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
