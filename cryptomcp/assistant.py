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

#: Модель выбрана владельцем 25.09.2026: самая дешёвая. До этого стоял Sonnet 5
#: (разборы Haiku получались пересказом цифр) — вернуть его можно переменной
#: BOT_CLAUDE_MODEL=claude-sonnet-5, но тогда поправить и цены ниже.
MODEL = os.environ.get("BOT_CLAUDE_MODEL", "claude-haiku-4-5")

#: Ответ в одно сообщение Telegram — это около 2000 символов; дальше модель
#: начинает пересказывать сама себя.
MAX_TOKENS = 2000

#: Потолок кругов «модель просит инструмент — мы отвечаем». Шести хватает на
#: самый длинный сценарий (проверка перед сделкой — пять инструментов), а
#: зацикливание на инструменте перестаёт стоить денег.
MAX_ROUNDS = 6

#: Цены Haiku 4.5, $ за миллион токенов (platform.claude.com/docs/en/about-claude/pricing).
#: Стоимость считается по фактическому `usage` ответа, а не по этой оценке, —
#: таблица нужна только чтобы перевести токены в деньги.
PRICE_INPUT = 1.0
PRICE_CACHE_WRITE = 1.25
PRICE_CACHE_READ = 0.1
PRICE_OUTPUT = 5.0

#: OpenAI: владелец выбрал gpt-6-luna 25.09.2026 как самую дешёвую. Работает,
#: когда задан OPENAI_API_KEY; без него бот остаётся на Claude.
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.environ.get("BOT_OPENAI_MODEL", "gpt-6-luna")
#: Рассуждение оплачивается как вывод; для пересказа данных инструментов
#: хватает низкого уровня.
OPENAI_REASONING = os.environ.get("BOT_OPENAI_REASONING", "low")

#: Цены gpt-6-luna, $ за миллион токенов (developers.openai.com, 25.09.2026):
#: вход, запись в кеш, чтение из кеша, вывод — тот же порядок, что у Claude.
OPENAI_PRICES = (0.10, 0.125, 0.01, 0.50)
ANTHROPIC_PRICES = (PRICE_INPUT, PRICE_CACHE_WRITE, PRICE_CACHE_READ, PRICE_OUTPUT)

#: Самый дорогой сценарий на Sonnet стоил 4–7¢ (замер 17.09.2026); Haiku вдвое
#: дешевле, оценка оставлена с запасом.
#: Нужна до запроса: фактическую цену узнаём только после ответа, а решать,
#: пускать ли сценарий в дневной лимит, надо заранее.
SCENARIO_ESTIMATE_USD = 0.05

#: Сколько прошлых реплик диалога уходит модели вместе с новой: вопрос и ответ
#: — две реплики, так что это последние семь обменов.
DIALOG_MEMORY = 14

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

#: Свободный разговор: без жёсткого шаблона ответа, но с теми же запретами.
CHAT_SYSTEM_PROMPT = """Ты — помощник по рынку криптовалют в Telegram-боте владельца. С тобой
разговаривают обычными сообщениями; в переписке есть память о последних репликах.
Отвечай по-русски, по существу и коротко — как в мессенджере. На болтовню
отвечай просто, без инструментов. Когда спрашивают про рынок или монету —
бери данные Binance инструментами, а не из головы. Тикер пиши без USDT, в
инструменты передавай с USDT (1000rats → 1000RATSUSDT).

Не используй жаргон индикаторов (перцентиль, ATR, BBW, EMA, RSI, OI, squeeze
и т. п.) — объясняй смысл словами. Никогда не пиши «купи», «продай», «вход»,
«цель», «стоп» и не давай вероятностей роста: говори о готовности к движению
и уровнях. Если данных не хватает — так и скажи. Ответ до 2000 символов,
разметка Telegram HTML: только <b>."""


@dataclass(frozen=True)
class Scenario:
    """Кнопка: свой набор инструментов и своя задача модели."""

    name: str
    title: str
    tools: tuple[str, ...]
    instruction: str
    needs_symbol: bool = False
    system: str = SYSTEM_PROMPT


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
    "chat": Scenario(
        name="chat",
        title="ответ",
        tools=(
            "get_market_snapshot", "get_squeeze_metrics", "get_key_levels",
            "get_derivatives", "get_watchlist", "get_accumulation",
        ),
        system=CHAT_SYSTEM_PROMPT,
        instruction="{payload}",
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
    prices: tuple[float, float, float, float] = ANTHROPIC_PRICES

    def add_openai(self, payload: dict[str, Any]) -> None:
        """Usage Responses API: input_tokens уже включает кешированные и записанные."""
        details = dict(payload.get("input_tokens_details") or {})
        cached = int(details.get("cached_tokens") or 0)
        written = int(details.get("cache_write_tokens") or 0)
        total = int(payload.get("input_tokens") or 0)
        self.input_tokens += max(0, total - cached - written)
        self.cache_read_tokens += cached
        self.cache_creation_tokens += written
        # Токены рассуждения входят в output_tokens и оплачиваются как вывод.
        self.output_tokens += int(payload.get("output_tokens") or 0)

    def add(self, payload: dict[str, Any]) -> None:
        self.input_tokens += int(payload.get("input_tokens") or 0)
        self.cache_creation_tokens += int(payload.get("cache_creation_input_tokens") or 0)
        self.cache_read_tokens += int(payload.get("cache_read_input_tokens") or 0)
        self.output_tokens += int(payload.get("output_tokens") or 0)

    @property
    def cost_usd(self) -> float:
        price_input, price_write, price_read, price_output = self.prices
        return (
            self.input_tokens * price_input
            + self.cache_creation_tokens * price_write
            + self.cache_read_tokens * price_read
            + self.output_tokens * price_output
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
    def from_env(cls) -> Assistant | OpenAIAssistant | None:
        """Ключ OpenAI важнее ключа Claude; без обоих ассистента нет, и это штатно."""
        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if openai_key:
            return OpenAIAssistant(openai_key)
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

    async def run(
        self,
        scenario: Scenario,
        *,
        history: list[dict[str, str]] | None = None,
        **params: Any,
    ) -> Answer:
        """Провести сценарий: инструменты, ответ и цена по фактическому usage.

        `history` — прошлые реплики диалога ({"role", "content"}), только
        тексты: вызовы инструментов в память не попадают, иначе каждая
        реплика тащила бы за собой килобайты данных.
        """
        tools = await tool_definitions(scenario.tools)
        # Кешируется постоянная часть запроса: инструкция и описания
        # инструментов. Внутри сценария идут 2–3 обращения, и повтор префикса
        # стоит десятую часть цены.
        system = [{
            "type": "text",
            "text": scenario.system,
            "cache_control": {"type": "ephemeral"},
        }]
        if tools:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        messages: list[dict[str, Any]] = [
            *(history or []),
            {"role": "user", "content": scenario.instruction.format(**params)},
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


class OpenAIAssistant:
    """Тот же цикл сценария через OpenAI Responses API.

    Запрос не хранится у OpenAI (`store: false`), поэтому рассуждение модели
    приходит зашифрованным и отсылается обратно вместе с остальным выводом:
    без него следующий круг после вызова инструмента не соберётся.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = OPENAI_MODEL,
        http: Any | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._http = http
        self._timeout = timeout

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        if self._http is not None:
            return await self._http(payload, headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(OPENAI_URL, json=payload, headers=headers)
        if response.status_code != 200:
            raise AssistantError(
                f"OpenAI ответил {response.status_code}: {response.text[:200]}"
            )
        return dict(response.json())

    async def run(
        self,
        scenario: Scenario,
        *,
        history: list[dict[str, str]] | None = None,
        **params: Any,
    ) -> Answer:
        tools = [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"] or "",
                "parameters": tool["input_schema"],
                # Схемы MCP не обязаны проходить строгий режим OpenAI.
                "strict": False,
            }
            for tool in await tool_definitions(scenario.tools)
        ]
        items: list[dict[str, Any]] = [
            *(history or []),
            {"role": "user", "content": scenario.instruction.format(**params)},
        ]
        usage = Usage(prices=OPENAI_PRICES)
        for round_number in range(1, MAX_ROUNDS + 1):
            payload: dict[str, Any] = {
                "model": self._model,
                "instructions": scenario.system,
                "input": list(items),
                # Рассуждение тратит тот же лимит вывода, что и ответ.
                "max_output_tokens": MAX_TOKENS * 2,
                "reasoning": {"effort": OPENAI_REASONING},
                "store": False,
                "include": ["reasoning.encrypted_content"],
            }
            if tools:
                payload["tools"] = tools
            body = await self._post(payload)
            usage.add_openai(dict(body.get("usage") or {}))
            output = list(body.get("output") or [])
            items.extend(output)
            calls = [item for item in output if item.get("type") == "function_call"]
            if not calls:
                text = "\n".join(
                    str(part.get("text", ""))
                    for item in output if item.get("type") == "message"
                    for part in item.get("content") or []
                    if part.get("type") == "output_text"
                ).strip()
                return Answer(text=text, usage=usage, rounds=round_number)
            results = await asyncio.gather(*(
                call_tool(str(call.get("name")), _arguments(call.get("arguments")))
                for call in calls
            ))
            items.extend(
                {"type": "function_call_output", "call_id": call.get("call_id"), "output": result}
                for call, result in zip(calls, results, strict=True)
            )
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


def _arguments(raw: Any) -> dict[str, Any]:
    """Аргументы вызова у OpenAI приходят строкой JSON."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def cost_words(value: float) -> str:
    """Стоимость в центах; ниже цента — сотые цента, иначе дешёвый ответ выглядит нулём.

    Раньше здесь было «<1¢», и Telegram в режиме HTML принимал это за тег и
    отказывался отправлять ответ целиком (25.09.2026).
    """
    cents = value * 100
    if cents < 1:
        return f"{cents:.2f}¢"
    return f"{cents:.1f}¢"


def scenario_payload(text: str, limit: int = 4000) -> str:
    """Данные шаблона для кнопки «Объяснить», обрезанные по здравому смыслу."""
    return text if len(text) <= limit else text[:limit] + "…"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
