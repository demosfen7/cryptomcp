"""Сценарии через Claude: деньги, лимит, инструменты и пределы (PLAN §4.44).

Сеть не используется: вместо Messages API подставляется поддельный
обработчик, который возвращает заранее собранные ответы вместе с `usage`.
"""

from __future__ import annotations

import pytest

from cryptomcp import assistant as claude
from cryptomcp.assistant import Answer, Assistant, Usage, cost_words


class FakeAPI:
    """Ответы Claude по сценарию теста; считает запросы и хранит payload."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.payloads: list[dict] = []

    async def __call__(self, payload, headers):
        self.payloads.append(payload)
        key = headers.get("x-api-key") or headers.get("authorization")
        assert key, "ключ обязан уходить заголовком, не в теле"
        return self._replies.pop(0) if self._replies else self._replies_exhausted()

    @staticmethod
    def _replies_exhausted():
        raise AssertionError("ответов больше нет, а запрос пришёл")


def text_reply(text, **usage):
    payload = {"input_tokens": 100, "output_tokens": 200}
    payload.update(usage)
    return {"content": [{"type": "text", "text": text}], "usage": payload}


def tool_reply(name, arguments=None, **usage):
    payload = {"input_tokens": 100, "output_tokens": 50}
    payload.update(usage)
    return {
        "content": [
            {"type": "tool_use", "id": "call_1", "name": name, "input": arguments or {}}
        ],
        "usage": payload,
    }


def test_cost_is_counted_from_usage_with_cache_prices():
    """Цена считается по фактическому usage, а не по оценке сценария."""
    usage = Usage(
        input_tokens=1_000, cache_creation_tokens=2_000,
        cache_read_tokens=10_000, output_tokens=1_500,
    )

    expected = (1_000 * 1.0 + 2_000 * 1.25 + 10_000 * 0.1 + 1_500 * 5.0) / 1_000_000

    assert usage.cost_usd == pytest.approx(expected)
    assert cost_words(usage.cost_usd) == "1.2¢"
    # Без «<»: в HTML Telegram это читалось как тег, и ответ не уходил.
    assert cost_words(0.0001) == "0.01¢"


def test_usage_sums_over_all_rounds():
    usage = Usage()
    usage.add({"input_tokens": 10, "output_tokens": 5})
    usage.add({"input_tokens": 7, "cache_read_input_tokens": 100, "output_tokens": 3})

    assert (usage.input_tokens, usage.output_tokens) == (17, 8)
    assert usage.cache_read_tokens == 100


@pytest.mark.asyncio
async def test_each_button_gets_only_its_own_tools():
    """Описания всех восемнадцати инструментов оплачивались бы каждым запросом."""
    analyse = await claude.tool_definitions(claude.SCENARIOS["analyse"].tools)
    weekly = await claude.tool_definitions(claude.SCENARIOS["weekly"].tools)

    assert {tool["name"] for tool in analyse} == set(claude.SCENARIOS["analyse"].tools)
    assert {tool["name"] for tool in weekly} == {"get_watchlist", "get_accumulation"}
    assert await claude.tool_definitions(()) == []


@pytest.mark.asyncio
async def test_system_prompt_and_tools_are_cached():
    """Внутри сценария 2–3 запроса: повтор префикса должен стоить десятую часть."""
    api = FakeAPI([text_reply("готово")])
    assistant = Assistant("key", http=api)

    await assistant.run(claude.SCENARIOS["weekly"], stamp="17.09, 12:00 UTC")

    payload = api.payloads[0]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["model"] == claude.MODEL
    assert payload["max_tokens"] == claude.MAX_TOKENS


@pytest.mark.asyncio
async def test_tool_results_go_back_to_the_model(monkeypatch):
    calls = []

    async def fake_tool(name, arguments):
        calls.append((name, arguments))
        return "список наблюдения: пусто"

    monkeypatch.setattr(claude, "call_tool", fake_tool)
    api = FakeAPI([
        tool_reply("get_watchlist", {"status": "all"}),
        text_reply("📈 Итоги недели"),
    ])

    answer = await Assistant("key", http=api).run(
        claude.SCENARIOS["weekly"], stamp="17.09, 12:00 UTC"
    )

    assert calls == [("get_watchlist", {"status": "all"})]
    assert answer.text == "📈 Итоги недели"
    assert answer.rounds == 2
    assert answer.usage.output_tokens == 250
    result_message = api.payloads[1]["messages"][-1]
    assert result_message["content"][0]["type"] == "tool_result"


@pytest.mark.asyncio
async def test_loop_stops_at_the_round_limit(monkeypatch):
    """Зацикливание на инструменте перестаёт стоить денег после шести кругов."""
    async def fake_tool(name, arguments):
        return "данные"

    monkeypatch.setattr(claude, "call_tool", fake_tool)
    api = FakeAPI([tool_reply("get_watchlist") for _ in range(claude.MAX_ROUNDS)])

    answer = await Assistant("key", http=api).run(
        claude.SCENARIOS["weekly"], stamp="17.09, 12:00 UTC"
    )

    assert answer.truncated is True
    assert answer.rounds == claude.MAX_ROUNDS
    assert len(api.payloads) == claude.MAX_ROUNDS
    assert "Не успел собрать ответ" in answer.text


@pytest.mark.asyncio
async def test_tool_failure_is_told_to_the_model_not_raised(monkeypatch):
    """Инструмент проекта и так отвечает отказом вместо трейсбека (§6.5)."""
    async def broken(**kwargs):
        raise RuntimeError("архив недоступен")

    from cryptomcp import server

    monkeypatch.setattr(server, "get_watchlist", broken, raising=False)

    result = await claude.call_tool("get_watchlist", {})

    assert "не отработал" in result and "архив недоступен" in result


@pytest.mark.asyncio
async def test_unknown_tool_does_not_crash_the_scenario():
    assert "недоступен" in await claude.call_tool("нет_такого", {})


def test_without_a_key_there_is_no_assistant(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Assistant.from_env() is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert isinstance(Assistant.from_env(), Assistant)


def test_answer_reports_zero_cost_without_usage():
    assert Answer(text="пусто").cost_usd == 0.0


def test_openai_usage_splits_cached_and_written_input():
    """input_tokens у OpenAI уже включает кеш — иначе он оплачивался бы дважды."""
    usage = Usage(prices=claude.OPENAI_PRICES)
    usage.add_openai({
        "input_tokens": 1_500,
        "input_tokens_details": {"cached_tokens": 1_200, "cache_write_tokens": 250},
        "output_tokens": 100,
    })

    assert (usage.input_tokens, usage.cache_read_tokens) == (50, 1_200)
    assert usage.cache_creation_tokens == 250
    expected = (50 * 0.10 + 250 * 0.125 + 1_200 * 0.01 + 100 * 0.50) / 1_000_000
    assert usage.cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_openai_tool_loop_echoes_output_and_returns_results(monkeypatch):
    calls = []

    async def fake_tool(name, arguments):
        calls.append((name, arguments))
        return "пусто"

    monkeypatch.setattr(claude, "call_tool", fake_tool)
    usage = {"input_tokens": 100, "output_tokens": 10}
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}
    call = {
        "type": "function_call", "call_id": "call_1",
        "name": "get_watchlist", "arguments": '{"status": "all"}',
    }
    message = {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "Итоги"}],
    }
    api = FakeAPI([
        {"output": [reasoning, call], "usage": usage},
        {"output": [message], "usage": usage},
    ])
    history = [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "да"}]

    answer = await claude.OpenAIAssistant("key", http=api).run(
        claude.SCENARIOS["chat"], history=history, payload="что в списке?",
    )

    assert answer.text == "Итоги"
    assert calls == [("get_watchlist", {"status": "all"})]
    first, second = api.payloads
    assert first["model"] == "gpt-6-luna"
    assert first["input"][:2] == history
    assert first["tools"][0]["type"] == "function"
    assert first["instructions"] == claude.CHAT_SYSTEM_PROMPT
    assert second["input"][-3:] == [
        reasoning, call,
        {"type": "function_call_output", "call_id": "call_1", "output": "пусто"},
    ]
    assert answer.usage.input_tokens == 200


def test_openai_key_wins_over_claude(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test")

    assert isinstance(Assistant.from_env(), claude.OpenAIAssistant)


def test_model_html_is_escaped_except_bold():
    from cryptomcp.bot import telegram_html

    assert telegram_html("<b>ID</b> <5% · 0.01¢ & <i>") == (
        "<b>ID</b> &lt;5% · 0.01¢ &amp; &lt;i&gt;"
    )
