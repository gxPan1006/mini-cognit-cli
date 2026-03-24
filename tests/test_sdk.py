"""Tests for the Cognit SDK — unit tests with a fake LLM provider."""

from __future__ import annotations

import asyncio
import json
import pytest

from cognit import CognitAgent, ChatResult, Toolset, Message, ToolCall, ToolResult, AgentConfig
from cognit.soul.agent import Agent
from cognit.soul.context import Context
from cognit.llm.generate import generate, step, GenerateResult, StepResult


# ---------------------------------------------------------------------------
# Fake provider: yields pre-scripted deltas so we never hit a real API
# ---------------------------------------------------------------------------

class FakeProvider:
    """A mock ChatProvider that returns scripted responses."""

    def __init__(self, responses: list[list[dict]]) -> None:
        """Each entry in `responses` is a list of deltas for one generate() call."""
        self._responses = list(responses)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def generate(self, system_prompt, tools, history):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        for delta in self._responses[idx]:
            yield delta


def text_response(text: str) -> list[dict]:
    """Helper: deltas for a plain text response."""
    return [{"type": "text_delta", "text": text}, {"type": "done"}]


def tool_call_response(tool_id: str, name: str, arguments: dict) -> list[dict]:
    """Helper: deltas for a single tool call."""
    return [
        {"type": "tool_call_delta", "index": 0, "id": tool_id, "name": name, "arguments": ""},
        {"type": "tool_call_delta", "index": 0, "id": "", "name": "", "arguments": json.dumps(arguments)},
        {"type": "done"},
    ]


# ---------------------------------------------------------------------------
# Message model tests
# ---------------------------------------------------------------------------

class TestMessage:
    def test_user_message(self):
        msg = Message.user("hello")
        assert msg.role == "user"
        assert msg.text == "hello"

    def test_assistant_message(self):
        msg = Message.assistant_text("world")
        assert msg.role == "assistant"
        assert msg.text == "world"

    def test_tool_result_message(self):
        msg = Message.tool_result("call_1", "output", is_error=False)
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"
        assert msg.text == "output"

    def test_to_openai_user(self):
        d = Message.user("hi").to_openai()
        assert d == {"role": "user", "content": "hi"}

    def test_to_openai_tool_result(self):
        d = Message.tool_result("c1", "result").to_openai()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "c1"
        assert d["content"] == "result"

    def test_tool_call_safe_arguments_valid(self):
        tc = ToolCall(id="1", name="test", arguments='{"a": 1}')
        assert tc.safe_arguments == '{"a": 1}'

    def test_tool_call_safe_arguments_malformed(self):
        tc = ToolCall(id="1", name="test", arguments='{"a": 1}{"b": 2}')
        parsed = json.loads(tc.safe_arguments)
        assert parsed == {"a": 1}

    def test_tool_call_safe_arguments_garbage(self):
        tc = ToolCall(id="1", name="test", arguments="not json at all")
        assert tc.safe_arguments == "{}"


# ---------------------------------------------------------------------------
# Context tests
# ---------------------------------------------------------------------------

class TestContext:
    def test_add_and_clear(self):
        ctx = Context()
        ctx.add_user("hello")
        assert len(ctx.messages) == 1
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_estimated_tokens(self):
        ctx = Context()
        ctx.add_user("a" * 400)  # ~100 tokens
        assert ctx.estimated_tokens() == 100

    def test_needs_compaction(self):
        ctx = Context(max_tokens=100)
        ctx.add_user("a" * 400)  # 100 tokens, limit is 100, reserve is 30000
        # With reserve=30000 and max=100, threshold is negative → always needs compaction
        assert ctx.needs_compaction()

    def test_replace_with_summary(self):
        ctx = Context()
        for i in range(10):
            ctx.add_user(f"msg {i}")
        ctx.replace_with_summary("summary of old messages", keep_last_n=3)
        assert len(ctx.messages) == 4  # 1 summary + 3 kept
        assert ctx.messages[0].text == "[Conversation summary]\nsummary of old messages"

    def test_repair_orphaned_tool_calls(self):
        ctx = Context()
        ctx.add_user("do something")
        # Add assistant with tool call but no matching tool result
        assistant_msg = Message(
            role="assistant",
            tool_calls=[ToolCall(id="tc_1", name="shell", arguments='{"cmd": "ls"}')],
        )
        ctx.add(assistant_msg)
        # No tool result added — simulate an interruption
        ctx.repair()
        # The orphaned assistant message should be removed
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"

    def test_repair_noop_when_valid(self):
        ctx = Context()
        ctx.add_user("do something")
        assistant_msg = Message(
            role="assistant",
            tool_calls=[ToolCall(id="tc_1", name="shell", arguments='{}')],
        )
        ctx.add(assistant_msg)
        ctx.add(Message.tool_result("tc_1", "output"))
        ctx.repair()
        assert len(ctx.messages) == 3  # user + assistant + tool_result


# ---------------------------------------------------------------------------
# Toolset tests
# ---------------------------------------------------------------------------

class TestToolset:
    def test_register_and_get(self):
        ts = Toolset()

        async def handler(x: str) -> str:
            return x

        ts.register("echo", "echoes input", {"type": "object", "properties": {"x": {"type": "string"}}}, handler)
        assert ts.get("echo") is not None
        assert ts.get("nonexistent") is None

    def test_decorator_registration(self):
        ts = Toolset()

        @ts.tool(name="greet", description="Says hello")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert ts.get("greet") is not None
        defs = ts.definitions()
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "greet"

    def test_definitions_openai_format(self):
        ts = Toolset()

        @ts.tool(name="test_fn", description="A test function")
        async def test_fn(query: str, count: int = 5) -> str:
            return "ok"

        defs = ts.definitions()
        assert len(defs) == 1
        fn = defs[0]
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "test_fn"
        assert "query" in fn["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        ts = Toolset()

        @ts.tool(name="add", description="adds numbers")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        tc = ToolCall(id="c1", name="add", arguments='{"a": 3, "b": 4}')
        result = await ts.execute(tc)
        assert result.content == "7"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        ts = Toolset()
        tc = ToolCall(id="c1", name="nonexistent", arguments="{}")
        result = await ts.execute(tc)
        assert result.is_error
        assert "unknown tool" in result.content.lower()

    @pytest.mark.asyncio
    async def test_execute_bad_json(self):
        ts = Toolset()

        @ts.tool(name="echo")
        async def echo(text: str) -> str:
            return text

        tc = ToolCall(id="c1", name="echo", arguments="not json")
        result = await ts.execute(tc)
        assert result.is_error


# ---------------------------------------------------------------------------
# generate() / step() tests
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.mark.asyncio
    async def test_text_response(self):
        provider = FakeProvider([text_response("Hello world")])
        result = await generate(provider, "system", [], [])
        assert result.message.text == "Hello world"
        assert result.message.role == "assistant"
        assert len(result.message.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_streaming_callback(self):
        provider = FakeProvider([text_response("Hi")])
        chunks = []
        result = await generate(provider, "system", [], [], on_text_delta=chunks.append)
        assert chunks == ["Hi"]

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        provider = FakeProvider([tool_call_response("tc1", "shell", {"cmd": "ls"})])
        result = await generate(provider, "system", [], [])
        assert len(result.message.tool_calls) == 1
        tc = result.message.tool_calls[0]
        assert tc.id == "tc1"
        assert tc.name == "shell"
        assert json.loads(tc.arguments) == {"cmd": "ls"}


class TestStep:
    @pytest.mark.asyncio
    async def test_step_no_tools(self):
        provider = FakeProvider([text_response("Done")])
        ts = Toolset()
        result = await step(provider, "system", [], [], ts)
        assert result.message.text == "Done"
        assert not result.has_tool_calls
        assert len(result.tool_results) == 0

    @pytest.mark.asyncio
    async def test_step_with_tool_execution(self):
        provider = FakeProvider([tool_call_response("tc1", "greet", {"name": "Alice"})])
        ts = Toolset()

        @ts.tool(name="greet")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        started, ended = [], []
        result = await step(
            provider, "system", ts.definitions(), [], ts,
            on_tool_start=lambda tc: started.append(tc.name),
            on_tool_end=lambda tc, r: ended.append(r),
        )
        assert result.has_tool_calls
        assert len(result.tool_results) == 1
        assert result.tool_results[0].text == "Hello, Alice!"
        assert started == ["greet"]
        assert ended == ["Hello, Alice!"]


# ---------------------------------------------------------------------------
# Agent loop tests
# ---------------------------------------------------------------------------

class TestAgent:
    @pytest.mark.asyncio
    async def test_single_text_turn(self):
        provider = FakeProvider([text_response("Hi there!")])
        ts = Toolset()
        agent = Agent(provider, ts, "You are helpful.", AgentConfig(max_steps=10))

        reply = await agent.run("Hello")
        assert reply == "Hi there!"
        assert len(agent.context.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_tool_then_text(self):
        """Agent calls a tool, gets result, then replies with text."""
        provider = FakeProvider([
            tool_call_response("tc1", "echo", {"text": "world"}),
            text_response("I echoed: world"),
        ])
        ts = Toolset()

        @ts.tool(name="echo")
        async def echo(text: str) -> str:
            return text

        agent = Agent(provider, ts, "system", AgentConfig(max_steps=10))
        reply = await agent.run("Echo world")
        assert reply == "I echoed: world"
        # user + assistant(tool_call) + tool_result + assistant(text)
        assert len(agent.context.messages) == 4

    @pytest.mark.asyncio
    async def test_max_steps_guard(self):
        """Agent stops after max_steps even if LLM keeps calling tools."""
        provider = FakeProvider([
            tool_call_response("tc1", "echo", {"text": "loop"}),
        ])
        ts = Toolset()

        @ts.tool(name="echo")
        async def echo(text: str) -> str:
            return text

        agent = Agent(provider, ts, "system", AgentConfig(max_steps=2))
        reply = await agent.run("Loop forever")
        assert "max steps" in reply.lower()

    @pytest.mark.asyncio
    async def test_multi_turn_context(self):
        """Context accumulates across multiple run() calls."""
        provider = FakeProvider([
            text_response("First reply"),
            text_response("Second reply"),
        ])
        ts = Toolset()
        agent = Agent(provider, ts, "system", AgentConfig(max_steps=10))

        await agent.run("Turn 1")
        await agent.run("Turn 2")
        # turn1: user + assistant, turn2: user + assistant
        assert len(agent.context.messages) == 4


# ---------------------------------------------------------------------------
# CognitAgent (SDK wrapper) tests
# ---------------------------------------------------------------------------

class TestCognitAgent:
    def test_init_with_builtin_tools(self):
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake", auto_config=False
        )
        tool_names = [t.name for t in agent.toolset._tools.values()]
        assert "read_file" in tool_names
        assert "shell" in tool_names

    def test_init_without_builtin_tools(self):
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake",
            auto_config=False, builtin_tools=False,
        )
        assert len(agent.toolset._tools) == 0

    def test_custom_tool_decorator(self):
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake",
            auto_config=False, builtin_tools=False,
        )

        @agent.tool(name="ping", description="pong")
        async def ping() -> str:
            return "pong"

        assert agent.toolset.get("ping") is not None

    def test_custom_system_prompt(self):
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake",
            auto_config=False, system_prompt="You are a cat.",
        )
        assert agent.system_prompt.startswith("You are a cat.")

    def test_clear(self):
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake", auto_config=False,
        )
        agent.agent.context.add_user("hello")
        assert len(agent.context.messages) == 1
        agent.clear()
        assert len(agent.context.messages) == 0

    @pytest.mark.asyncio
    async def test_chat_returns_chat_result(self):
        """Patch the inner agent to use a fake provider."""
        agent = CognitAgent(
            api_key="sk-test", base_url="http://fake",
            auto_config=False, builtin_tools=False,
        )
        # Replace provider with fake
        agent.agent.provider = FakeProvider([text_response("SDK works!")])

        result = await agent.chat("test")
        assert isinstance(result, ChatResult)
        assert result.text == "SDK works!"
        assert result.steps == 1
