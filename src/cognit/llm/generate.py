"""generate() and step() — the two core LLM call primitives.

generate() = pure LLM call + streaming assembly
step()     = generate() + tool dispatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from cognit.llm.message import Message, TextPart, ThinkPart, ToolCall


@dataclass
class GenerateResult:
    """The assembled result of one LLM generate() call."""
    message: Message
    # Could add usage/token stats here later


@dataclass
class StepResult:
    """Result of one step: LLM message + tool results (if any)."""
    message: Message
    tool_results: list[Message] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.message.tool_calls)


# ---------------------------------------------------------------------------
# generate() — call LLM, stream text, assemble message
# ---------------------------------------------------------------------------

async def generate(
    provider: Any,  # ChatProvider
    system_prompt: str,
    tools: Sequence[dict[str, Any]],
    history: Sequence[Message],
    on_text_delta: Callable[[str], None] | None = None,
    on_thinking_delta: Callable[[str], None] | None = None,
) -> GenerateResult:
    """Call the LLM, stream deltas, return assembled Message."""

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    # tool_calls accumulator: index -> {id, name, arguments}
    tc_accum: dict[int, dict[str, str]] = {}

    stream = provider.generate(system_prompt, tools, history)
    async for delta in stream:
        dtype = delta["type"]

        if dtype == "thinking_delta":
            text = delta["text"]
            thinking_parts.append(text)
            if on_thinking_delta:
                on_thinking_delta(text)

        elif dtype == "text_delta":
            text = delta["text"]
            text_parts.append(text)
            if on_text_delta:
                on_text_delta(text)

        elif dtype == "tool_call_delta":
            idx = delta["index"]
            if idx not in tc_accum:
                tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
            acc = tc_accum[idx]
            if delta["id"]:
                acc["id"] = delta["id"]
            if delta["name"]:
                acc["name"] = delta["name"]
            acc["arguments"] += delta["arguments"]

    # Assemble message
    content = []
    full_thinking = "".join(thinking_parts)
    if full_thinking:
        content.append(ThinkPart(full_thinking))

    full_text = "".join(text_parts)
    if full_text:
        content.append(TextPart(full_text))

    tool_calls = []
    for idx in sorted(tc_accum):
        acc = tc_accum[idx]
        tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=acc["arguments"]))

    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    return GenerateResult(message=message)


# ---------------------------------------------------------------------------
# step() — generate + tool dispatch
# ---------------------------------------------------------------------------

async def step(
    provider: Any,  # ChatProvider
    system_prompt: str,
    tools: Sequence[dict[str, Any]],
    history: Sequence[Message],
    tool_executor: Any,  # Toolset
    on_text_delta: Callable[[str], None] | None = None,
    on_thinking_delta: Callable[[str], None] | None = None,
    on_tool_start: Callable[[ToolCall], None] | None = None,
    on_tool_end: Callable[[ToolCall, str], None] | None = None,
) -> StepResult:
    """One step = generate() + execute all tool calls."""

    result = await generate(provider, system_prompt, tools, history, on_text_delta, on_thinking_delta)
    msg = result.message

    tool_results: list[Message] = []

    for tc in msg.tool_calls:
        if on_tool_start:
            on_tool_start(tc)

        tr = await tool_executor.execute(tc)
        tool_results.append(Message.tool_result(tc.id, tr.content, tr.is_error))

        if on_tool_end:
            on_tool_end(tc, tr.content)

    return StepResult(message=msg, tool_results=tool_results)
