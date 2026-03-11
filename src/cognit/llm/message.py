"""Unified message model — the lingua franca between LLM providers and the agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Content parts (polymorphic by `type`)
# ---------------------------------------------------------------------------

@dataclass
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ThinkPart:
    text: str
    type: Literal["think"] = "think"


ContentPart = TextPart | ThinkPart

# ---------------------------------------------------------------------------
# Tool call & result
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, as returned by the LLM

    @property
    def safe_arguments(self) -> str:
        """Return arguments guaranteed to be valid JSON (for API serialization).

        If the LLM produced malformed arguments (e.g. two JSON objects
        concatenated), extract the first valid JSON object.
        """
        try:
            json.loads(self.arguments)
            return self.arguments
        except (json.JSONDecodeError, ValueError):
            # Try to extract the first JSON object
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(self.arguments)
                return json.dumps(obj)
            except (json.JSONDecodeError, ValueError):
                return "{}"


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # only for role="tool"

    # -- helpers -------------------------------------------------------------

    @property
    def text(self) -> str:
        """Concatenated text of all TextParts."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @staticmethod
    def system(text: str) -> Message:
        return Message(role="system", content=[TextPart(text)])

    @staticmethod
    def user(text: str) -> Message:
        return Message(role="user", content=[TextPart(text)])

    @staticmethod
    def assistant_text(text: str) -> Message:
        return Message(role="assistant", content=[TextPart(text)])

    @staticmethod
    def tool_result(tool_call_id: str, content: str, is_error: bool = False) -> Message:
        return Message(
            role="tool",
            content=[TextPart(content)],
            tool_call_id=tool_call_id,
        )

    # -- serialization to OpenAI format -------------------------------------

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions message format."""
        msg: dict[str, Any] = {"role": self.role}

        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id
            msg["content"] = self.text
            return msg

        if self.content:
            msg["content"] = self.text

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.safe_arguments},
                }
                for tc in self.tool_calls
            ]

        return msg
