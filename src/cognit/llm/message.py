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

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TextPart:
        return TextPart(text=d["text"])


@dataclass
class ThinkPart:
    text: str
    type: Literal["think"] = "think"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "think", "text": self.text}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ThinkPart:
        return ThinkPart(text=d["text"])


@dataclass
class ImagePart:
    """An image content part, stored as a base64 data URI or a URL."""
    url: str  # base64 data URI (data:image/png;base64,...) or https:// URL
    detail: str = "auto"  # "auto", "low", "high"
    type: Literal["image"] = "image"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "image", "url": self.url, "detail": self.detail}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ImagePart:
        return ImagePart(url=d["url"], detail=d.get("detail", "auto"))


ContentPart = TextPart | ThinkPart | ImagePart

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

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ToolCall:
        return ToolCall(id=d["id"], name=d["name"], arguments=d["arguments"])


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
    def user_with_images(text: str, image_urls: list[str]) -> Message:
        """Create a user message with text and image(s)."""
        parts: list[ContentPart] = [TextPart(text)]
        for url in image_urls:
            parts.append(ImagePart(url=url))
        return Message(role="user", content=parts)

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

    @property
    def has_images(self) -> bool:
        return any(isinstance(p, ImagePart) for p in self.content)

    # -- serialization to OpenAI format -------------------------------------

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions message format."""
        msg: dict[str, Any] = {"role": self.role}

        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id
            msg["content"] = self.text
            return msg

        if self.content:
            # If message has images, use the multimodal content array format
            if self.has_images:
                content_parts: list[dict[str, Any]] = []
                for p in self.content:
                    if isinstance(p, TextPart):
                        content_parts.append({"type": "text", "text": p.text})
                    elif isinstance(p, ImagePart):
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": p.url, "detail": p.detail},
                        })
                    # ThinkParts are not sent to OpenAI in content
                msg["content"] = content_parts
            else:
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

    # -- dict serialization (for session persistence) -----------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d: dict[str, Any] = {"role": self.role}
        if self.content:
            d["content"] = [p.to_dict() for p in self.content]
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Message:
        """Deserialize from a plain dict."""
        content: list[ContentPart] = []
        for p in d.get("content", []):
            ptype = p.get("type", "text")
            if ptype == "text":
                content.append(TextPart.from_dict(p))
            elif ptype == "think":
                content.append(ThinkPart.from_dict(p))
            elif ptype == "image":
                content.append(ImagePart.from_dict(p))

        tool_calls = [ToolCall.from_dict(tc) for tc in d.get("tool_calls", [])]

        return Message(
            role=d["role"],
            content=content,
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
        )
