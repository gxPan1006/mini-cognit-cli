"""ChatProvider protocol — the contract every LLM provider must satisfy."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, Sequence

from cognit.llm.message import Message


class ChatProvider(Protocol):
    """Abstract LLM provider interface (duck-typed via Protocol)."""

    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[dict[str, Any]],
        history: Sequence[Message],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream chat completion chunks.

        Each yielded dict follows the OpenAI delta format:
          {"type": "text_delta", "text": "..."} or
          {"type": "tool_call_delta", "index": 0, "id": "...", "name": "...", "arguments": "..."}
          {"type": "done"}
        """
        ...
