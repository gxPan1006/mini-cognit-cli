"""Context — manages the conversation message history."""

from __future__ import annotations

from cognit.llm.message import Message


class Context:
    """Holds the conversation history for one session."""

    def __init__(self, max_tokens: int = 128_000) -> None:
        self._messages: list[Message] = []
        self._max_tokens = max_tokens

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_user(self, text: str) -> None:
        self._messages.append(Message.user(text))

    def add_tool_result(self, tool_call_id: str, content: str, is_error: bool = False) -> None:
        self._messages.append(Message.tool_result(tool_call_id, content, is_error))

    def estimated_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(
            len(m.text) + sum(len(tc.arguments) for tc in m.tool_calls)
            for m in self._messages
        )
        return total_chars // 4

    def needs_compaction(self, reserve: int = 30_000) -> bool:
        """True if the context is getting close to the limit."""
        return self.estimated_tokens() > (self._max_tokens - reserve)

    def replace_with_summary(self, summary: str, keep_last_n: int = 4) -> None:
        """Replace old messages with a summary, keeping the last N messages."""
        if len(self._messages) <= keep_last_n:
            return
        kept = self._messages[-keep_last_n:]
        self._messages = [Message.system(f"[Conversation summary]\n{summary}")] + kept

    def clear(self) -> None:
        self._messages.clear()
