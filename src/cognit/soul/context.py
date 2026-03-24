"""Context — manages the conversation message history."""

from __future__ import annotations

import logging

from cognit.llm.message import Message

logger = logging.getLogger(__name__)


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
        """Rough token estimate: ~4 chars per token, images ~1000 tokens each."""
        from cognit.llm.message import ImagePart

        total_chars = 0
        image_count = 0
        for m in self._messages:
            total_chars += len(m.text) + sum(len(tc.arguments) for tc in m.tool_calls)
            image_count += sum(1 for p in m.content if isinstance(p, ImagePart))
        return total_chars // 4 + image_count * 1000

    def needs_compaction(self, reserve: int = 30_000) -> bool:
        """True if the context is getting close to the limit."""
        return self.estimated_tokens() > (self._max_tokens - reserve)

    def replace_with_summary(self, summary: str, keep_last_n: int = 4) -> None:
        """Replace old messages with a summary, keeping the last N messages."""
        if len(self._messages) <= keep_last_n:
            return
        kept = self._messages[-keep_last_n:]
        self._messages = [Message.system(f"[Conversation summary]\n{summary}")] + kept

    def repair(self) -> None:
        """Fix corrupted context: ensure every assistant tool_call has a matching tool_result.

        If the last assistant message has tool_calls without corresponding tool_results,
        remove the orphaned assistant message to prevent API 400 errors.
        """
        if not self._messages:
            return

        # Find the last assistant message with tool_calls
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                # Check that each tool_call has a tool_result after it
                expected_ids = {tc.id for tc in msg.tool_calls}
                found_ids = set()
                for j in range(i + 1, len(self._messages)):
                    if self._messages[j].role == "tool" and self._messages[j].tool_call_id:
                        found_ids.add(self._messages[j].tool_call_id)
                missing = expected_ids - found_ids
                if missing:
                    logger.warning("Repairing context: removing orphaned assistant message "
                                   "at index %d with %d missing tool_results (ids: %s)",
                                   i, len(missing), missing)
                    # Remove the assistant message and any partial tool_results
                    self._messages = self._messages[:i]
                    return
                break  # found a valid assistant+tool_results pair, stop

    def clear(self) -> None:
        self._messages.clear()
