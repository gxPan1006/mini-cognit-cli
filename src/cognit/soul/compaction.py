"""Compaction — summarize old context to stay within token limits."""

from __future__ import annotations

from typing import Any

from cognit.llm.message import Message
from cognit.soul.context import Context


COMPACTION_PROMPT = """\
Summarize the conversation so far in a concise way. Preserve:
- Key decisions and conclusions
- Important file paths, function names, and code snippets
- Any pending tasks or open questions
Keep the summary under 2000 tokens. Be factual and direct."""


async def compact_context(
    provider: Any,  # ChatProvider
    context: Context,
    system_prompt: str,
) -> None:
    """Summarize early messages and replace them with a summary."""
    if not context.needs_compaction():
        return

    # Build a summarization request from all messages
    history_text = []
    for m in context.messages:
        prefix = m.role.upper()
        history_text.append(f"[{prefix}]: {m.text}")
        for tc in m.tool_calls:
            history_text.append(f"  -> tool_call: {tc.name}({tc.arguments[:200]})")

    summary_request = COMPACTION_PROMPT + "\n\n" + "\n".join(history_text)

    # Use the LLM to generate a summary (non-streaming, simple call)
    from cognit.llm.generate import generate

    result = await generate(
        provider=provider,
        system_prompt="You are a helpful summarizer.",
        tools=[],
        history=[Message.user(summary_request)],
    )

    summary = result.message.text
    context.replace_with_summary(summary)
