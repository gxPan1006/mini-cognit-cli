"""Agent — the core agent loop. Heart of the system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from cognit.llm.generate import step
from cognit.llm.message import ToolCall
from cognit.soul.compaction import compact_context
from cognit.soul.context import Context
from cognit.soul.toolset import Toolset

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_steps: int = 50
    max_tokens: int = 128_000


class Agent:
    """The agent brain: receives user input, loops LLM + tools until done."""

    def __init__(
        self,
        provider: Any,  # ChatProvider
        toolset: Toolset,
        system_prompt: str,
        config: AgentConfig | None = None,
    ) -> None:
        self.provider = provider
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.context = Context(max_tokens=(config or AgentConfig()).max_tokens)
        self.config = config or AgentConfig()

    async def run(
        self,
        user_input: str,
        on_text_delta: Callable[[str], None] | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
        on_tool_start: Callable[[ToolCall], None] | None = None,
        on_tool_end: Callable[[ToolCall, str], None] | None = None,
        approval_callback: Callable[[ToolCall], bool] | None = None,
    ) -> str:
        """Handle one user turn: loop until the LLM stops calling tools.

        Returns the final assistant text response.
        """
        logger.info("Agent.run() called — user_input=%r (len=%d)", user_input[:200], len(user_input))
        self.context.add_user(user_input)

        steps = 0
        final_text = ""

        while True:
            # --- guard: max steps ---
            steps += 1
            if steps > self.config.max_steps:
                logger.warning("Max steps (%d) reached, stopping agent loop", self.config.max_steps)
                final_text = "[max steps reached, stopping]"
                break

            logger.info("Agent step %d/%d — context messages=%d, est_tokens=%d",
                        steps, self.config.max_steps, len(self.context.messages), self.context.estimated_tokens())

            # --- compact if context is too long ---
            if self.context.needs_compaction():
                logger.info("Context needs compaction (est_tokens=%d, max=%d)",
                            self.context.estimated_tokens(), self.context._max_tokens)
                if on_text_delta:
                    on_text_delta("\n[compacting context...]\n")
                await compact_context(self.provider, self.context, self.system_prompt)
                logger.info("Compaction done — est_tokens now=%d", self.context.estimated_tokens())

            # --- one step: LLM call + tool execution ---
            try:
                result = await step(
                    provider=self.provider,
                    system_prompt=self.system_prompt,
                    tools=self.toolset.definitions(),
                    history=self.context.messages,
                    tool_executor=self.toolset,
                    on_text_delta=on_text_delta,
                    on_thinking_delta=on_thinking_delta,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                )
            except Exception:
                logger.exception("Exception during step %d", steps)
                raise

            # Add assistant message + tool results to context atomically
            # Tool results MUST be added right after assistant message to avoid
            # corrupted context (API requires every tool_use has a tool_result)
            self.context.add(result.message)
            for tr in result.tool_results:
                self.context.add(tr)

            logger.info("Step %d result — text_len=%d, tool_calls=%d",
                        steps, len(result.message.text), len(result.message.tool_calls))
            for tc in result.message.tool_calls:
                logger.info("  tool_call: %s(id=%s, args_len=%d)", tc.name, tc.id, len(tc.arguments))
            for tr in result.tool_results:
                logger.info("  tool_result: call_id=%s, content_len=%d",
                            tr.tool_call_id, len(tr.text))

            if not result.has_tool_calls:
                # LLM didn't call any tools → it's done responding
                final_text = result.message.text
                logger.info("Agent loop finished — no tool calls, final_text_len=%d", len(final_text))
                break

            # --- tool approval check ---
            for tc in result.message.tool_calls:
                tool_def = self.toolset.get(tc.name)
                if tool_def and tool_def.requires_approval and approval_callback:
                    logger.info("Requesting approval for tool %s (id=%s)", tc.name, tc.id)
                    if not approval_callback(tc):
                        logger.info("User rejected tool %s (id=%s)", tc.name, tc.id)
                        return "[tool rejected by user]"

        logger.info("Agent.run() returning — total steps=%d, final_text_len=%d", steps, len(final_text))
        return final_text
