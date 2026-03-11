"""Agent — the core agent loop. Heart of the system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cognit.llm.generate import step
from cognit.llm.message import ToolCall
from cognit.soul.compaction import compact_context
from cognit.soul.context import Context
from cognit.soul.toolset import Toolset


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
        on_tool_start: Callable[[ToolCall], None] | None = None,
        on_tool_end: Callable[[ToolCall, str], None] | None = None,
        approval_callback: Callable[[ToolCall], bool] | None = None,
    ) -> str:
        """Handle one user turn: loop until the LLM stops calling tools.

        Returns the final assistant text response.
        """
        self.context.add_user(user_input)

        steps = 0
        final_text = ""

        while True:
            # --- guard: max steps ---
            steps += 1
            if steps > self.config.max_steps:
                final_text = "[max steps reached, stopping]"
                break

            # --- compact if context is too long ---
            if self.context.needs_compaction():
                if on_text_delta:
                    on_text_delta("\n[compacting context...]\n")
                await compact_context(self.provider, self.context, self.system_prompt)

            # --- one step: LLM call + tool execution ---
            result = await step(
                provider=self.provider,
                system_prompt=self.system_prompt,
                tools=self.toolset.definitions(),
                history=self.context.messages,
                tool_executor=self.toolset,
                on_text_delta=on_text_delta,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )

            # Add assistant message to context
            self.context.add(result.message)

            if not result.has_tool_calls:
                # LLM didn't call any tools → it's done responding
                final_text = result.message.text
                break

            # --- tool approval check ---
            for tc in result.message.tool_calls:
                tool_def = self.toolset.get(tc.name)
                if tool_def and tool_def.requires_approval and approval_callback:
                    if not approval_callback(tc):
                        # User rejected → stop
                        self.context.add_tool_result(
                            tc.id, "Tool execution rejected by user.", is_error=True
                        )
                        return "[tool rejected by user]"

            # Add tool results to context
            for tr in result.tool_results:
                self.context.add(tr)

        return final_text
