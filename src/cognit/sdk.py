"""SDK — programmatic interface for using Cognit as a library.

Usage:

    from cognit import CognitAgent

    agent = CognitAgent(model="gpt-4o", api_key="sk-...")

    # Register custom tools
    @agent.tool(name="my_tool", description="Does something useful")
    async def my_tool(query: str) -> str:
        return f"result for {query}"

    # Single turn
    reply = await agent.chat("Hello!")

    # With streaming callbacks
    reply = await agent.chat(
        "Read main.py",
        on_text_delta=lambda t: print(t, end=""),
    )

    # Multi-turn (context is preserved between calls)
    await agent.chat("What files are in this directory?")
    await agent.chat("Now read the first one.")

    # Reset conversation
    agent.clear()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Sequence

from cognit.llm.message import Message, ImagePart
from cognit.llm.openai_provider import OpenAIProvider
from cognit.soul.agent import Agent, AgentConfig
from cognit.soul.toolset import Toolset

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are Cognit, a powerful AI coding assistant.
You help users with software engineering tasks: writing code, debugging, exploring codebases, and more.

## Working principles
- Read files before modifying them.
- Prefer editing existing files over creating new ones.
- Use the shell tool for system commands (git, build tools, etc.).
- Use grep to search codebases efficiently.
- Use glob to discover files by name pattern.
- Use fetch_url to read web pages or API endpoints — do NOT use shell/curl to scrape websites.
- Use web_search to look up information online.
- Use read_media to view and analyze image files.
- Be concise and direct. Lead with the answer, not the reasoning.
- When you complete a task, briefly confirm what you did.

## Current directory
{cwd}

## Current date
{date}
"""


def _register_builtin_tools(toolset: Toolset) -> None:
    """Register the default built-in tools."""
    from cognit.tools import file_read, file_write, shell, grep, web_search
    from cognit.tools import glob_tool, fetch_url, media_read

    file_read.register(toolset)
    file_write.register(toolset)
    shell.register(toolset)
    grep.register(toolset)
    web_search.register(toolset)
    glob_tool.register(toolset)
    fetch_url.register(toolset)
    media_read.register(toolset)


@dataclass
class ChatResult:
    """Result of a single chat turn."""
    text: str
    steps: int  # how many agent steps were taken


class CognitAgent:
    """High-level SDK entry point for using Cognit programmatically.

    Wraps provider + toolset + agent into a single, easy-to-use object.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        max_steps: int = 50,
        max_tokens: int = 128_000,
        thinking: bool = False,
        thinking_budget: int = 10000,
        system_prompt: str | None = None,
        builtin_tools: bool = True,
        auto_config: bool = True,
    ) -> None:
        # --- Resolve config (CLI args > cognit.toml > env vars) ---
        resolved_model = model
        resolved_base_url = base_url
        resolved_api_key = api_key
        resolved_max_tokens = max_tokens

        if auto_config and (not resolved_base_url or not resolved_api_key):
            from cognit.config import load_config

            cfg = load_config()
            model_cfg = cfg.models.get("default")
            if model_cfg:
                prov_cfg = cfg.providers.get(model_cfg.provider)
                if prov_cfg:
                    resolved_base_url = resolved_base_url or prov_cfg.base_url
                    resolved_api_key = resolved_api_key or prov_cfg.api_key
                    if model == "gpt-4o":  # only override if caller didn't specify
                        resolved_model = model_cfg.model
                        resolved_max_tokens = model_cfg.max_context_size

        # --- Provider ---
        self.provider = OpenAIProvider(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )

        # --- Toolset ---
        self.toolset = Toolset()
        if builtin_tools:
            _register_builtin_tools(self.toolset)

        # --- System prompt (with optional skills) ---
        from datetime import datetime

        base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(
            cwd=os.getcwd(),
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Load Agent Skills
        try:
            from cognit.soul.skills import load_all_skills, skills_to_prompt_section
            skills = load_all_skills()
            skills_section = skills_to_prompt_section(skills)
            if skills_section:
                base_prompt += skills_section
                logger.info("Loaded %d skill(s) into system prompt", len(skills))
        except Exception as e:
            logger.warning("Failed to load skills: %s", e)

        self.system_prompt = base_prompt

        # --- Agent ---
        agent_config = AgentConfig(max_steps=max_steps, max_tokens=resolved_max_tokens)
        self.agent = Agent(self.provider, self.toolset, self.system_prompt, agent_config)

    # -- Convenience: delegate tool registration --------------------------------

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        requires_approval: bool = False,
    ) -> Callable:
        """Decorator to register a custom tool."""
        return self.toolset.tool(name, description, parameters, requires_approval)

    # -- Core API ---------------------------------------------------------------

    async def chat(
        self,
        message: str,
        *,
        images: list[str] | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
        on_tool_start: Callable | None = None,
        on_tool_end: Callable | None = None,
        approval_callback: Callable | None = None,
    ) -> ChatResult:
        """Send a message and get the agent's response.

        Args:
            message: User message text.
            images: Optional list of image data URIs (base64) or URLs to attach.
            on_text_delta: Called with each text chunk as it streams.
            on_thinking_delta: Called with each thinking chunk (if thinking enabled).
            on_tool_start: Called when a tool execution begins. Receives (ToolCall).
            on_tool_end: Called when a tool execution ends. Receives (ToolCall, result_str).
            approval_callback: Called to approve dangerous tools.

        Returns:
            ChatResult with the final text and step count.
        """
        initial_messages = len(self.agent.context.messages)

        text = await self.agent.run(
            message,
            images=images,
            on_text_delta=on_text_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            approval_callback=approval_callback,
        )

        steps = sum(
            1 for m in self.agent.context.messages[initial_messages:]
            if m.role == "assistant"
        )

        return ChatResult(text=text, steps=steps)

    def clear(self) -> None:
        """Clear conversation context, starting fresh."""
        self.agent.context.clear()

    @property
    def context(self):
        """Access the conversation context (message history)."""
        return self.agent.context
