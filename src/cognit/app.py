"""App — the orchestrator that wires everything together."""

from __future__ import annotations

import os

from cognit.llm.openai_provider import OpenAIProvider
from cognit.soul.agent import Agent, AgentConfig
from cognit.soul.toolset import Toolset
from cognit.tools import file_read, file_write, shell, grep, web_search
from cognit.ui.terminal import TerminalUI


SYSTEM_PROMPT = """\
You are Cognit, a powerful AI coding assistant running in the user's terminal.
You help users with software engineering tasks: writing code, debugging, exploring codebases, and more.

## Working principles
- Read files before modifying them.
- Prefer editing existing files over creating new ones.
- Use the shell tool for system commands (git, build tools, etc.).
- Use grep to search codebases efficiently.
- Use web_search to look up information online — do NOT use shell/curl to scrape websites.
- Be concise and direct. Lead with the answer, not the reasoning.
- When you complete a task, briefly confirm what you did.

## Current directory
{cwd}
"""


class App:
    """Top-level application that wires provider + tools + agent + UI."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 50,
        yolo: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_steps = max_steps
        self.yolo = yolo

    async def run(self) -> None:
        # --- 1. Create LLM provider (CLI args > cognit.toml > env vars) ---
        model = self.model
        base_url = self.base_url
        api_key = self.api_key

        if not base_url or not api_key:
            from cognit.config import load_config
            cfg = load_config()
            # Find the default model config, fall back to first available
            model_cfg = cfg.models.get("default")
            if model_cfg:
                prov_cfg = cfg.providers.get(model_cfg.provider)
                if prov_cfg:
                    base_url = base_url or prov_cfg.base_url
                    api_key = api_key or prov_cfg.api_key
                    if model == "gpt-4o":  # only override if user didn't specify
                        model = model_cfg.model

        provider = OpenAIProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        # --- 2. Build toolset ---
        toolset = Toolset()
        file_read.register(toolset)
        file_write.register(toolset)
        shell.register(toolset)
        grep.register(toolset)
        web_search.register(toolset)

        # --- 3. Build system prompt ---
        system_prompt = SYSTEM_PROMPT.format(cwd=os.getcwd())

        # --- 4. Create agent ---
        config = AgentConfig(max_steps=self.max_steps)
        agent = Agent(provider, toolset, system_prompt, config)

        # --- 5. Create UI ---
        ui = TerminalUI()
        mode_label = f"{model} [yellow bold]YOLO[/yellow bold]" if self.yolo else model
        ui.print_welcome(mode_label)

        # --- 6. Main REPL loop ---
        while True:
            try:
                user_input = await ui.get_input()
            except KeyboardInterrupt:
                break

            if not user_input:
                continue

            # --- slash commands ---
            if user_input.startswith("/"):
                cmd = user_input.lower().strip()
                if cmd in ("/exit", "/quit", "/q"):
                    ui.print_info("Goodbye!")
                    break
                elif cmd == "/clear":
                    agent.context.clear()
                    ui.print_info("Context cleared.")
                    continue
                elif cmd == "/help":
                    ui.console.print(
                        "  [bold]/exit[/bold]   — quit\n"
                        "  [bold]/clear[/bold]  — clear conversation context\n"
                        "  [bold]/help[/bold]   — show this help\n"
                    )
                    continue
                else:
                    ui.print_error(f"Unknown command: {cmd}")
                    continue

            # --- run agent ---
            ui.start_response()
            try:
                approval = None if self.yolo else ui.approval_callback
                await agent.run(
                    user_input,
                    on_text_delta=ui.on_text_delta,
                    on_tool_start=ui.on_tool_start,
                    on_tool_end=ui.on_tool_end,
                    approval_callback=approval,
                )
            except KeyboardInterrupt:
                ui.print_info("\n[interrupted]")
            except Exception as e:
                ui.print_error(f"{type(e).__name__}: {e}")
            finally:
                ui.end_response()
