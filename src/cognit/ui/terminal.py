"""Terminal UI — input with prompt-toolkit, output with rich."""

from __future__ import annotations

import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from cognit.llm.message import ToolCall


class TerminalUI:
    """Handles all terminal I/O for the agent."""

    def __init__(self) -> None:
        self.console = Console()
        self._session = PromptSession()
        self._current_text = ""
        self._streaming = False
        self._needs_newline = False  # track if we need a newline before tool output

    def print_welcome(self, model: str) -> None:
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]mini-cognit-cli[/bold cyan]  v0.1.0\n"
                f"Model: [green]{model}[/green]\n"
                f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
                title="cognit",
                border_style="cyan",
            )
        )
        self.console.print()

    async def get_input(self) -> str:
        """Get user input with prompt-toolkit (supports multi-line with Alt+Enter)."""
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        try:
            text = await self._session.prompt_async(
                HTML("<cyan><b>❯ </b></cyan>"),
                multiline=False,
                key_bindings=bindings,
            )
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            return "/exit"

    def start_response(self) -> None:
        """Called when the agent starts responding."""
        self._current_text = ""
        self._streaming = True
        self._needs_newline = False
        self.console.print()

    def on_text_delta(self, text: str) -> None:
        """Stream text as it arrives from the LLM."""
        print(text, end="", flush=True)
        self._current_text += text
        self._needs_newline = not text.endswith("\n")

    def _ensure_newline(self) -> None:
        """Make sure we're on a fresh line before printing structured output."""
        if self._needs_newline:
            print()
            self._needs_newline = False

    def end_response(self) -> None:
        """Called when the agent finishes responding."""
        if self._streaming:
            self._ensure_newline()
            self._streaming = False
            self.console.print()

    def on_tool_start(self, tc: ToolCall) -> None:
        """Display tool execution start."""
        self._ensure_newline()
        args_preview = tc.arguments[:120] + ("..." if len(tc.arguments) > 120 else "")
        self.console.print(
            f"\n  [yellow]⚙ {escape(tc.name)}[/yellow]  [dim]{escape(args_preview)}[/dim]",
        )

    def on_tool_end(self, tc: ToolCall, result: str) -> None:
        """Display tool execution result (compact preview)."""
        lines = result.split("\n")
        preview_lines = lines[:6]
        preview = "\n    ".join(preview_lines)
        if len(lines) > 6:
            preview += f"\n    [dim]... ({len(lines) - 6} more lines)[/dim]"
        self.console.print(f"    [dim green]{escape(preview)}[/dim green]")
        self.console.print()  # blank line after tool result

    def approval_callback(self, tc: ToolCall) -> bool:
        """Ask user for approval before running a dangerous tool."""
        self._ensure_newline()
        self.console.print(
            f"\n  [bold yellow]⚠ Approve:[/bold yellow] [bold]{tc.name}[/bold]"
        )
        # Show command preview for shell
        args_preview = tc.arguments[:300]
        self.console.print(f"    [dim]{escape(args_preview)}[/dim]")
        try:
            answer = input("    Allow? [Y/n]: ").strip().lower()
            return answer in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def print_error(self, msg: str) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {escape(msg)}")

    def print_info(self, msg: str) -> None:
        self.console.print(f"[dim]{escape(msg)}[/dim]")
