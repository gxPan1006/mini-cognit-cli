"""Terminal UI — input with prompt-toolkit, output with rich."""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from cognit.llm.message import ToolCall

logger = logging.getLogger(__name__)


class FileAtCompleter(Completer):
    """Autocomplete file paths after '@' character."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        # Find the last '@' that starts a file reference
        at_pos = text.rfind("@")
        if at_pos == -1:
            return

        # The partial path typed after @
        partial = text[at_pos + 1:]

        # Don't complete if there's a space before the partial (it's a new word, not a path)
        if " " in partial and not partial.replace(" ", "").strip():
            return

        # Expand the directory part
        if "/" in partial:
            dir_part = os.path.dirname(partial)
            base_part = os.path.basename(partial)
            search_dir = Path(dir_part)
        else:
            dir_part = ""
            base_part = partial
            search_dir = Path(".")

        if not search_dir.is_dir():
            return

        try:
            entries = sorted(search_dir.iterdir())
        except PermissionError:
            return

        count = 0
        for entry in entries:
            name = entry.name
            # Skip hidden files
            if name.startswith("."):
                continue
            # Skip common noise
            if name in ("node_modules", "__pycache__", ".git", ".venv"):
                continue

            if dir_part:
                rel_path = f"{dir_part}/{name}"
            else:
                rel_path = name

            if not name.lower().startswith(base_part.lower()):
                continue

            # Add trailing / for directories
            display = name + ("/" if entry.is_dir() else "")
            yield Completion(
                rel_path,
                start_position=-len(partial),
                display=display,
                display_meta="dir" if entry.is_dir() else "",
            )

            count += 1
            if count >= 50:
                break


def _grab_clipboard_image() -> str | None:
    """Try to grab an image from the macOS clipboard. Returns base64 data URI or None."""
    try:
        # Check if clipboard has image data
        result = subprocess.run(
            ["osascript", "-e", "clipboard info"],
            capture_output=True, text=True, timeout=5,
        )
        if "«class PNGf»" not in result.stdout and "«class TIFF»" not in result.stdout:
            return None

        # Extract PNG data from clipboard via a temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        # Use osascript to save clipboard image
        script = f'''
        set pngData to the clipboard as «class PNGf»
        set filePath to POSIX file "{tmp_path}"
        set fileRef to open for access filePath with write permission
        write pngData to fileRef
        close access fileRef
        '''
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, timeout=10,
        )

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "rb") as f:
                data = f.read()
            os.unlink(tmp_path)
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:image/png;base64,{b64}"
        else:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None
    except Exception as e:
        logger.warning("Failed to grab clipboard image: %s", e)
        return None


class TerminalUI:
    """Handles all terminal I/O for the agent."""

    def __init__(self) -> None:
        self.console = Console()
        self._completer = FileAtCompleter()
        self._session = PromptSession(completer=self._completer)
        self._current_text = ""
        self._streaming = False
        self._needs_newline = False
        self._in_thinking = False
        self._pending_images: list[str] = []  # base64 data URIs to attach

    def print_welcome(self, model: str, session_id: str = "") -> None:
        session_info = f"\nSession: [yellow]{session_id}[/yellow]" if session_id else ""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]mini-cognit-cli[/bold cyan]  v0.1.0\n"
                f"Model: [green]{model}[/green]{session_info}\n"
                f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.\n"
                f"Use [bold]@path[/bold] to reference files, [bold]/paste-image[/bold] to paste from clipboard.",
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

    def take_pending_images(self) -> list[str]:
        """Take and clear any pending images. Returns list of base64 data URIs."""
        images = self._pending_images[:]
        self._pending_images.clear()
        return images

    def paste_image_from_clipboard(self) -> bool:
        """Try to paste an image from clipboard. Returns True if successful."""
        data_uri = _grab_clipboard_image()
        if data_uri:
            self._pending_images.append(data_uri)
            self.console.print("  [green]Image pasted from clipboard[/green]")
            return True
        else:
            self.console.print("  [yellow]No image found in clipboard[/yellow]")
            return False

    def attach_image_file(self, file_path: str) -> bool:
        """Attach an image file. Returns True if successful."""
        path = os.path.expanduser(file_path)
        if not os.path.isfile(path):
            self.console.print(f"  [red]File not found: {path}[/red]")
            return False

        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(ext)
        if not mime_type:
            self.console.print(f"  [red]Unsupported image format: {ext}[/red]")
            return False

        try:
            with open(path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            data_uri = f"data:{mime_type};base64,{b64}"
            self._pending_images.append(data_uri)
            self.console.print(f"  [green]Image attached: {os.path.basename(path)}[/green]")
            return True
        except Exception as e:
            self.console.print(f"  [red]Error reading image: {e}[/red]")
            return False

    def start_response(self) -> None:
        """Called when the agent starts responding."""
        self._current_text = ""
        self._streaming = True
        self._needs_newline = False
        self.console.print()

    def on_thinking_delta(self, text: str) -> None:
        """Stream thinking/reasoning content in dimmed style."""
        if not self._in_thinking:
            self._in_thinking = True
            print("\033[2m", end="", flush=True)
        print(text, end="", flush=True)
        self._needs_newline = not text.endswith("\n")

    def _end_thinking(self) -> None:
        """Close thinking block if active."""
        if self._in_thinking:
            self._in_thinking = False
            self._ensure_newline()
            print("\033[0m", end="", flush=True)
            self.console.print("  [dim]───[/dim]")

    def on_text_delta(self, text: str) -> None:
        """Stream text as it arrives from the LLM."""
        self._end_thinking()
        print(text, end="", flush=True)
        self._current_text += text
        self._needs_newline = not text.endswith("\n")

    def _ensure_newline(self) -> None:
        if self._needs_newline:
            print()
            self._needs_newline = False

    def end_response(self) -> None:
        """Called when the agent finishes responding."""
        if self._streaming:
            self._end_thinking()
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
        self.console.print()

    def approval_callback(self, tc: ToolCall) -> bool:
        """Ask user for approval before running a dangerous tool."""
        self._ensure_newline()
        self.console.print(
            f"\n  [bold yellow]⚠ Approve:[/bold yellow] [bold]{tc.name}[/bold]"
        )
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
