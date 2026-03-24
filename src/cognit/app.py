"""App — the CLI REPL that wires CognitAgent + TerminalUI together."""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

from cognit.sdk import CognitAgent
from cognit.soul.session import (
    new_session_id, save_session, load_session, list_sessions, get_latest_session_id,
)
from cognit.ui.terminal import TerminalUI


def _expand_at_references(text: str) -> str:
    """Expand @file references in user input.

    @path/to/file is replaced with the file content inline.
    Keeps original @reference as a label so the LLM knows what file it came from.
    """
    pattern = r"@([\w./\-~][\w./\-~]*)"
    matches = re.findall(pattern, text)
    if not matches:
        return text

    for ref in matches:
        path = os.path.expanduser(ref)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                # Truncate very large files
                if len(content) > 100_000:
                    content = content[:100_000] + "\n[... truncated, file too large ...]"
                replacement = f"@{ref}\n```\n{content}\n```"
                text = text.replace(f"@{ref}", replacement, 1)
            except (PermissionError, OSError) as e:
                logger.warning("Failed to read @reference %s: %s", ref, e)
        # If not a file, leave the @reference as-is (might be a mention or email)

    return text


class App:
    """Top-level CLI application: wraps CognitAgent with a terminal REPL."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 50,
        yolo: bool = False,
        thinking: bool = False,
        thinking_budget: int = 10000,
        session_id: str | None = None,
        resume: bool = False,
    ) -> None:
        self.yolo = yolo
        self.thinking = thinking

        self.agent = CognitAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )

        # --- Session ---
        self._session_id: str
        if resume:
            # Resume most recent session
            latest = get_latest_session_id()
            if latest:
                self._session_id = latest
                self._restore_session(latest)
            else:
                self._session_id = new_session_id()
                logger.info("No previous session to resume, starting new: %s", self._session_id)
        elif session_id:
            self._session_id = session_id
            self._restore_session(session_id)
        else:
            self._session_id = new_session_id()

    def _restore_session(self, session_id: str) -> None:
        """Restore conversation context from a saved session."""
        try:
            messages, metadata = load_session(session_id)
            self.agent.agent.context._messages = messages
            logger.info("Restored session %s (%d messages)", session_id, len(messages))
        except FileNotFoundError:
            logger.warning("Session %s not found, starting fresh", session_id)
            self._session_id = new_session_id()
        except Exception as e:
            logger.error("Failed to restore session %s: %s", session_id, e)
            self._session_id = new_session_id()

    def _save_session(self) -> None:
        """Save current session to disk."""
        try:
            save_session(
                self._session_id,
                self.agent.agent.context.messages,
                model=self.agent.provider.model_name,
            )
        except Exception as e:
            logger.error("Failed to save session: %s", e)

    async def run(self) -> None:
        # --- UI ---
        ui = TerminalUI()
        mode_label = self.agent.provider.model_name
        if self.thinking:
            mode_label += " [magenta bold]THINKING[/magenta bold]"
        if self.yolo:
            mode_label += " [yellow bold]YOLO[/yellow bold]"
        ui.print_welcome(mode_label, session_id=self._session_id)

        # Show restored messages count
        msg_count = len(self.agent.agent.context.messages)
        if msg_count > 0:
            ui.print_info(f"Restored {msg_count} messages from previous session.")

        logger.info(
            "REPL loop starting — model=%s, max_steps=%d, session=%s",
            self.agent.provider.model_name,
            self.agent.agent.config.max_steps,
            self._session_id,
        )

        # --- Main REPL loop ---
        turn = 0
        while True:
            try:
                user_input = await ui.get_input()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt during input, exiting REPL")
                break

            if not user_input:
                continue

            # --- slash commands ---
            if user_input.startswith("/"):
                cmd = user_input.strip()
                cmd_lower = cmd.lower()
                logger.info("Slash command: %s", cmd)

                if cmd_lower in ("/exit", "/quit", "/q"):
                    self._save_session()
                    ui.print_info("Session saved. Goodbye!")
                    break

                elif cmd_lower == "/clear":
                    self.agent.clear()
                    self._session_id = new_session_id()
                    ui.print_info(f"Context cleared. New session: {self._session_id}")
                    continue

                elif cmd_lower == "/compact":
                    ui.print_info("Compacting context...")
                    from cognit.soul.compaction import compact_context
                    ctx = self.agent.agent.context
                    old_tokens = ctx.estimated_tokens()
                    # Force compaction regardless of threshold
                    if len(ctx.messages) > 4:
                        from cognit.llm.generate import generate
                        from cognit.llm.message import Message
                        from cognit.soul.compaction import COMPACTION_PROMPT

                        history_text = []
                        for m in ctx.messages:
                            prefix = m.role.upper()
                            history_text.append(f"[{prefix}]: {m.text}")
                            for tc in m.tool_calls:
                                history_text.append(f"  -> tool_call: {tc.name}({tc.arguments[:200]})")
                        summary_request = COMPACTION_PROMPT + "\n\n" + "\n".join(history_text)
                        result = await generate(
                            provider=self.agent.provider,
                            system_prompt="You are a helpful summarizer.",
                            tools=[],
                            history=[Message.user(summary_request)],
                        )
                        ctx.replace_with_summary(result.message.text)
                        new_tokens = ctx.estimated_tokens()
                        ui.print_info(f"Compacted: {old_tokens} → {new_tokens} tokens (kept last 4 messages + summary)")
                    else:
                        ui.print_info("Not enough messages to compact.")
                    continue

                elif cmd_lower == "/help":
                    ui.console.print(
                        "  [bold]/exit[/bold]         — quit (auto-saves session)\n"
                        "  [bold]/clear[/bold]        — clear conversation context\n"
                        "  [bold]/compact[/bold]      — manually compact/summarize context\n"
                        "  [bold]/sessions[/bold]     — list saved sessions\n"
                        "  [bold]/paste-image[/bold]  — paste image from clipboard\n"
                        "  [bold]/img <path>[/bold]   — attach an image file\n"
                        "  [bold]/help[/bold]         — show this help\n"
                        "\n"
                        "  [bold]@path/to/file[/bold] — reference a file inline (autocomplete with Tab)\n"
                    )
                    continue

                elif cmd_lower == "/sessions":
                    sessions = list_sessions(limit=10)
                    if not sessions:
                        ui.print_info("No saved sessions.")
                    else:
                        ui.console.print("  [bold]Recent sessions:[/bold]")
                        for s in sessions:
                            active = " [green]← current[/green]" if s["session_id"] == self._session_id else ""
                            ui.console.print(
                                f"  [cyan]{s['session_id']}[/cyan]  "
                                f"[dim]{s.get('model', '?')}[/dim]  "
                                f"{s.get('message_count', 0)} msgs  "
                                f"[dim]{s.get('updated_at', '')[:19]}[/dim]{active}"
                            )
                    continue

                elif cmd_lower == "/paste-image":
                    ui.paste_image_from_clipboard()
                    continue

                elif cmd_lower.startswith("/img "):
                    img_path = cmd[5:].strip()
                    ui.attach_image_file(img_path)
                    continue

                else:
                    ui.print_error(f"Unknown command: {cmd}")
                    continue

            # --- expand @ references ---
            user_input = _expand_at_references(user_input)

            # --- collect pending images ---
            images = ui.take_pending_images()

            # --- run agent ---
            turn += 1
            logger.info("=== Turn %d start === user_input=%r", turn, user_input[:200])
            ui.start_response()
            try:
                approval = None if self.yolo else ui.approval_callback
                await self.agent.chat(
                    user_input,
                    images=images or None,
                    on_text_delta=ui.on_text_delta,
                    on_thinking_delta=ui.on_thinking_delta if self.thinking else None,
                    on_tool_start=ui.on_tool_start,
                    on_tool_end=ui.on_tool_end,
                    approval_callback=approval,
                )
                logger.info("=== Turn %d completed normally ===", turn)
            except KeyboardInterrupt:
                logger.info("=== Turn %d interrupted by user (KeyboardInterrupt) ===", turn)
                ui.print_info("\n[interrupted]")
                self.agent.context.repair()
            except Exception as e:
                logger.exception("=== Turn %d failed with exception ===", turn)
                ui.print_error(f"{type(e).__name__}: {e}")
                self.agent.context.repair()
            finally:
                ui.end_response()

            # Auto-save after each turn
            self._save_session()
