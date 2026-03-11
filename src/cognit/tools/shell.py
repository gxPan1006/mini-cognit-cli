"""Tool: execute shell commands."""

from __future__ import annotations

import asyncio
import os
import signal

from cognit.soul.toolset import Toolset

TOOL_NAME = "shell"
DESCRIPTION = (
    "Execute a shell command and return its stdout/stderr. "
    "Use for system commands, git, build tools, etc. "
    "Set background=true for long-running processes (servers, watchers) — "
    "the command will be started and control returns immediately."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds. Default: 30.",
        },
        "background": {
            "type": "boolean",
            "description": "Run in background (for servers, watchers). Default: false.",
        },
    },
    "required": ["command"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(
        name=TOOL_NAME,
        description=DESCRIPTION,
        parameters=PARAMETERS,
        requires_approval=True,
    )
    async def shell(command: str, timeout: int = 30, background: bool = False) -> str:
        if background:
            # Start process detached, don't wait for it
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"Background process started (pid: {proc.pid})"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            return f"Command timed out after {timeout}s. Use background=true for long-running processes."
        except Exception as e:
            return f"Error running command: {e}"

        output_parts = []
        if stdout:
            out = stdout.decode("utf-8", errors="replace")
            if len(out) > 50_000:
                out = out[:25_000] + "\n\n... [truncated] ...\n\n" + out[-25_000:]
            output_parts.append(out)
        if stderr:
            err = stderr.decode("utf-8", errors="replace")
            if len(err) > 20_000:
                err = err[:10_000] + "\n\n... [truncated] ...\n\n" + err[-10_000:]
            output_parts.append(f"[stderr]\n{err}")

        result = "\n".join(output_parts) if output_parts else "(no output)"

        if proc.returncode != 0:
            result = f"[exit code: {proc.returncode}]\n{result}"

        return result
