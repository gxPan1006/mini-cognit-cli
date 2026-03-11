"""Tool: read a file from disk."""

from __future__ import annotations

import os

from cognit.soul.toolset import Toolset

TOOL_NAME = "read_file"
DESCRIPTION = "Read the contents of a file. Returns the file content with line numbers."
PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-based). Default: 1.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read. Default: 500.",
        },
    },
    "required": ["file_path"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def read_file(file_path: str, offset: int = 1, limit: int = 500) -> str:
        path = os.path.expanduser(file_path)
        if not os.path.isfile(path):
            return f"Error: file not found: {path}"

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except PermissionError:
            return f"Error: permission denied: {path}"

        total = len(lines)
        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")

        header = f"[{path}] ({total} lines total, showing {start+1}-{min(end, total)})"
        return header + "\n" + "\n".join(numbered)
