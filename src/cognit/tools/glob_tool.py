"""Tool: glob file pattern matching."""

from __future__ import annotations

import os
from pathlib import Path

from cognit.soul.toolset import Toolset

TOOL_NAME = "glob"
DESCRIPTION = (
    "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). "
    "Returns matching file paths sorted by modification time (newest first). "
    "Use this to discover files by name or extension."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files (e.g. '**/*.py', 'src/*.ts', '*.json').",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in. Default: current directory.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Default: 100.",
        },
    },
    "required": ["pattern"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def glob(pattern: str, path: str = ".", max_results: int = 100) -> str:
        search_path = Path(os.path.expanduser(path)).resolve()

        if not search_path.is_dir():
            return f"Error: directory not found: {search_path}"

        try:
            matches = list(search_path.glob(pattern))
        except ValueError as e:
            return f"Error: invalid glob pattern: {e}"

        # Filter out hidden dirs and common noise
        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox"}
        filtered = []
        for m in matches:
            parts = m.relative_to(search_path).parts
            if any(p in skip_dirs or p.startswith(".") for p in parts[:-1]):
                continue
            if m.is_file():
                filtered.append(m)

        # Sort by modification time (newest first)
        filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        filtered = filtered[:max_results]

        if not filtered:
            return f"No files matching '{pattern}' in {search_path}"

        lines = []
        for f in filtered:
            rel = f.relative_to(search_path)
            lines.append(str(rel))

        header = f"Found {len(filtered)} file(s) matching '{pattern}':"
        return header + "\n" + "\n".join(lines)
