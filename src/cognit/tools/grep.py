"""Tool: search file contents with regex."""

from __future__ import annotations

import os
import re

from cognit.soul.toolset import Toolset

TOOL_NAME = "grep"
DESCRIPTION = "Search for a regex pattern in files. Returns matching lines with file paths and line numbers."
PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in. Default: current directory.",
        },
        "glob_filter": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g. '*.py', '*.ts'). Default: all files.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of matches to return. Default: 50.",
        },
    },
    "required": ["pattern"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def grep(
        pattern: str,
        path: str = ".",
        glob_filter: str = "",
        max_results: int = 50,
    ) -> str:
        import fnmatch

        search_path = os.path.expanduser(path)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        matches: list[str] = []

        if os.path.isfile(search_path):
            _search_file(search_path, regex, matches, max_results)
        elif os.path.isdir(search_path):
            for root, _dirs, files in os.walk(search_path):
                # Skip hidden dirs and common noise
                _dirs[:] = [d for d in _dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", ".git")]
                for fname in files:
                    if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                        continue
                    if len(matches) >= max_results:
                        break
                    fpath = os.path.join(root, fname)
                    _search_file(fpath, regex, matches, max_results)
                if len(matches) >= max_results:
                    break
        else:
            return f"Error: path not found: {search_path}"

        if not matches:
            return "No matches found."

        header = f"Found {len(matches)} match(es):"
        return header + "\n" + "\n".join(matches)


def _search_file(fpath: str, regex: re.Pattern, matches: list[str], limit: int) -> None:
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if len(matches) >= limit:
                    return
                if regex.search(line):
                    matches.append(f"{fpath}:{i}: {line.rstrip()}")
    except (PermissionError, OSError):
        pass
