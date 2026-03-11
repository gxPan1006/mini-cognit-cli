"""Tool: write or edit files on disk."""

from __future__ import annotations

import os

from cognit.soul.toolset import Toolset

WRITE_NAME = "write_file"
WRITE_DESC = "Write content to a file. Creates the file if it doesn't exist."
WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "The full content to write to the file.",
        },
    },
    "required": ["file_path", "content"],
}

EDIT_NAME = "edit_file"
EDIT_DESC = "Replace an exact string in a file with new content. The old_string must be unique in the file."
EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the file to edit.",
        },
        "old_string": {
            "type": "string",
            "description": "The exact string to find and replace (must be unique in the file).",
        },
        "new_string": {
            "type": "string",
            "description": "The replacement string.",
        },
    },
    "required": ["file_path", "old_string", "new_string"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=WRITE_NAME, description=WRITE_DESC, parameters=WRITE_PARAMS)
    async def write_file(file_path: str, content: str) -> str:
        path = os.path.expanduser(file_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} chars to {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"

    @toolset.tool(name=EDIT_NAME, description=EDIT_DESC, parameters=EDIT_PARAMS)
    async def edit_file(file_path: str, old_string: str, new_string: str) -> str:
        path = os.path.expanduser(file_path)
        if not os.path.isfile(path):
            return f"Error: file not found: {path}"

        with open(path, encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file."
        if count > 1:
            return f"Error: old_string found {count} times — must be unique. Provide more context."

        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully edited {path}"
