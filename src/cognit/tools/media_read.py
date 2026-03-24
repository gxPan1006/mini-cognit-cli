"""Tool: read image/media files and return as base64 for vision models."""

from __future__ import annotations

import base64
import mimetypes
import os

from cognit.soul.toolset import Toolset

TOOL_NAME = "read_media"
DESCRIPTION = (
    "Read an image file and return it as base64 for vision analysis. "
    "Supports PNG, JPEG, GIF, WebP, and SVG. "
    "Use this when you need to view or analyze an image file."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the image file to read.",
        },
    },
    "required": ["file_path"],
}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

SUPPORTED_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def read_media(file_path: str) -> str:
        path = os.path.expanduser(file_path)

        if not os.path.isfile(path):
            return f"Error: file not found: {path}"

        ext = os.path.splitext(path)[1].lower()
        mime_type = SUPPORTED_TYPES.get(ext)
        if not mime_type:
            # Try mimetypes as fallback
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith("image/"):
                supported = ", ".join(SUPPORTED_TYPES.keys())
                return f"Error: unsupported file type '{ext}'. Supported: {supported}"

        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            return f"Error: file too large ({file_size / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB"

        try:
            with open(path, "rb") as f:
                data = f.read()
        except PermissionError:
            return f"Error: permission denied: {path}"

        b64 = base64.b64encode(data).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

        return f"__IMAGE__:{data_uri}"
