"""Session persistence — save and restore conversation sessions."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from cognit.llm.message import Message

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".cognit" / "sessions"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(
    session_id: str,
    messages: list[Message],
    model: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a session to disk. Returns the file path."""
    _ensure_dir()
    path = session_path(session_id)

    # Read existing to preserve created_at
    created_at = datetime.now().isoformat()
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
            created_at = existing.get("created_at", created_at)
        except (json.JSONDecodeError, OSError):
            pass

    data = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "model": model,
        "message_count": len(messages),
        **(metadata or {}),
        "messages": [m.to_dict() for m in messages],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Session saved: %s (%d messages)", session_id, len(messages))
    return path


def load_session(session_id: str) -> tuple[list[Message], dict[str, Any]]:
    """Load a session from disk. Returns (messages, metadata)."""
    path = session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    messages = [Message.from_dict(m) for m in data.get("messages", [])]
    metadata = {k: v for k, v in data.items() if k != "messages"}

    logger.info("Session loaded: %s (%d messages)", session_id, len(messages))
    return messages, metadata


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List recent sessions, sorted by last update (newest first)."""
    _ensure_dir()
    sessions = []

    for p in SESSIONS_DIR.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            sessions.append({
                "session_id": data.get("session_id", p.stem),
                "model": data.get("model", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": data.get("message_count", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions[:limit]


def get_latest_session_id() -> str | None:
    """Get the most recently updated session ID, or None."""
    sessions = list_sessions(limit=1)
    return sessions[0]["session_id"] if sessions else None


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if deleted."""
    path = session_path(session_id)
    if path.exists():
        path.unlink()
        logger.info("Session deleted: %s", session_id)
        return True
    return False
