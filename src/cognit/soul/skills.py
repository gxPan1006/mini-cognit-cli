"""Agent Skills — load SKILL.md files and inject into the system prompt."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories to search for skills, in priority order
SKILL_SEARCH_PATHS = [
    Path.cwd() / ".cognit" / "skills",
    Path.home() / ".cognit" / "skills",
    # Compatibility with Claude Code and Codex
    Path.cwd() / ".claude" / "skills",
    Path.home() / ".claude" / "skills",
    Path.cwd() / ".codex" / "skills",
    Path.home() / ".codex" / "skills",
]


def discover_skills() -> list[Path]:
    """Find all SKILL.md files in known locations."""
    found: list[Path] = []
    seen: set[str] = set()

    for search_dir in SKILL_SEARCH_PATHS:
        if not search_dir.is_dir():
            continue
        for skill_file in search_dir.rglob("SKILL.md"):
            # Deduplicate by resolved path
            resolved = str(skill_file.resolve())
            if resolved not in seen:
                seen.add(resolved)
                found.append(skill_file)

    logger.info("Discovered %d skill(s)", len(found))
    for f in found:
        logger.debug("  skill: %s", f)
    return found


def load_skill(path: Path) -> tuple[str, str]:
    """Load a skill file. Returns (name, content).

    The name is derived from the parent directory name or the file path.
    If the skill file has YAML frontmatter (---...---), it is stripped.
    """
    content = path.read_text(encoding="utf-8").strip()

    # Derive name from directory structure
    # e.g. .cognit/skills/code-review/SKILL.md -> "code-review"
    name = path.parent.name if path.parent.name != "skills" else path.stem

    # Strip optional YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            # Extract name from frontmatter if present
            frontmatter = content[3:end]
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    extracted = line[5:].strip().strip('"').strip("'")
                    if extracted:
                        name = extracted
            content = content[end + 3:].strip()

    return name, content


def load_all_skills() -> list[tuple[str, str]]:
    """Discover and load all skills. Returns list of (name, content)."""
    skills = []
    for path in discover_skills():
        try:
            name, content = load_skill(path)
            skills.append((name, content))
            logger.info("Loaded skill: %s (%s)", name, path)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", path, e)
    return skills


def skills_to_prompt_section(skills: list[tuple[str, str]]) -> str:
    """Format loaded skills into a section for the system prompt."""
    if not skills:
        return ""

    parts = ["\n## Agent Skills\nThe following skills provide additional knowledge and workflows:\n"]
    for name, content in skills:
        parts.append(f"### {name}\n{content}\n")

    return "\n".join(parts)
