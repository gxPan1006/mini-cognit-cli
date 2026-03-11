"""Tool: web search via DuckDuckGo."""

from __future__ import annotations

from cognit.soul.toolset import Toolset

TOOL_NAME = "web_search"
DESCRIPTION = (
    "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
    "Use this to look up documentation, find solutions, or research topics."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return. Default: 5.",
        },
    },
    "required": ["query"],
}


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def web_search(query: str, max_results: int = 5) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            return "Error: ddgs is not installed. Run: pip install ddgs"

        try:
            results = list(DDGS().text(query, max_results=max_results))
        except Exception as e:
            return f"Search error: {type(e).__name__}: {e}"

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"{i}. {title}\n   {url}\n   {body}")

        return "\n\n".join(lines)
