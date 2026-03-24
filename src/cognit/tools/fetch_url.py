"""Tool: fetch and extract content from a URL."""

from __future__ import annotations

import re

from cognit.soul.toolset import Toolset

TOOL_NAME = "fetch_url"
DESCRIPTION = (
    "Fetch a webpage or API endpoint and return its content. "
    "For HTML pages, extracts readable text. For JSON APIs, returns raw JSON. "
    "Use this to read documentation, API responses, or any web content."
)
PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to fetch.",
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum content length in characters. Default: 50000.",
        },
    },
    "required": ["url"],
}


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping tags and scripts."""
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
    # Remove HTML comments
    html = re.sub(r"<!--[\s\S]*?-->", "", html)
    # Replace block elements with newlines
    html = re.sub(r"<(?:br|p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode common entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    lines = [line.strip() for line in html.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def register(toolset: Toolset) -> None:
    @toolset.tool(name=TOOL_NAME, description=DESCRIPTION, parameters=PARAMETERS)
    async def fetch_url(url: str, max_length: int = 50000) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx is not installed. Run: pip install httpx"

        if not url.startswith(("http://", "https://")):
            return f"Error: invalid URL (must start with http:// or https://): {url}"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CognitAgent/0.1)",
                })
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e}"
        except httpx.RequestError as e:
            return f"Request error: {type(e).__name__}: {e}"

        content_type = resp.headers.get("content-type", "")
        body = resp.text

        # For HTML, extract readable text
        if "html" in content_type:
            body = _extract_text_from_html(body)

        if len(body) > max_length:
            body = body[:max_length] + f"\n\n[truncated — {len(body)} total chars, showing first {max_length}]"

        return f"[{url}] ({resp.status_code}, {content_type})\n\n{body}"
