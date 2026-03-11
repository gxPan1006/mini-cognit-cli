"""OpenAI-compatible ChatProvider — works with OpenAI, local models, and any compatible API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Sequence

import openai

from cognit.llm.message import Message

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]  # seconds


class OpenAIProvider:
    """ChatProvider backed by OpenAI's Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[dict[str, Any]],
        history: Sequence[Message],
    ) -> AsyncIterator[dict[str, Any]]:
        """Call OpenAI Chat Completions (streaming) and yield normalized deltas."""

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(m.to_openai() for m in history)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        # Retry on transient errors (400/429/5xx)
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = await self._client.chat.completions.create(**kwargs)

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    # Text content
                    if delta.content:
                        yield {"type": "text_delta", "text": delta.content}

                    # Tool call deltas
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            yield {
                                "type": "tool_call_delta",
                                "index": tc_delta.index,
                                "id": tc_delta.id or "",
                                "name": (tc_delta.function.name if tc_delta.function else "") or "",
                                "arguments": (tc_delta.function.arguments if tc_delta.function else "") or "",
                            }

                yield {"type": "done"}
                return  # success, exit retry loop

            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as e:
                last_error = e
                status = getattr(e, "status_code", None)
                # Don't retry on client errors (auth, bad request, etc.)
                if status and 400 <= status < 500:
                    raise
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_error  # type: ignore[misc]
