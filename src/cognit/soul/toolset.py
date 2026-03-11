"""Toolset — tool registration, schema generation, and dispatch."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from cognit.llm.message import ToolCall, ToolResult


@dataclass
class ToolDef:
    """A registered tool definition."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[str]]
    requires_approval: bool = False


class Toolset:
    """Registry of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Awaitable[str]],
        requires_approval: bool = False,
    ) -> None:
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_approval=requires_approval,
        )

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        requires_approval: bool = False,
    ) -> Callable:
        """Decorator to register an async function as a tool."""
        def decorator(fn: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
            tool_name = name or fn.__name__
            tool_desc = description or (inspect.getdoc(fn) or "")
            tool_params = parameters or _infer_parameters(fn)
            self.register(tool_name, tool_desc, tool_params, fn, requires_approval)
            return fn
        return decorator

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool definitions for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in self._tools.values()
        ]

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        td = self._tools.get(tool_call.name)
        if td is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Error: unknown tool '{tool_call.name}'",
                is_error=True,
            )

        try:
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}
        except json.JSONDecodeError:
            # LLM may produce malformed JSON (e.g. two objects concatenated).
            # Try to extract the first valid JSON object.
            try:
                decoder = json.JSONDecoder()
                args, _ = decoder.raw_decode(tool_call.arguments)
            except (json.JSONDecodeError, ValueError) as e2:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    content=f"Error parsing tool arguments: {e2}",
                    is_error=True,
                )

        try:
            result = await td.handler(**args)
            return ToolResult(tool_call_id=tool_call.id, content=result)
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool execution error: {type(e).__name__}: {e}",
                is_error=True,
            )


def _infer_parameters(fn: Callable) -> dict[str, Any]:
    """Infer a basic JSON Schema from function signature type hints."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

    for param_name, param in sig.parameters.items():
        json_type = type_map.get(param.annotation, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
