"""mini-cognit-cli — A minimal CLI AI agent, also usable as a Python SDK."""

__version__ = "0.2.1"

from cognit.sdk import CognitAgent, ChatResult
from cognit.soul.toolset import Toolset
from cognit.soul.agent import AgentConfig
from cognit.llm.message import Message, ToolCall, ToolResult
from cognit.llm.provider import ChatProvider
from cognit.llm.openai_provider import OpenAIProvider

__all__ = [
    "CognitAgent",
    "ChatResult",
    "Toolset",
    "AgentConfig",
    "ChatProvider",
    "OpenAIProvider",
    "Message",
    "ToolCall",
    "ToolResult",
]
