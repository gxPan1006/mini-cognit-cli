# mini-cognit-cli

A minimal, hackable CLI AI agent for software engineering — built to be understood, modified, and extended.

Unlike heavyweight AI coding tools, mini-cognit-cli is intentionally small (~1500 lines of Python) so you can read the entire codebase in an afternoon and make it your own.

## Features

- **Agentic loop** — the LLM autonomously calls tools, observes results, and iterates until the task is done
- **Built-in tools** — file read/write, shell execution, grep search, web search — everything needed for coding tasks
- **Tool approval** — dangerous operations (shell commands) require user confirmation; bypass with `--yolo` when you trust it
- **Context compaction** — automatically summarizes conversation history when approaching token limits, so long sessions don't break
- **Any OpenAI-compatible API** — works with OpenAI, Anthropic (via proxies like subrouter), local models, or any provider with an OpenAI-compatible endpoint
- **TOML config** — configure providers, models, and defaults in a simple `cognit.toml` file
- **Streaming output** — responses stream token-by-token with rich terminal formatting

## Quickstart

```bash
# Clone and install
git clone https://github.com/gxPan1006/mini-cognit-cli.git
cd mini-cognit-cli
pip install -e .

# Option 1: Configure via cognit.toml
cp cognit.toml.example cognit.toml
# Edit cognit.toml with your API key and preferred model

# Option 2: Use environment variables
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional

# Start chatting
cognit chat
```

## Usage

```bash
# Basic usage (reads from cognit.toml or env vars)
cognit chat

# Specify model and provider directly
cognit chat --model gpt-4o --api-key sk-... --base-url https://api.openai.com/v1

# YOLO mode — auto-approve all tool executions (no confirmation prompts)
cognit chat --yolo

# Limit agent loop steps per turn
cognit chat --max-steps 20

# Show version
cognit version
```

### In-session commands

| Command  | Description                    |
|----------|--------------------------------|
| `/help`  | Show available commands        |
| `/clear` | Clear conversation context     |
| `/exit`  | Quit the session               |

## Configuration

Create a `cognit.toml` in your project directory or `~/.config/cognit/cognit.toml`:

```toml
[providers.my_provider]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."

[models.default]
provider = "my_provider"
model = "gpt-4o"
max_context_size = 128000
```

Config priority: **CLI flags > cognit.toml > environment variables**.

## Architecture

```
src/cognit/
├── cli.py              # CLI entry point (Typer)
├── app.py              # Orchestrator — wires everything together
├── config.py           # TOML config loading
├── llm/
│   ├── provider.py     # Abstract LLM provider interface
│   ├── openai_provider.py  # OpenAI-compatible implementation
│   ├── generate.py     # LLM call + tool execution step
│   └── message.py      # Message, ToolCall, ToolResult types
├── soul/
│   ├── agent.py        # Core agent loop
│   ├── context.py      # Conversation context management
│   ├── compaction.py   # Context summarization
│   └── toolset.py      # Tool registry and dispatch
├── tools/
│   ├── file_read.py    # Read files with line numbers
│   ├── file_write.py   # Write/edit files
│   ├── grep.py         # Search file contents
│   ├── shell.py        # Shell command execution
│   └── web_search.py   # Web search via DuckDuckGo
└── ui/
    └── terminal.py     # Terminal I/O (prompt-toolkit + rich)
```

The design is modular — each component has a clear responsibility:

- **`soul/agent.py`** — the brain. Runs the LLM-tool loop until the task is complete.
- **`soul/toolset.py`** — tool registry with decorator-based registration and automatic JSON Schema inference.
- **`llm/`** — provider abstraction. Swap in any OpenAI-compatible API.
- **`tools/`** — each tool is a self-contained module. Adding a new tool is ~30 lines of code.

## Adding a custom tool

```python
# src/cognit/tools/my_tool.py
from cognit.soul.toolset import Toolset

def register(toolset: Toolset) -> None:
    @toolset.tool(
        name="my_tool",
        description="Does something useful.",
        requires_approval=False,  # set True for dangerous operations
    )
    async def my_tool(arg1: str, arg2: int = 10) -> str:
        # Your logic here
        return f"Result: {arg1}, {arg2}"
```

Then register it in `app.py`:

```python
from cognit.tools import my_tool
my_tool.register(toolset)
```

## Requirements

- Python >= 3.12
- Dependencies: `openai`, `prompt-toolkit`, `rich`, `typer`, `tomlkit`, `tiktoken`

## License

[MIT](LICENSE)
