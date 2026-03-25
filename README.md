<p align="center">
  <h1 align="center">mini-cognit-cli</h1>
  <p align="center">
    <strong>The AI coding agent you can read in an afternoon.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/gxPan1006/mini-cognit-cli/stargazers"><img src="https://img.shields.io/github/stars/gxPan1006/mini-cognit-cli?style=flat&color=gold" alt="Stars"></a>
    <a href="https://pypi.org/project/mini-cognit-cli/"><img src="https://img.shields.io/pypi/v/mini-cognit-cli?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/mini-cognit-cli/"><img src="https://img.shields.io/pypi/dm/mini-cognit-cli?color=green" alt="Downloads"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+">
  </p>
</p>

<!-- TODO: Add demo GIF here -->
<!-- <p align="center"><img src="demo.gif" width="700" alt="demo"></p> -->

```bash
pip install mini-cognit-cli
```

---

## Why mini-cognit-cli?

Every AI coding agent (Claude Code, aider, Kimi CLI, Codex) is a black box with tens of thousands of lines of code. **mini-cognit-cli** takes the opposite approach:

- **~2000 lines of Python** — read the entire codebase, understand every decision
- **Add a custom tool in ~30 lines** — no plugin system, no abstractions, just a function
- **Any OpenAI-compatible API** — OpenAI, Anthropic (via proxy), Ollama, LM Studio, anything
- **Fork and make it yours** — designed to be the starting point for *your* agent

| | Claude Code | aider | Kimi CLI | **mini-cognit-cli** |
|---|---|---|---|---|
| Codebase | ~100k+ lines | ~50k+ lines | ~30k+ lines | **~2k lines** |
| Time to understand | Weeks | Days | Days | **An afternoon** |
| Add a custom tool | Plugin system | Moderate | Plugin / MCP | **~30 lines** |
| Provider lock-in | Anthropic | Any | Multi-provider | **Any OpenAI-compatible** |
| Language | TypeScript | Python | Python | **Python** |

## Quickstart

```bash
# Install
pip install "mini-cognit-cli[cli]"

# Configure (pick one)
export OPENAI_API_KEY="sk-..."                    # env var
# OR: create cognit.toml with provider config     # config file

# Go
cognit chat
```

That's it. The agent can read/write files, run shell commands, search code, browse the web, and more.

## Features

**Tools** — file read/write/edit, shell execution, grep, glob, web search, URL fetch, image reading

**Session management** — auto-save, `--continue` to resume, `--session <id>` to pick a specific session

**Image support** — `/paste-image` from clipboard, `/img <path>` to attach, vision models analyze images

**@ file references** — type `@filename` with Tab autocomplete to reference files in your message

**Agent Skills** — drop a `SKILL.md` in `.cognit/skills/` to teach the agent new workflows (compatible with Claude/Codex skill format)

**Context compaction** — auto-summarizes when approaching token limits, or `/compact` to manually compress

**Extended thinking** — `--thinking` flag for reasoning models

**Tool approval** — shell commands need confirmation; `--yolo` to auto-approve everything

## Usage

```bash
# Basic
cognit chat

# Specify model
cognit chat -m gpt-4o

# Resume last session
cognit chat --continue

# YOLO mode (auto-approve all tools)
cognit chat --yolo

# Extended thinking
cognit chat --thinking

# List saved sessions
cognit sessions
```

### In-session commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation and start new session |
| `/compact` | Manually compress context |
| `/sessions` | List saved sessions |
| `/paste-image` | Paste image from clipboard |
| `/img <path>` | Attach an image file |
| `@path/to/file` | Reference a file (Tab to autocomplete) |
| `/exit` | Quit (auto-saves session) |

## Configuration

Create `cognit.toml` in your project directory or `~/.config/cognit/`:

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

Priority: **CLI flags > cognit.toml > environment variables**.

## SDK usage

```python
from cognit import CognitAgent

agent = CognitAgent(model="gpt-4o", api_key="sk-...")

# Add custom tools
@agent.tool(name="my_tool", description="Does something")
async def my_tool(query: str) -> str:
    return f"result for {query}"

# Chat (context preserved between calls)
result = await agent.chat("What files are here?")
result = await agent.chat("Read the first one.")

# With image
result = await agent.chat("Describe this", images=["data:image/png;base64,..."])
```

## Architecture

```
src/cognit/          (~2000 lines total)
├── cli.py              ← CLI entry point (Typer)
├── app.py              ← REPL orchestrator
├── config.py           ← TOML config loading
├── sdk.py              ← Programmatic SDK interface
├── llm/
│   ├── provider.py     ← Abstract LLM interface
│   ├── openai_provider.py  ← OpenAI-compatible implementation
│   ├── generate.py     ← LLM call + streaming assembly
│   └── message.py      ← Message model (text, images, tool calls)
├── soul/
│   ├── agent.py        ← Core agent loop (the brain)
│   ├── context.py      ← Conversation history
│   ├── compaction.py   ← Context summarization
│   ├── session.py      ← Session persistence
│   ├── skills.py       ← Agent Skills (SKILL.md) loading
│   └── toolset.py      ← Tool registry + dispatch
├── tools/
│   ├── file_read.py    ← Read files
│   ├── file_write.py   ← Write/edit files
│   ├── glob_tool.py    ← File pattern matching
│   ├── grep.py         ← Search file contents
│   ├── shell.py        ← Shell execution
│   ├── web_search.py   ← Web search (DuckDuckGo)
│   ├── fetch_url.py    ← Fetch web pages
│   └── media_read.py   ← Read images for vision
└── ui/
    └── terminal.py     ← Terminal I/O (prompt-toolkit + rich)
```

Each module has a clear responsibility. The entire flow:

```
User input → Agent.run() → LLM call → Tool execution → Loop until done → Response
```

## Adding a custom tool

```python
# src/cognit/tools/my_tool.py
from cognit.soul.toolset import Toolset

def register(toolset: Toolset) -> None:
    @toolset.tool(name="my_tool", description="Does something useful.")
    async def my_tool(query: str, limit: int = 10) -> str:
        return f"Result: {query} (limit={limit})"
```

Register in `sdk.py`'s `_register_builtin_tools()`:

```python
from cognit.tools import my_tool
my_tool.register(toolset)
```

That's it. The decorator auto-infers JSON Schema from type hints.

## Agent Skills

Drop a `SKILL.md` file in `.cognit/skills/your-skill/SKILL.md`:

```markdown
---
name: code-review
---
When asked to review code, follow these steps:
1. Read the changed files
2. Check for bugs, security issues, and style problems
3. Suggest improvements with code examples
```

Skills are auto-discovered from `.cognit/skills/`, `~/.cognit/skills/`, and compatible with `.claude/skills/` and `.codex/skills/`.

## Requirements

- Python >= 3.12
- Works with any OpenAI-compatible API (OpenAI, Anthropic via proxy, Ollama, LM Studio, etc.)

## License

[MIT](LICENSE)
