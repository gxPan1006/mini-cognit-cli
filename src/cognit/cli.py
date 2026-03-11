"""CLI entry point — defines the `cognit` command using Typer."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer

cli = typer.Typer(
    name="cognit",
    help="mini-cognit-cli — A minimal CLI AI agent.",
    add_completion=False,
)


@cli.command()
def chat(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or set OPENAI_API_KEY)"),
    max_steps: int = typer.Option(50, "--max-steps", help="Max agent loop steps per turn"),
    yolo: bool = typer.Option(False, "--yolo", "-y", help="Bypass all tool approval prompts (auto-approve everything)"),
    thinking: bool = typer.Option(False, "--thinking", "-t", help="Enable extended thinking (reasoning models)"),
    thinking_budget: int = typer.Option(10000, "--thinking-budget", help="Max tokens for thinking (default: 10000)"),
) -> None:
    """Start an interactive chat session with the AI agent."""
    from cognit.app import App

    app = App(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_steps=max_steps,
        yolo=yolo,
        thinking=thinking,
        thinking_budget=thinking_budget,
    )
    asyncio.run(app.run())


@cli.command()
def version() -> None:
    """Show version info."""
    from cognit import __version__
    typer.echo(f"mini-cognit-cli v{__version__}")


if __name__ == "__main__":
    cli()
