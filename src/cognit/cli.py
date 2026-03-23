"""CLI entry point — defines the `cognit` command using Typer."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

cli = typer.Typer(
    name="cognit",
    help="mini-cognit-cli — A minimal CLI AI agent.",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> Path:
    """Configure logging to file. Returns the log file path."""
    log_dir = Path.home() / ".cognit" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


@cli.command()
def chat(
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or set OPENAI_API_KEY)"),
    max_steps: int = typer.Option(50, "--max-steps", help="Max agent loop steps per turn"),
    yolo: bool = typer.Option(False, "--yolo", "-y", help="Bypass all tool approval prompts (auto-approve everything)"),
    thinking: bool = typer.Option(False, "--thinking", "-t", help="Enable extended thinking (reasoning models)"),
    thinking_budget: int = typer.Option(10000, "--thinking-budget", help="Max tokens for thinking (default: 10000)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug-level logging"),
) -> None:
    """Start an interactive chat session with the AI agent."""
    log_file = _setup_logging(verbose)

    logger = logging.getLogger("cognit")
    logger.info("Session started — model=%s, max_steps=%d, yolo=%s, thinking=%s", model, max_steps, yolo, thinking)
    logger.info("Log file: %s", log_file)

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
    try:
        asyncio.run(app.run())
    except Exception:
        logger.exception("Unhandled exception in app.run()")
        raise
    finally:
        logger.info("Session ended")
        typer.echo(f"\n📋 Log saved to: {log_file}")


@cli.command()
def version() -> None:
    """Show version info."""
    from cognit import __version__
    typer.echo(f"mini-cognit-cli v{__version__}")


if __name__ == "__main__":
    cli()
