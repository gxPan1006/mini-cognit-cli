"""Configuration loading from TOML files and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomlkit
except ImportError:
    tomlkit = None  # type: ignore


CONFIG_FILENAME = "cognit.toml"


@dataclass
class ProviderConfig:
    type: str = "openai"
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    max_context_size: int = 128_000


@dataclass
class Config:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    max_steps: int = 50


def find_config_file() -> Path | None:
    """Search for cognit.toml in current dir, then ~/.config/cognit/."""
    candidates = [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".config" / "cognit" / CONFIG_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config() -> Config:
    """Load config from TOML file, with env var overrides."""
    config = Config()

    config_file = find_config_file()
    if config_file and tomlkit:
        with open(config_file) as f:
            data = tomlkit.load(f)

        for name, prov_data in data.get("providers", {}).items():
            config.providers[name] = ProviderConfig(
                type=prov_data.get("type", "openai"),
                base_url=prov_data.get("base_url"),
                api_key=prov_data.get("api_key"),
            )

        for name, model_data in data.get("models", {}).items():
            config.models[name] = ModelConfig(
                provider=model_data.get("provider", "openai"),
                model=model_data.get("model", "gpt-4o"),
                max_context_size=model_data.get("max_context_size", 128_000),
            )

    # Env var overrides
    if api_key := os.environ.get("OPENAI_API_KEY"):
        if "openai" not in config.providers:
            config.providers["openai"] = ProviderConfig()
        config.providers["openai"].api_key = api_key

    if base_url := os.environ.get("OPENAI_BASE_URL"):
        if "openai" not in config.providers:
            config.providers["openai"] = ProviderConfig()
        config.providers["openai"].base_url = base_url

    return config
