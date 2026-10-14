"""Typed view of config/config.yaml. Unknown keys are rejected so a typo cannot silently fall back to a default."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    size: str = "4b"
    precision: str = "bf16"
    backend: str = "transformers"
    device: str = "cuda"
    max_tokens: int = 12000
    vllm_url: str | None = None


@dataclass(frozen=True)
class SearchConfig:
    backend: str = "ddgs"
    max_results: int = 8
    region: str = "us-en"
    engines: str = "bing, yahoo, duckduckgo"
    timeout_s: float = 10.0


@dataclass(frozen=True)
class ExtractConfig:
    min_chars: int = 40
    max_chars: int = 600


@dataclass(frozen=True)
class RankConfig:
    prefilter_k: int = 64
    abstain_threshold: float = 0.5


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    rank: RankConfig = field(default_factory=RankConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def _build(cls, raw: dict, where: str):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(unknown)} (allowed: {', '.join(known)})")
    missing = sorted(set(known) - set(raw))
    if missing:
        print(f"config: {where} uses defaults for {', '.join(missing)}")
    kwargs = {}
    for name, value in raw.items():
        sub = known[name].default_factory if known[name].default_factory is not dataclasses.MISSING else None
        kwargs[name] = _build(sub, value or {}, f"{where}.{name}") if dataclasses.is_dataclass(sub) else value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _build(Config, raw, "config")
