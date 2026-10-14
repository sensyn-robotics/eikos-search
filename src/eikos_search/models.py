"""Registry of the Eikos builds: the one place that maps (size, precision) to a checkpoint and loads it."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig

HF_OWNER = "caiovicentino1"
BACKENDS = ("transformers", "vllm")


@dataclass(frozen=True)
class EikosBuild:
    size: str
    precision: str
    repo: str
    weights_gb: float

    @property
    def name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def backends(self) -> tuple[str, ...]:
        # fp8/int4 are compressed-tensors checkpoints; the upstream LetterAdapter loads local weights as bf16 only.
        return BACKENDS if self.precision == "bf16" else ("vllm",)

    def __str__(self) -> str:
        return f"{self.name} ({self.size}, {self.precision})"


REGISTRY: dict[tuple[str, str], EikosBuild] = {
    (b.size, b.precision): b for b in [
        EikosBuild("4b", "bf16", f"{HF_OWNER}/Eikos-4B", 9.3),
        EikosBuild("4b", "fp8", f"{HF_OWNER}/Eikos-4B-FP8", 5.8),
        EikosBuild("4b", "int4", f"{HF_OWNER}/Eikos-4B-INT4", 4.0),
        EikosBuild("27b", "bf16", f"{HF_OWNER}/Eikos-27B", 55.6),
        EikosBuild("27b", "fp8", f"{HF_OWNER}/Eikos-27B-FP8", 31.2),
        EikosBuild("27b", "int4", f"{HF_OWNER}/Eikos-27B-INT4", 19.4),
    ]
}


def resolve(size: str, precision: str) -> EikosBuild:
    """Accepts '4b'/'4B'/'4' and 'bf16'/'BF16'; anything not in the registry is rejected."""
    key = (str(size).lower().removesuffix("b") + "b", str(precision).lower())
    if key not in REGISTRY:
        allowed = ", ".join(f"{s}/{p}" for s, p in REGISTRY)
        raise ValueError(f"no Eikos build for size={size!r} precision={precision!r} (known: {allowed})")
    return REGISTRY[key]


def build_for(cfg: ModelConfig) -> EikosBuild:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"model.backend={cfg.backend!r} is not one of {BACKENDS}")
    build = resolve(cfg.size, cfg.precision)
    if cfg.backend not in build.backends:
        raise ValueError(f"{build} needs model.backend=vllm (a compressed-tensors build cannot run in transformers)")
    if cfg.backend == "vllm" and not cfg.vllm_url:
        raise ValueError("model.backend=vllm needs model.vllm_url (start third_party/eikos/scripts/serve_vllm.sh)")
    return build


class Decider:
    """One loaded Eikos model. `choose` returns calibrated probabilities over (key, description) options."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.build = build_for(cfg)
        self._adapter = None
        self._lock = threading.Lock()

    def __str__(self) -> str:
        where = self.cfg.vllm_url if self.cfg.backend == "vllm" else self.cfg.device
        return f"{self.build} via {self.cfg.backend} on {where}"

    def load(self) -> None:
        if self._adapter is not None:
            return
        from huggingface_hub import snapshot_download

        local = snapshot_download(self.build.repo) if self.cfg.backend == "transformers" else snapshot_download(
            self.build.repo, allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*"])
        dcfg = json.loads((Path(local) / "decision_config.json").read_text())
        # decision_core reads PROMPT_STYLE at import time; the released models were trained with this style.
        os.environ["PROMPT_STYLE"] = dcfg["prompt_version"].rsplit("-", 1)[-1]
        import decision_core
        from letter_adapter import LetterAdapter

        decision_core.set_max_one_pass(dcfg.get("max_one_pass"))
        src = f"vllm:{self.cfg.vllm_url.rstrip('/')}|{local}" if self.cfg.backend == "vllm" else local
        calib = str(Path(local) / dcfg["calib"]) if dcfg.get("calib") else None
        adapter = LetterAdapter(src, device=self.cfg.device, max_tokens=self.cfg.max_tokens, calib=calib)
        adapter.load()
        self._adapter = adapter
        print(f"model: {self} (one pass up to {decision_core.MAX_ONE_PASS} options)", flush=True)

    def choose(self, evidence: str, criterion: str, options: list[tuple[str, str]]) -> dict[str, float]:
        self.load()
        question = {"type": "choice", "instructions": criterion}
        with self._lock:  # one forward pass at a time on a local GPU
            probs, _ = self._adapter.dist_any(evidence, question, options)
        return probs
