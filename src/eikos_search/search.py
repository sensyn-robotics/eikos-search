"""Registry of web search backends. `search.backend` in the config names one; each returns SearchHit records."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import SearchConfig


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


def _ddgs(query: str, cfg: SearchConfig) -> list[SearchHit]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        rows = DDGS().text(query, region=cfg.region, backend=cfg.engines, max_results=cfg.max_results)
    except DDGSException as e:  # raised for "no results" as well as for engine failures
        print(f"search returned nothing: {e}", flush=True)
        return []
    return [SearchHit(r.get("title", ""), r["href"], r.get("body", "")) for r in rows if r.get("href")]


BACKENDS: dict[str, Callable[[str, SearchConfig], list[SearchHit]]] = {"ddgs": _ddgs}


def search(query: str, cfg: SearchConfig) -> list[SearchHit]:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"search.backend={cfg.backend!r} is not one of {sorted(BACKENDS)}")
    hits = [h for h in BACKENDS[cfg.backend](query, cfg) if h.url.startswith(("http://", "https://"))]
    return list({h.url: h for h in hits}.values())[: cfg.max_results]
