"""eikos-search <config.yaml> [--repl]: serve the web UI, or answer questions typed in the terminal."""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import Pipeline


def repl(pipe: Pipeline) -> None:
    while True:
        try:
            q = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not q:
            continue
        a = pipe.answer(q)
        if a.found:
            print(f'\n  "{a.text}"\n\n  {a.title}\n  {a.url}\n  p(sentence)={a.probability:.2f} p(answered)={a.answered:.2f}  ({a.candidates} sentences from {a.articles} articles, {a.seconds:.1f}s)')
        else:
            print(f"\n  no article states this exactly (p(answered)={a.answered:.2f}, {a.articles} articles, {a.seconds:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="eikos-search", description=__doc__)
    ap.add_argument("config", help="path to config/config.yaml")
    ap.add_argument("--repl", action="store_true", help="ask in the terminal instead of serving the web UI")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pipe = Pipeline(cfg)
    pipe.chooser.load()
    if args.repl:
        repl(pipe)
        return
    import uvicorn

    from .web import create_app

    uvicorn.run(create_app(pipe), host=cfg.server.host, port=cfg.server.port)
