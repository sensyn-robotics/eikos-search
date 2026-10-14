"""Local web UI: a question box, and the answer shown highlighted inside its source paragraph."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .pipeline import Pipeline

INDEX = (Path(__file__).parent / "templates" / "index.html").read_text()


class Query(BaseModel):
    question: str


def create_app(pipe: Pipeline) -> FastAPI:
    app = FastAPI(title="eikos-search")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX

    @app.post("/api/answer")
    async def answer(q: Query) -> dict:
        if not q.question.strip():
            raise HTTPException(422, "empty question")
        return (await run_in_threadpool(pipe.answer, q.question)).to_dict()

    return app
