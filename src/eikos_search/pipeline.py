"""question -> search -> fetch -> sentences -> Eikos choice -> the verbatim sentence and where it came from."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .config import Config
from .extract import fetch_articles, segment
from .models import Decider
from .rank import Chooser, pick
from .search import search


@dataclass(frozen=True)
class Answer:
    question: str
    found: bool
    probability: float
    answered: float
    text: str | None = None
    paragraph: str | None = None
    offset_in_paragraph: int | None = None
    title: str | None = None
    url: str | None = None
    articles: int = 0
    candidates: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(self, cfg: Config, chooser: Chooser | None = None):
        self.cfg = cfg
        self.chooser = chooser or Decider(cfg.model)
        print(f"search backend: {cfg.search.backend}", flush=True)

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("empty question")
        articles = fetch_articles(search(question, self.cfg.search), self.cfg.search.timeout_s)
        passages = [p for a in articles for p in segment(a, self.cfg.extract)]
        r = pick(question, passages, self.chooser, self.cfg.rank)
        common = {"question": question, "probability": r.probability, "answered": r.answered,
                  "articles": len(articles), "candidates": r.candidates, "seconds": time.perf_counter() - t0}
        if r.passage is None:
            return Answer(found=False, **common)
        p = r.passage
        assert p.article.text[p.start:p.end] == p.text  # what is shown is a slice of the article, never generated
        return Answer(found=True, text=p.text, paragraph=p.paragraph, offset_in_paragraph=p.start - p.para_start,
                      title=p.article.title, url=p.article.url, **common)
