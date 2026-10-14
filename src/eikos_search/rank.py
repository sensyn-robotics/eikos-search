"""Pick the passage that answers the question: a BM25 prefilter, then one Eikos choice over the survivors."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rank_bm25 import BM25Okapi

from .config import RankConfig
from .extract import Passage

NONE_KEY = "none"
CRITERION = ("Which passage directly and explicitly states the answer to the question in the evidence? "
             f"Choose '{NONE_KEY}' if no passage answers it.")
_CJK = r"぀-ヿ㐀-鿿가-힯"
_TOKEN = re.compile(rf"[{_CJK}]+|[^\W{_CJK}]+")


class Chooser(Protocol):
    def choose(self, evidence: str, criterion: str, options: list[tuple[str, str]]) -> dict[str, float]: ...


@dataclass(frozen=True)
class Ranked:
    passage: Passage | None
    probability: float  # of the shown sentence
    answered: float     # 1 - p(none): that some candidate states the answer
    candidates: int


def tokenize(text: str) -> list[str]:
    """Latin words as words; CJK runs as character bigrams, since they carry no spaces."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if re.match(rf"[{_CJK}]", tok):
            out += [tok[i:i + 2] for i in range(max(1, len(tok) - 1))]
        else:
            out.append(tok)
    return out


def prefilter(question: str, passages: list[Passage], k: int) -> list[Passage]:
    unique = list({p.text: p for p in reversed(passages)}.values())[::-1]  # first occurrence of each sentence
    if len(unique) <= k:
        return unique
    scores = BM25Okapi([tokenize(p.text) for p in unique]).get_scores(tokenize(question))
    order = sorted(range(len(unique)), key=lambda i: -scores[i])[:k]
    return [unique[i] for i in sorted(order)]


def options_for(passages: list[Passage]) -> list[tuple[str, str]]:
    return [(f"p{i}", f"{p.text} (from: {p.article.title})") for i, p in enumerate(passages)] + [
        (NONE_KEY, "none of the passages answers the question")]


def pick(question: str, passages: list[Passage], chooser: Chooser, cfg: RankConfig) -> Ranked:
    cands = prefilter(question, passages, cfg.prefilter_k)
    if not cands:
        return Ranked(None, 0.0, 0.0, 0)
    probs = chooser.choose(f"Question: {question}", CRITERION, options_for(cands))
    answered = 1.0 - probs[NONE_KEY]
    best = max(probs, key=probs.get)
    # Several sentences often state the same fact and split the mass between them, so the confidence that an answer
    # exists is 1 - p(none), not the top sentence's own probability.
    if best == NONE_KEY or answered < cfg.abstain_threshold:
        return Ranked(None, 0.0, answered, len(cands))
    return Ranked(cands[int(best[1:])], probs[best], answered, len(cands))
