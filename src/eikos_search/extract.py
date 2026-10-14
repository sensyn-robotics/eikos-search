"""Fetch an article, extract its main text, and cut it into sentences that keep their exact character offsets."""
from __future__ import annotations

import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import trafilatura

from .config import ExtractConfig
from .search import SearchHit

MAX_REDIRECTS = 5
USER_AGENT = "eikos-search/0.1 (+https://github.com/sensyn-robotics/eikos-search)"
# A sentence ends at ./!/? followed by whitespace, or right after a CJK full stop (CJK text has no spaces). Wikipedia
# citation markers like "[20]" sit between the stop and the space; they fall in the gap between the two sentences.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\[\d+\])*\s+(?=\S)|(?<=[。！？])(?:\[\d+\])*")


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class Passage:
    article: Article
    start: int
    end: int
    para_start: int
    para_end: int

    @property
    def text(self) -> str:
        return self.article.text[self.start:self.end]

    @property
    def paragraph(self) -> str:
        return self.article.text[self.para_start:self.para_end]


class UnsafeURL(ValueError):
    pass


def public_address(url: httpx.URL) -> str:
    """The address to connect to for `url`, only if every address its host resolves to is public. Search results and
    their redirects are third-party input; without this a result could make us fetch localhost or the LAN."""
    if url.scheme not in ("http", "https") or not url.host:
        raise UnsafeURL(f"not an http(s) URL: {url}")
    host = url.host.rstrip(".").lower()
    try:
        infos = socket.getaddrinfo(host, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeURL(f"cannot resolve {host}: {e}") from e
    addrs = {ipaddress.ip_address(info[4][0].split("%")[0]) for info in infos}
    bad = [a for a in addrs if not a.is_global or a.is_multicast]
    if not addrs or bad:
        raise UnsafeURL(f"{host} resolves to a non-public address {bad[0] if bad else ''}")
    return str(min(addrs, key=lambda a: a.version))


def _get_pinned(client: httpx.Client, url: httpx.URL) -> httpx.Response:
    """GET `url` on the address just validated, so a second DNS answer (rebinding) cannot redirect the connection;
    Host header and TLS SNI keep the real hostname, so certificate checks are unchanged."""
    ip = public_address(url)
    target = url.copy_with(host=ip)
    return client.get(target, headers={"Host": url.netloc.decode("ascii")}, extensions={"sni_hostname": url.host})


def fetch_article(hit: SearchHit, timeout_s: float) -> Article | None:
    url = httpx.URL(hit.url)
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
            for _ in range(MAX_REDIRECTS + 1):
                r = _get_pinned(client, url)
                if not r.is_redirect:
                    break
                url = url.join(r.headers["location"])
            else:
                raise UnsafeURL(f"more than {MAX_REDIRECTS} redirects")
            r.raise_for_status()
    except (httpx.HTTPError, UnsafeURL, KeyError) as e:
        print(f"fetch skipped {hit.url}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None
    text = trafilatura.extract(r.text, url=str(url), include_comments=False, include_tables=False,
                               favor_precision=True)
    return Article(hit.title, hit.url, text) if text else None


def fetch_articles(hits: list[SearchHit], timeout_s: float) -> list[Article]:
    with ThreadPoolExecutor(max_workers=max(1, len(hits))) as pool:
        return [a for a in pool.map(lambda h: fetch_article(h, timeout_s), hits) if a]


def _spans(text: str, start: int, end: int, splitter: re.Pattern) -> list[tuple[int, int]]:
    """(start, end) of each non-empty piece of text[start:end], trimmed of surrounding whitespace."""
    out, cur = [], start
    for m in [*splitter.finditer(text, start, end), None]:
        s, e = cur, (m.start() if m else end)
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
        if m:
            cur = m.end()
    return out


def segment(article: Article, cfg: ExtractConfig) -> list[Passage]:
    """Sentences within paragraphs. Each passage is a slice of article.text, so what is shown is never rewritten."""
    return [Passage(article, s, e, ps, pe)
            for ps, pe in _spans(article.text, 0, len(article.text), re.compile(r"\n+"))
            for s, e in _spans(article.text, ps, pe, _SENTENCE_END)
            if cfg.min_chars <= e - s <= cfg.max_chars]
