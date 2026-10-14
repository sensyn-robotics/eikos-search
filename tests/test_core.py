import textwrap

import pytest
from fastapi.testclient import TestClient

from eikos_search import pipeline as pipeline_mod
from eikos_search.config import Config, ExtractConfig, ModelConfig, RankConfig, load_config
from eikos_search.extract import Article, segment
from eikos_search.models import REGISTRY, build_for, resolve
from eikos_search.pipeline import Pipeline
from eikos_search.rank import NONE_KEY, pick, prefilter, tokenize
from eikos_search.search import SearchHit
from eikos_search.web import create_app

ARTICLE = Article("Water", "https://example.org/water", textwrap.dedent("""\
    Water is a chemical substance.  At sea level, pure water boils at 100 degrees Celsius.
    Ice melts at zero degrees Celsius under standard pressure! Is that surprising?
    水は摂氏100度で沸騰する。氷は摂氏0度で溶ける。"""))


class Stub:
    """Gives `prob` to the option whose description contains `needle`, the rest to 'none'."""

    def __init__(self, needle, prob):
        self.needle, self.prob, self.seen = needle, prob, None

    def choose(self, evidence, criterion, options):
        self.seen = options
        hit = next((k for k, d in options if self.needle and self.needle in d), NONE_KEY)
        probs = dict.fromkeys((k for k, _ in options), 0.0) | {NONE_KEY: 1 - self.prob}
        return probs | {hit: self.prob}


def test_segment_is_verbatim():
    ps = segment(ARTICLE, ExtractConfig(min_chars=5, max_chars=600))
    assert [p.text for p in ps][:3] == ["Water is a chemical substance.", "At sea level, pure water boils at 100 degrees Celsius.",
                                        "Ice melts at zero degrees Celsius under standard pressure!"]
    assert "水は摂氏100度で沸騰する。" in [p.text for p in ps]
    for p in ps:
        assert ARTICLE.text[p.start:p.end] == p.text and p.text in p.paragraph


def test_segment_splits_after_citation_marker():
    art = Article("t", "https://e.org", "It was completed in March 1889.[20] Because the lifts were not ready, they walked.")
    assert [p.text for p in segment(art, ExtractConfig(min_chars=5, max_chars=600))] == [
        "It was completed in March 1889.", "Because the lifts were not ready, they walked."]


def test_segment_length_bounds():
    assert all(20 <= len(p.text) <= 40 for p in segment(ARTICLE, ExtractConfig(min_chars=20, max_chars=40)))


def test_tokenize_cjk_bigrams():
    assert tokenize("Water 沸騰する") == ["water", "沸騰", "騰す", "する"]


def test_prefilter_keeps_relevant_and_dedups():
    ps = segment(ARTICLE, ExtractConfig(min_chars=5, max_chars=600))
    top = prefilter("boiling point of water at sea level", ps + ps, k=1)
    assert [p.text for p in top] == ["At sea level, pure water boils at 100 degrees Celsius."]
    assert len(prefilter("x", ps + ps, k=100)) == len(ps)


@pytest.mark.parametrize("prob,found", [(0.9, True), (0.45, False)])
def test_pick_abstains_below_threshold(prob, found):
    ps = segment(ARTICLE, ExtractConfig(min_chars=5, max_chars=600))
    r = pick("water boils", ps, Stub("boils at 100", prob), RankConfig(prefilter_k=64, abstain_threshold=0.5))
    assert (r.passage is not None) == found
    if found:
        assert r.passage.text == "At sea level, pure water boils at 100 degrees Celsius."


def test_pick_none_option_abstains():
    ps = segment(ARTICLE, ExtractConfig(min_chars=5, max_chars=600))
    stub = Stub(None, 0.95)
    assert pick("who won", ps, stub, RankConfig()).passage is None
    assert stub.seen[-1][0] == NONE_KEY


def test_registry_resolution():
    assert resolve("4B", "BF16").repo == "caiovicentino1/Eikos-4B"
    assert resolve("27", "int4").repo == "caiovicentino1/Eikos-27B-INT4"
    assert len(REGISTRY) == 6
    with pytest.raises(ValueError):
        resolve("8b", "bf16")
    with pytest.raises(ValueError):
        build_for(ModelConfig(precision="int4", backend="transformers"))
    with pytest.raises(ValueError):
        build_for(ModelConfig(backend="vllm"))
    assert build_for(ModelConfig(size="27b", precision="fp8", backend="vllm", vllm_url="http://h:8001")).name == "Eikos-27B-FP8"


def test_config_rejects_unknown_keys(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("model: {size: 27b, sise: 4b}\n")
    with pytest.raises(ValueError, match="sise"):
        load_config(f)
    f.write_text("model: {size: 27b}\n")
    assert load_config(f).model.size == "27b" and load_config(f).rank.prefilter_k == 64


def test_repo_config_loads():
    cfg = load_config("config/config.yaml")
    assert build_for(cfg.model).name == "Eikos-4B"


def test_pipeline_and_web(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "search", lambda q, cfg: [SearchHit("Water", ARTICLE.url, "")])
    monkeypatch.setattr(pipeline_mod, "fetch_articles", lambda hits, t: [ARTICLE])
    client = TestClient(create_app(Pipeline(Config(), chooser=Stub("boils at 100", 0.8))))
    a = client.post("/api/answer", json={"question": "At what temperature does water boil?"}).json()
    assert a["found"] and a["url"] == ARTICLE.url
    assert a["paragraph"][a["offset_in_paragraph"]:][: len(a["text"])] == a["text"]
    assert client.post("/api/answer", json={"question": "  "}).status_code == 422
    assert "eikos-search" in client.get("/").text


def test_pick_answers_when_mass_is_split_between_correct_sentences():
    ps = segment(ARTICLE, ExtractConfig(min_chars=5, max_chars=600))

    class Split:
        def choose(self, evidence, criterion, options):
            probs = dict.fromkeys((k for k, _ in options), 0.0) | {NONE_KEY: 0.01}
            hits = [k for k, d in options if "100" in d]
            return probs | {k: 0.99 / len(hits) for k in hits}

    r = pick("water boils", ps, Split(), RankConfig(abstain_threshold=0.5))
    assert r.passage is not None and "100" in r.passage.text and r.answered == pytest.approx(0.99)


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost:8000/api", "http://10.0.0.5/", "http://169.254.169.254/latest",
                                 "http://[::1]/", "file:///etc/passwd", "http://0.0.0.0/"])
def test_fetch_refuses_non_public_targets(url):
    import httpx

    from eikos_search.extract import UnsafeURL, fetch_article, public_address
    with pytest.raises(UnsafeURL):
        public_address(httpx.URL(url))
    assert fetch_article(SearchHit("x", url, ""), 2) is None
