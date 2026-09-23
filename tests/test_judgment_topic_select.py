"""Jev chooses which topics to retrieve from the fused ranking's wider pool."""

import json

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime, select_topics
from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import JudgmentConfig, TagSummary


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _runtime(mode, probs, *, fail=False, **cfg):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if fail:
            return httpx.Response(500)
        answers = {k: {"type": "noul", "noul": probs.get(k, 0.0)} for k in body["questions"]}
        return httpx.Response(200, json={"model": "jev-t", "answers": answers,
                                         "usage": {"input_tokens": 5, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    config = JudgmentConfig(mode="legacy", seams={"topic_select": mode}, **cfg)
    return build_runtime(config, environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


RANKED = ["vitamin-d", "selank", "vast-ai", "hcg-dosing", "weekly", "camera"]
TEXTS = {tag: f"{tag} summary" for tag in RANKED}


def _load(tags):
    return {tag: TEXTS[tag] for tag in tags if tag in TEXTS}


def test_legacy_takes_the_top_of_the_fused_ranking_without_calling_jev():
    rt, seen = _runtime("legacy", {})
    assert select_topics("q", RANKED, _load, max_results=3, runtime=rt) == (RANKED[:3], False)
    assert seen == []


def test_jev_picks_relevant_topics_from_beyond_the_fused_top():
    rt, seen = _runtime("jev", {"hcg-dosing": 0.94, "weekly": 0.94, "selank": 0.6, "vitamin-d": 0.1},
                        topic_pool_size=6, topic_min_probability=0.5)
    tags, by_jev = select_topics("how long does the kit last", RANKED, _load, max_results=3, runtime=rt)
    # Equal probability keeps the fused order; irrelevant topics are left out.
    assert (tags, by_jev) == (["hcg-dosing", "weekly", "selank"], True)
    assert seen[0]["state"]["candidates"]["hcg-dosing"] == "hcg-dosing summary"
    assert len(seen[0]["state"]["candidates"]) == 6


def test_pool_is_bounded_by_the_configured_size():
    rt, seen = _runtime("jev", {"selank": 0.9}, topic_pool_size=3)
    select_topics("q", RANKED, _load, max_results=2, runtime=rt)
    assert set(seen[0]["state"]["candidates"]) == {"vitamin-d", "selank", "vast-ai"}


def test_nothing_relevant_keeps_the_legacy_choice():
    rt, _ = _runtime("jev", {tag: 0.1 for tag in RANKED}, topic_min_probability=0.5)
    assert select_topics("q", RANKED, _load, max_results=2, runtime=rt) == (RANKED[:2], False)


def test_a_failed_jev_call_keeps_the_legacy_choice():
    rt, _ = _runtime("jev", {}, fail=True)
    assert select_topics("q", RANKED, _load, max_results=2, runtime=rt) == (RANKED[:2], False)


def test_shadow_uses_the_legacy_choice():
    rt, seen = _runtime("shadow", {"hcg-dosing": 0.99})
    assert select_topics("q", RANKED, _load, max_results=2, runtime=rt) == (RANKED[:2], False)
    assert len(seen) == 1


def test_chosen_topic_summaries_lead_the_retrieved_items():
    class _Store:
        def get_tag_summary(self, tag, conversation_id=""):
            return TagSummary(tag=tag, summary=f"{tag}: kit lasts about 23 months",
                              summary_tokens=10, source_canonical_turn_ids=[f"ct-{tag}"])

    retriever = ContextRetriever.__new__(ContextRetriever)
    retriever.store = _Store()
    retriever._conversation_id = "conv"
    rt, _ = _runtime("jev", {"hcg-dosing": 0.9}, topic_pool_size=6)
    retriever.judgment_runtime = rt
    tags, by_jev, loaded = retriever._select_topics("q", RANKED, 3, None, "")
    assert (tags, by_jev) == (["hcg-dosing"], True)
    item = retriever._tag_summary_item(loaded["hcg-dosing"])
    assert item.ref == "tag-summary-hcg-dosing"
    assert "23 months" in item.summary
    assert item.metadata.canonical_turn_ids == ["ct-hcg-dosing"]


def _pool_retriever(mode, source, stored):
    class _Store:
        def load_tag_summary_embeddings(self, conversation_id=None):
            return stored

    retriever = ContextRetriever.__new__(ContextRetriever)
    retriever.store = _Store()
    retriever._conversation_id = "conv"
    retriever._session_state_provider = None
    rt, _ = _runtime(mode, {}, topic_pool_size=2, topic_pool_source=source)
    retriever.judgment_runtime = rt
    return retriever


def test_embedding_pool_is_the_most_similar_topics():
    stored = {"far": [0.0, 1.0], "near": [1.0, 0.1], "mid": [1.0, 1.0]}
    retriever = _pool_retriever("jev", "embedding", stored)
    assert retriever._embedding_pool_enabled() is True
    pool = retriever._embedding_pool([1.0, 0.0])
    assert list(pool) == ["near", "mid"]
    assert pool["near"] > pool["mid"]


@pytest.mark.parametrize(("mode", "source"), [("legacy", "embedding"), ("jev", "fused"), ("shadow", "embedding")])
def test_embedding_pool_needs_live_topic_selection_from_embeddings(mode, source):
    """Legacy and shadow keep the fused pool; so does a live choice from the fused source."""
    retriever = _pool_retriever(mode, source, {"a": [1.0, 0.0]})
    assert retriever._embedding_pool_enabled() is False
