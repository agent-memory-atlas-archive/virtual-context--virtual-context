"""Continuation calls of one turn reuse the previous call's tagging and scoring."""
from __future__ import annotations

from virtual_context.core.retriever import ContextRetriever
from virtual_context.types import RetrieverConfig, StrategyConfig, TagResult, TagStats

from tests.conftest import MockTagGenerator


class _MemoProvider:
    def __init__(self):
        self.saved: dict[tuple[str, str], dict] = {}
        self.loads = 0

    def load_retrieval_memo(self, conversation_id, memo_key):
        self.loads += 1
        return self.saved.get((conversation_id, memo_key))

    def save_retrieval_memo(self, conversation_id, memo_key, payload, **_):
        self.saved[(conversation_id, memo_key)] = payload

    def load_tag_stats_snapshot(self, conversation_id):
        return None

    def save_tag_stats_snapshot(self, *a, **k):
        return None


class _Store:
    """Only what retrieve() touches on the no-summary path."""

    def get_all_tags(self, **kwargs):
        return [TagStats(tag="crochet", usage_count=3), TagStats(tag="cooking", usage_count=1)]

    def get_summaries_by_tags(self, *a, **k):
        return []

    def search(self, *a, **k):
        return []

    def get_actionable_fact_tags(self, *a, **k):
        return set()

    def get_facts_by_tags(self, *a, **k):
        return []

    def get_all_facts(self, *a, **k):
        return []

    def __getattr__(self, name):
        raise AttributeError(name)


class _Tagger(MockTagGenerator):
    def generate_tags(self, text, existing_tags=None, **kwargs):
        self.calls.append(text)
        return TagResult(tags=["crochet"], primary="crochet", source="llm", related_tags=[], query_embedding=None)


def _retriever(provider, tagger):
    config = RetrieverConfig(
        tag_context_max_tokens=30000,
        strategy_configs={"default": StrategyConfig(min_overlap=1, max_results=10)},
    )
    return ContextRetriever(
        tagger, _Store(), config, conversation_id="conv-memo",
        inbound_tagger=tagger, session_state_provider=provider,
    )


def test_same_inputs_reuse_tagging_and_scoring():
    provider, tagger = _MemoProvider(), _Tagger()
    r = _retriever(provider, tagger)
    first = r.retrieve("tell me about crochet", current_active_tags=[], context_turns=["earlier"])
    assert first.retrieval_metadata.get("memo") == "miss"
    assert len(tagger.calls) == 1 and len(provider.saved) == 1
    second = r.retrieve("tell me about crochet", current_active_tags=[], context_turns=["earlier"])
    assert second.retrieval_metadata.get("memo") == "hit"
    assert len(tagger.calls) == 1                      # no second tagging
    assert second.retrieval_scores == first.retrieval_scores
    assert second.tags_matched == first.tags_matched


def test_changed_inputs_miss_the_memo():
    provider, tagger = _MemoProvider(), _Tagger()
    r = _retriever(provider, tagger)
    r.retrieve("tell me about crochet", current_active_tags=[], context_turns=["earlier"])
    r.retrieve("tell me about crochet", current_active_tags=["cooking"], context_turns=["earlier"])
    r.retrieve("tell me about crochet", current_active_tags=[], context_turns=["later"])
    r.retrieve("something else entirely", current_active_tags=[], context_turns=["earlier"])
    assert len(tagger.calls) == 4 and len(provider.saved) == 4


def test_no_provider_means_no_memo():
    tagger = _Tagger()
    r = ContextRetriever(tagger, _Store(), RetrieverConfig(), conversation_id="conv", inbound_tagger=tagger)
    out = r.retrieve("tell me about crochet", current_active_tags=[])
    assert "memo" not in out.retrieval_metadata
