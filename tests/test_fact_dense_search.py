"""Dense fact retrieval ranks inside the store and returns only the winners.

The first dense path loaded every live fact row and every JSON vector of the
conversation on each request, then ranked them in Python. A store that
implements ``search_fact_embeddings`` does the ranking itself and hands back
just the top facts with their scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from virtual_context.core.retriever import ContextRetriever
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, RetrieverConfig, ScoringConfig

CONV = "conv-dense"
MODEL = "all-MiniLM-L6-v2"


def _fact(fid, conv=CONV, superseded_by=None, tags=None):
    fact = Fact(
        id=fid, subject="user", verb="logged", object=fid, conversation_id=conv,
        mentioned_at=datetime(2026, 1, 1, tzinfo=timezone.utc), superseded_by=superseded_by,
    )
    if tags:
        fact.tags = list(tags)
    return fact


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(db_path=str(tmp_path / "dense.db"))
    s.store_facts([
        _fact("a"), _fact("b"), _fact("c"), _fact("tie"),
        _fact("old", superseded_by="a"), _fact("elsewhere", conv="other"),
        _fact("short"),
    ])
    s.store_fact_embeddings("a", CONV, MODEL, [1.0, 0.0, 0.0])
    s.store_fact_embeddings("tie", CONV, MODEL, [2.0, 0.0, 0.0])
    s.store_fact_embeddings("b", CONV, MODEL, [0.6, 0.8, 0.0])
    s.store_fact_embeddings("c", CONV, "another-model", [1.0, 0.0, 0.0])
    s.store_fact_embeddings("old", CONV, MODEL, [1.0, 0.0, 0.0])
    s.store_fact_embeddings("elsewhere", "other", MODEL, [1.0, 0.0, 0.0])
    s.store_fact_embeddings("short", CONV, MODEL, [1.0, 0.0])
    return s


@pytest.mark.regression("BUG-082")
def test_search_returns_live_current_model_facts_best_first(store):
    hits = store.search_fact_embeddings(CONV, MODEL, [[1.0, 0.0, 0.0]], limit=10)
    assert [(f.id, round(s, 4)) for f, s in hits] == [("a", 1.0), ("tie", 1.0), ("b", 0.6)]
    assert all(isinstance(f, Fact) for f, _ in hits)


@pytest.mark.regression("BUG-082")
def test_search_limit_and_best_of_several_queries(store):
    assert [f.id for f, _ in store.search_fact_embeddings(CONV, MODEL, [[1.0, 0.0, 0.0]], limit=1)] == ["a"]
    hits = store.search_fact_embeddings(CONV, MODEL, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], limit=10)
    scores = {f.id: round(s, 4) for f, s in hits}
    assert scores == {"a": 1.0, "tie": 1.0, "b": 0.8}


class _SearchOnly:
    """A store whose full vector load must not be touched."""

    def __init__(self, inner):
        self._inner = inner
        self.searches = 0

    def query_facts(self, **kwargs):
        return self._inner.query_facts(**kwargs)

    def search_fact_embeddings(self, *args, **kwargs):
        self.searches += 1
        return self._inner.search_fact_embeddings(*args, **kwargs)

    def load_fact_embeddings(self, *args, **kwargs):
        raise AssertionError("dense retrieval loaded every fact vector")


def _retriever(store, vec, *, ctx_turns=0):
    return ContextRetriever(
        tag_generator=MagicMock(),
        store=store,
        config=RetrieverConfig(
            fact_dense_retrieval=True, fact_dense_top_n=2, prefetch_facts=False,
            scoring=ScoringConfig(embedding_context_turns=ctx_turns),
        ),
        conversation_id=CONV,
        query_embed_fn=lambda texts: [list(vec) for _ in texts],
    )


@pytest.mark.regression("BUG-082")
def test_retriever_ranks_through_the_store_search(store):
    wrapped = _SearchOnly(store)
    meta: dict = {}
    union = _retriever(wrapped, [1.0, 0.0, 0.0])._fetch_facts_dense("q", None, [], meta)
    assert wrapped.searches == 1
    assert meta["fact_dense_rank_by_id"] == {"a": 0, "tie": 1}
    assert meta["fact_dense_score_by_id"]["a"] == pytest.approx(1.0)
    assert [f.id for f in union[:2]] == ["a", "tie"]
    assert {f.id for f in union} >= {"a", "b", "tie"}
    assert meta["fact_source_by_id"]["a"] == "both"


@pytest.mark.regression("BUG-082")
def test_retriever_passes_the_recent_turn_query_too(store):
    wrapped = _SearchOnly(store)
    seen = []
    original = wrapped.search_fact_embeddings

    def spy(conv, model, queries, **kwargs):
        seen.append(len(queries))
        return original(conv, model, queries, **kwargs)

    wrapped.search_fact_embeddings = spy
    _retriever(wrapped, [1.0, 0.0, 0.0], ctx_turns=2)._fetch_facts_dense("q", ["earlier turn"], [], {})
    assert seen == [2]
