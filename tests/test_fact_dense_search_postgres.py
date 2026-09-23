"""PostgreSQL twin of test_fact_dense_search.py.

Needs a database where the pgvector extension is installed; the semantic
vector migration then adds the fact vector cache.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.types import Fact

MODEL = "all-MiniLM-L6-v2"


def v(*head):
    return [*head] + [0.0] * (384 - len(head))


def _vector_installed() -> bool:
    if not pg_dsn():
        return False
    try:
        return pg_test_conn().execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is not None
    except Exception:
        return False


_pg_required = pytest.mark.skipif(not _vector_installed(), reason="DATABASE_URL not set or pgvector not installed")


@pytest.fixture(scope="module")
def store():
    from virtual_context.storage.postgres import PostgresStore
    store = PostgresStore(pg_dsn())
    assert store.migrate_semantic_vectors(dry_run=False)["ready"]
    return store


def _fact(fid, conv, superseded_by=None):
    return Fact(
        id=fid, subject="user", verb="logged", object=fid, conversation_id=conv,
        mentioned_at=datetime(2026, 1, 1, tzinfo=timezone.utc), superseded_by=superseded_by,
    )


@_pg_required
@pytest.mark.regression("BUG-082")
def test_search_ranks_in_the_database_pg(store):
    conv = f"pg-dense-{uuid.uuid4().hex[:8]}"
    ids = {name: f"{conv}-{name}" for name in ("a", "tie", "b", "c", "old")}
    store.store_facts([
        _fact(ids["a"], conv), _fact(ids["tie"], conv), _fact(ids["b"], conv), _fact(ids["c"], conv),
        _fact(ids["old"], conv, superseded_by=ids["a"]),
    ])
    store.store_fact_embeddings(ids["a"], conv, MODEL, v(1.0, 0.0, 0.0))
    store.store_fact_embeddings(ids["tie"], conv, MODEL, v(2.0, 0.0, 0.0))
    store.store_fact_embeddings(ids["b"], conv, MODEL, v(0.6, 0.8, 0.0))
    store.store_fact_embeddings(ids["c"], conv, "another-model", [1.0, 0.0, 0.0])
    store.store_fact_embeddings(ids["old"], conv, MODEL, v(1.0, 0.0, 0.0))

    hits = store.search_fact_embeddings(conv, MODEL, [v(1.0, 0.0, 0.0)], limit=10)
    assert [(f.id, round(s, 4)) for f, s in hits] == [(ids["a"], 1.0), (ids["tie"], 1.0), (ids["b"], 0.6)]
    both = store.search_fact_embeddings(conv, MODEL, [v(0.0, 1.0, 0.0), v(1.0, 0.0, 0.0)], limit=10)
    assert {f.id: round(s, 4) for f, s in both} == {ids["a"]: 1.0, ids["tie"]: 1.0, ids["b"]: 0.8}
    assert [f.id for f, _ in store.search_fact_embeddings(conv, MODEL, [v(1.0, 0.0, 0.0)], limit=1)] == [ids["a"]]


@_pg_required
@pytest.mark.regression("BUG-082")
def test_superseded_facts_filling_the_candidate_pool_do_not_hide_live_ones_pg(store):
    conv = f"pg-dense-{uuid.uuid4().hex[:8]}"
    live = f"{conv}-live"
    stale = [f"{conv}-s{i}" for i in range(3)]
    store.store_facts([_fact(live, conv)] + [_fact(fid, conv, superseded_by=live) for fid in stale])
    for fid in stale:
        store.store_fact_embeddings(fid, conv, MODEL, v(1.0, 0.0, 0.0))
    store.store_fact_embeddings(live, conv, MODEL, v(0.0, 1.0, 0.0))
    hits = store.search_fact_embeddings(conv, MODEL, [v(1.0, 0.0, 0.0)], limit=1)
    assert [(f.id, round(s, 4)) for f, s in hits] == [(live, 0.0)]


@_pg_required
@pytest.mark.regression("BUG-082")
def test_writes_after_migration_are_searchable_and_other_models_do_not_block_pg(store):
    conv = f"pg-dense-{uuid.uuid4().hex[:8]}"
    fid, other = f"{conv}-new", f"{conv}-other"
    store.store_facts([_fact(fid, conv), _fact(other, conv)])
    store.store_fact_embeddings(fid, conv, MODEL, [1.0] + [0.0] * 383)
    store.store_fact_embeddings(other, conv, "another-model", [1.0, 0.0])
    assert store.vector_search_ready(MODEL)
    hits = store.search_fact_embeddings(conv, MODEL, [[1.0] + [0.0] * 383], limit=5)
    assert [(f.id, round(s, 4)) for f, s in hits] == [(fid, 1.0)]
    assert store.search_fact_embeddings(conv, "another-model", [[1.0, 0.0]], limit=5) is None
