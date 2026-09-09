"""BUG-074: strict history must leave durable, role-local retrieval chunks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_audience_reassignment import ACTOR, ORIGINAL, OTHER, OWNER, TENANT, _ingest
from tests.test_audience_reassignment_postgres import _disposable_dsn, stack as source_stack
from virtual_context.config import VirtualContextConfig
from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.quote_search import find_quote
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.store import canonical_rows_to_history
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.state import ProxyState
from virtual_context.types import (
    SpeakerRetrievalContext,
    StorageConfig,
    TagGeneratorConfig,
    TagResult,
)

pytestmark = [
    pytest.mark.regression("BUG-074"),
    pytest.mark.skipif(not _disposable_dsn(), reason="requires vc_disposable_* database"),
]


@pytest.fixture
def indexed_history():
    generator = source_stack.__wrapped__()
    inner, reconciler = next(generator)
    try:
        with inner.pool.connection() as conn:
            conn.execute("TRUNCATE canonical_turn_chunks")
        yield inner, reconciler
    finally:
        generator.close()


def _runtime(inner):
    store = CompositeStore(segments=inner, facts=inner, fact_links=inner, state=inner, search=inner)
    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(backend="postgres", postgres_dsn=_disposable_dsn()),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
    pipeline = TaggingPipeline(
        tag_generator=SimpleNamespace(
            generate_tags=lambda *args, **kwargs: TagResult(
                tags=["synthetic"], primary="synthetic", source="keyword"
            )
        ),
        turn_tag_index=TurnTagIndex(),
        store=store,
        semantic=semantic,
        engine_state=SimpleNamespace(lifecycle_epoch=1),
        config=config,
        tag_splitter=None,
        canonicalizer=None,
        telemetry=None,
        monitor=None,
        compactor=None,
        save_state_callback=lambda *args, **kwargs: None,
    )
    return store, semantic, pipeline


def _chunks(inner):
    with inner.pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM canonical_turn_chunks ORDER BY canonical_turn_id,side,chunk_index"
        ).fetchall()


def _source_rows(inner):
    with inner.pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM canonical_message_sources ORDER BY message_id"
        ).fetchall()


def _context(audience=ORIGINAL):
    return SpeakerRetrievalContext(
        tenant_id=TENANT,
        owner_conversation_id=OWNER,
        audience_conversation_id=audience,
        audience_channel_scope="conversation",
        request_origin_channel_id="1000000000000000999",
    )


def test_pg_strict_history_indexes_six_rows_before_durable_worker_skip(indexed_history):
    inner, reconciler = indexed_history
    for number in (1, 2):
        _ingest(
            reconciler,
            message_id=f"400000000000000008{number}",
            body=f"A distinct synthetic observation number {number}.",
        )
    store, semantic, pipeline = _runtime(inner)
    original = store.get_all_canonical_turns(OWNER)
    assert len(original) == 6 and not _chunks(inner)
    source_rows = _source_rows(inner)
    assert not find_quote(
        store,
        semantic,
        "copper lantern",
        conversation_id=OWNER,
        max_results=20,
        speaker_context=_context(),
    )["results"]
    history = canonical_rows_to_history(original, include_tagging_identity=True)
    assert (
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
        == 3
    )
    pending = store.iter_complete_untagged_canonical_groups(
        conversation_id=OWNER, expected_lifecycle_epoch=1, batch_size=32
    )
    state = SimpleNamespace(engine=SimpleNamespace(_semantic=semantic, _tagging=pipeline))
    for group in pending:
        assert ProxyState._run_tagging_pipeline(state, group, expected_lifecycle_epoch=1)
    assert pending == []
    physical = {row.canonical_turn_id: row for row in store.get_all_canonical_turns(OWNER)}
    assert all(row.tagged_at for row in physical.values())
    chunks = _chunks(inner)
    assert len(chunks) == 6
    for chunk in chunks:
        row = physical[str(chunk["canonical_turn_id"])]
        assert chunk["side"] in {"user", "assistant"}
        assert chunk["text"] == getattr(row, chunk["side"] + "_content")
        if chunk["side"] == "assistant":
            assert not row.sender_actor_id
        else:
            assert row.sender_actor_id == f"actor:discord:{ACTOR}"
    assert _source_rows(inner) == source_rows
    result = find_quote(
        store,
        semantic,
        "copper lantern",
        conversation_id=OWNER,
        max_results=20,
        speaker_context=_context(),
    )
    assistant_ids = {row.canonical_turn_id for row in physical.values() if row.assistant_content}
    found = [item for item in result["results"] if item.get("source_role") == "assistant"]
    assert {item["segment_ref"].removeprefix("canonical_turn_") for item in found} == assistant_ids
    assert all(item["excerpt"] == "Assistant: A synthetic source response." for item in found)
    assert not find_quote(
        store,
        semantic,
        "copper lantern",
        conversation_id=OWNER,
        max_results=20,
        speaker_context=_context(OTHER),
    )["results"]


@pytest.mark.parametrize("failure", ["provider", "storage"])
def test_pg_strict_history_failed_side_stays_pending_for_durable_retry(
    indexed_history, monkeypatch, failure
):
    inner, _reconciler = indexed_history
    store, semantic, pipeline = _runtime(inner)
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    before_sources = _source_rows(inner)
    progress = []
    checkpoints = []
    pipeline._save_state_callback = lambda *args, **kwargs: checkpoints.append(kwargs)
    original = (
        semantic._embed_fn if failure == "provider" else store.store_canonical_turn_chunk_embeddings
    )
    calls = 0

    def fail_second_side(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second-side outage")
        return original(*args, **kwargs)

    if failure == "provider":
        monkeypatch.setattr(semantic, "_embed_fn", fail_second_side)
    else:
        monkeypatch.setattr(store, "store_canonical_turn_chunk_embeddings", fail_second_side)
    with pytest.raises(RuntimeError, match="embedding"):
        pipeline.ingest_history(
            history,
            require_existing_canonical=True,
            expected_lifecycle_epoch=1,
            progress_callback=lambda *args: progress.append(args),
        )
    assert len(_chunks(inner)) == 1
    assert all(not row.tagged_at for row in store.get_all_canonical_turns(OWNER))
    assert pipeline._turn_tag_index.entries == checkpoints == progress == []
    pending = store.iter_complete_untagged_canonical_groups(
        conversation_id=OWNER, expected_lifecycle_epoch=1, batch_size=32
    )
    assert len(pending) == 1
    monkeypatch.undo()
    state = SimpleNamespace(engine=SimpleNamespace(_semantic=semantic, _tagging=pipeline))
    assert ProxyState._run_tagging_pipeline(state, pending[0], expected_lifecycle_epoch=1)
    assert len(_chunks(inner)) == 2
    assert all(row.tagged_at for row in store.get_all_canonical_turns(OWNER))
    assert not store.iter_complete_untagged_canonical_groups(
        conversation_id=OWNER, expected_lifecycle_epoch=1, batch_size=32
    )
    assert _source_rows(inner) == before_sources
