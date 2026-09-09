"""BUG-074: strict history tagging cannot strand canonical retrieval chunks.

Identifiers and message content are synthetic; immutable source-attested pairs
exercise the same admission and strict-history paths as native ingestion.
"""

from types import SimpleNamespace

import pytest

from tests.test_audience_reassignment import OWNER, stack as source_stack
from virtual_context.config import VirtualContextConfig
from virtual_context.core.composite_store import CompositeStore
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.core.store import canonical_rows_to_history
from virtual_context.core.tagging_pipeline import TaggingPipeline
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.state import ProxyState
from virtual_context.types import StorageConfig, TagGeneratorConfig, TagResult

pytestmark = pytest.mark.regression("BUG-074")


@pytest.fixture
def indexed_history(tmp_path):
    yield from source_stack.__wrapped__(tmp_path)


def runtime(inner):
    store = CompositeStore(segments=inner, facts=inner, fact_links=inner, state=inner, search=inner)
    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(backend="sqlite"),
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


def chunks(inner):
    return [
        dict(row)
        for row in inner._get_conn().execute(
            "SELECT * FROM canonical_turn_chunks ORDER BY canonical_turn_id,side,chunk_index"
        )
    ]


def test_strict_history_indexes_before_durable_worker_skips_tagged_pair(indexed_history):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    original = store.get_all_canonical_turns(OWNER)
    history = canonical_rows_to_history(original, include_tagging_identity=True)
    assert len(original) == 2 and not chunks(inner)
    before_sources = [
        dict(row) for row in inner._get_conn().execute("SELECT * FROM canonical_message_sources")
    ]
    assert (
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
        == 1
    )
    # This is the durable worker's exact queue. Previously strict-history
    # ingestion marked both rows tagged and made the embedding worker skip them.
    pending = store.iter_complete_untagged_canonical_groups(
        conversation_id=OWNER, expected_lifecycle_epoch=1, batch_size=32
    )
    state = SimpleNamespace(engine=SimpleNamespace(_semantic=semantic, _tagging=pipeline))
    for group in pending:
        assert ProxyState._run_tagging_pipeline(state, group, expected_lifecycle_epoch=1)
    assert pending == []
    physical = {row.canonical_turn_id: row for row in store.get_all_canonical_turns(OWNER)}
    assert all(row.tagged_at for row in physical.values())
    stored_chunks = chunks(inner)
    assert len(stored_chunks) == 2
    for chunk in stored_chunks:
        row = physical[chunk["canonical_turn_id"]]
        assert chunk["side"] in {"user", "assistant"}
        assert chunk["text"] == getattr(row, chunk["side"] + "_content")
        if chunk["side"] == "assistant":
            assert not row.sender_actor_id
    assert [
        dict(row) for row in inner._get_conn().execute("SELECT * FROM canonical_message_sources")
    ] == before_sources


@pytest.mark.parametrize("turn_kind", ["normal", "stub", "tool"])
def test_partial_embedding_failure_leaves_pair_pending_then_retries(
    indexed_history, monkeypatch, turn_kind
):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    rows = store.get_all_canonical_turns(OWNER)
    history = canonical_rows_to_history(rows, include_tagging_identity=True)
    if turn_kind == "stub":
        from virtual_context.core import tagging_pipeline

        tagging_pipeline._ensure_engine_imports()
        monkeypatch.setattr(tagging_pipeline, "_is_stub_content_fn", lambda _text: True)
    elif turn_kind == "tool":
        monkeypatch.setattr(pipeline, "_is_tool_turn", lambda _messages: True)
        pipeline._engine_state.tool_tag_counter = 0
    checkpoints = []
    progress = []
    pipeline._save_state_callback = lambda *args, **kwargs: checkpoints.append(kwargs)
    calls = []

    def fail_second_side(texts):
        calls.extend(texts)
        if len(calls) == 2:
            raise RuntimeError("synthetic second-side outage")
        return [[1.0, 0.0] for _ in texts]

    semantic._embed_fn = fail_second_side
    with pytest.raises(RuntimeError, match="embedding"):
        pipeline.ingest_history(
            history,
            require_existing_canonical=True,
            expected_lifecycle_epoch=1,
            progress_callback=lambda *args: progress.append(args),
        )
    assert len(chunks(inner)) == 1
    assert all(not row.tagged_at for row in store.get_all_canonical_turns(OWNER))
    assert pipeline._turn_tag_index.entries == []
    assert checkpoints == progress == []
    pending = store.iter_complete_untagged_canonical_groups(
        conversation_id=OWNER, expected_lifecycle_epoch=1, batch_size=32
    )
    assert len(pending) == 1
    semantic._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
    assert (
        pipeline.ingest_history(
            history,
            require_existing_canonical=True,
            expected_lifecycle_epoch=1,
            progress_callback=lambda *args: progress.append(args),
        )
        == 1
    )
    assert len(chunks(inner)) == 2
    assert all(row.tagged_at for row in store.get_all_canonical_turns(OWNER))
    assert len(pipeline._turn_tag_index.entries) == len(checkpoints) == 1
    assert len(progress) == (0 if turn_kind == "tool" else 1)


def test_epoch_change_during_tagging_refuses_embedding_and_progress(indexed_history):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    calls = []
    semantic._embed_fn = lambda texts: calls.extend(texts) or [[1.0, 0.0] for _ in texts]

    def bump_epoch(*args, **kwargs):
        conn = inner._get_conn()
        conn.execute(
            "UPDATE conversations SET lifecycle_epoch=lifecycle_epoch+1 WHERE conversation_id=?",
            (OWNER,),
        )
        conn.commit()
        return TagResult(tags=["synthetic"], primary="synthetic", source="keyword")

    pipeline._tag_generator.generate_tags = bump_epoch
    with pytest.raises(RuntimeError, match="epoch"):
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
    assert calls == chunks(inner) == pipeline._turn_tag_index.entries == []
    assert all(not row.tagged_at for row in store.get_all_canonical_turns(OWNER))


def test_already_tagged_replay_and_durable_worker_do_not_double_embed(indexed_history):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    rows = store.get_all_canonical_turns(OWNER)
    calls = []
    semantic._embed_fn = lambda texts: calls.extend(texts) or [[1.0, 0.0] for _ in texts]
    state = SimpleNamespace(engine=SimpleNamespace(_semantic=semantic, _tagging=pipeline))
    assert ProxyState._run_tagging_pipeline(state, rows, expected_lifecycle_epoch=1)
    assert len(calls) == 2
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    assert (
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
        == 1
    )
    assert len(calls) == 2
    assert len(chunks(inner)) == 2


def test_mismatched_history_refuses_before_embedding(indexed_history):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    history[1].content = "An unrelated synthetic response."
    calls = []
    semantic._embed_fn = lambda texts: calls.extend(texts) or [[1.0, 0.0] for _ in texts]
    with pytest.raises(RuntimeError, match="exact row identity"):
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
    assert calls == chunks(inner) == pipeline._turn_tag_index.entries == []
    assert all(not row.tagged_at for row in store.get_all_canonical_turns(OWNER))


def test_tag_cas_rejects_lifecycle_change_during_embedding(indexed_history):
    inner, _rec = indexed_history
    store, semantic, pipeline = runtime(inner)
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    calls = []
    checkpoints = []
    pipeline._save_state_callback = lambda *args, **kwargs: checkpoints.append(kwargs)

    def late_epoch_change(texts):
        calls.extend(texts)
        if len(calls) == 2:
            conn = inner._get_conn()
            conn.execute(
                "UPDATE conversations SET lifecycle_epoch=lifecycle_epoch+1 WHERE conversation_id=?",
                (OWNER,),
            )
            conn.commit()
        return [[1.0, 0.0] for _ in texts]

    semantic._embed_fn = late_epoch_change
    with pytest.raises(RuntimeError, match="strict canonical tagging"):
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
    assert len(calls) == 2
    assert all(not row.tagged_at for row in store.get_all_canonical_turns(OWNER))
    assert checkpoints == pipeline._turn_tag_index.entries == []


def test_failed_later_pair_does_not_advance_checkpoint_or_index(indexed_history):
    from tests.test_audience_reassignment import _ingest

    inner, rec = indexed_history
    _ingest(rec, message_id="4000000000000000074", body="A later synthetic observation.")
    store, semantic, pipeline = runtime(inner)
    pipeline.config.tag_generator.context_bleed_threshold = 0
    history = canonical_rows_to_history(
        store.get_all_canonical_turns(OWNER), include_tagging_identity=True
    )
    checkpoints = []
    progress = []
    calls = []
    pipeline._save_state_callback = lambda *args, **kwargs: checkpoints.append(kwargs)

    def fail_later_pair(texts):
        calls.extend(texts)
        if len(calls) == 4:
            raise RuntimeError("synthetic later-pair outage")
        return [[1.0, 0.0] for _ in texts]

    semantic._embed_fn = fail_later_pair
    with pytest.raises(RuntimeError, match="embedding"):
        pipeline.ingest_history(
            history,
            require_existing_canonical=True,
            expected_lifecycle_epoch=1,
            progress_callback=lambda *args: progress.append(args),
        )
    rows = store.get_all_canonical_turns(OWNER)
    assert [bool(row.tagged_at) for row in rows] == [True, True, False, False]
    assert [entry.turn_number for entry in pipeline._turn_tag_index.entries] == [0]
    assert [call[0] for call in progress] == [1]
    assert checkpoints == []
    semantic._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
    assert (
        pipeline.ingest_history(
            history, require_existing_canonical=True, expected_lifecycle_epoch=1
        )
        == 2
    )
    assert len(chunks(inner)) == 4
    assert [entry.turn_number for entry in pipeline._turn_tag_index.entries] == [0, 1]
    assert checkpoints == [{"last_completed_turn": 1, "last_indexed_turn": 1}]


def test_legacy_combined_row_indexes_both_role_lanes_before_tagging(tmp_path):
    from virtual_context.core.canonical_turns import compute_turn_hash_from_raw
    from virtual_context.storage.sqlite import SQLiteStore

    inner = SQLiteStore(tmp_path / "synthetic-combined.db")
    inner.activate_conversation(OWNER)
    inner.upsert_conversation(tenant_id="synthetic-tenant", conversation_id=OWNER)
    user, assistant = (
        "A synthetic historical user statement.",
        "A synthetic historical assistant response.",
    )
    turn_hash, normalized_user, normalized_assistant = compute_turn_hash_from_raw(
        user, assistant, version=1
    )
    inner.save_canonical_turn(
        OWNER,
        0,
        user,
        assistant,
        canonical_turn_id="synthetic-legacy-combined",
        turn_group_number=0,
        turn_hash=turn_hash,
        hash_version=1,
        normalized_user_text=normalized_user,
        normalized_assistant_text=normalized_assistant,
    )
    try:
        store, _semantic, pipeline = runtime(inner)
        history = canonical_rows_to_history(
            store.get_all_canonical_turns(OWNER), include_tagging_identity=True
        )
        assert (
            pipeline.ingest_history(
                history, require_existing_canonical=True, expected_lifecycle_epoch=1
            )
            == 1
        )
        assert [
            (chunk["canonical_turn_id"], chunk["side"], chunk["text"]) for chunk in chunks(inner)
        ] == [
            ("synthetic-legacy-combined", "assistant", assistant),
            ("synthetic-legacy-combined", "user", user),
        ]
        assert store.get_all_canonical_turns(OWNER)[0].tagged_at
    finally:
        inner.close()
