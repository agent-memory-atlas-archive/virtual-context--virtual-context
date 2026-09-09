"""BUG-072: exact completions retain their source channel without actor bleed."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.quote_search import find_quote
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Message, SpeakerRetrievalContext, build_user_turn_metadata


pytestmark = pytest.mark.regression("BUG-072")

OWNER = "completion-channel-test"
CHANNEL = "100000000000000001"
OTHER_CHANNEL = "100000000000000002"
ACTOR = "100000000000000003"
MESSAGE = "100000000000000004"
BODY = "Did you accept the wager result?"
ANSWER = "I concede the lunar-crown wager. You won the agreed title."


@pytest.fixture
def pipeline(tmp_path):
    store = SQLiteStore(tmp_path / "completion.db")
    store.activate_conversation(OWNER)
    store.upsert_conversation(tenant_id="test-tenant", conversation_id=OWNER)
    config = VirtualContextConfig(conversation_id=OWNER)
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    rec = IngestReconciler(store=store, semantic=semantic)
    yield rec, store, semantic
    store.close()


def _claim(channel_id=CHANNEL):
    digest = hashlib.sha256(BODY.encode()).hexdigest()
    return {
        "version": 1,
        "agent_scope_id": "test-agent",
        "platform": "discord",
        "account_id": "test-account",
        "message_id": MESSAGE,
        "channel_id": channel_id,
        "guild_id": "100000000000000005",
        "author_id": ACTOR,
        "transport_body_sha256": digest,
        "canonical_body_sha256": digest,
        "projection_version": "openclaw-current-user-v1",
        "reply_target_message_id": "",
    }


def _completion(rec, *, assistant_channel="", claim_channel=CHANNEL, attested=True):
    return rec.ingest_single(
        OWNER,
        user_content=BODY,
        assistant_content=ANSWER,
        user_sender="Synthetic Participant",
        user_sender_actor_id=f"actor:discord:{ACTOR}",
        user_origin_channel_id=CHANNEL,
        user_origin_channel_label="#workshop",
        assistant_origin_channel_id=assistant_channel,
        user_reply_edge={
            "source_message_id": MESSAGE,
            "audience_conversation_id": OWNER,
            "audience_attribution_version": 1,
        },
        user_source_claim=_claim(claim_channel) if attested else None,
        expected_conversation_generation=rec._store.get_conversation_generation(OWNER),
        expected_lifecycle_epoch=1,
    )


@pytest.mark.parametrize("assistant_channel", ["", CHANNEL])
def test_attested_completion_is_recalled_across_channels(pipeline, assistant_channel):
    rec, store, semantic = pipeline
    _completion(rec, assistant_channel=assistant_channel)
    rows = store.get_all_canonical_turns(OWNER)
    assistant = next(row for row in rows if row.assistant_content)
    context = SpeakerRetrievalContext(
        tenant_id="test-tenant",
        owner_conversation_id=OWNER,
        audience_conversation_id=OWNER,
        audience_channel_scope="conversation",
        audience_channel_id="",
        request_origin_channel_id=OTHER_CHANNEL,
    )
    result = find_quote(
        store, semantic, "lunar-crown", conversation_id=OWNER,
        speaker_context=context,
    )
    assert result["found"] is True, result
    assert any(ANSWER in row["excerpt"] for row in result["results"])
    assert assistant.origin_channel_id == CHANNEL
    assert assistant.sender == assistant.sender_actor_id == ""
    assert assistant.source_message_id == assistant.reply_subject_actor_id == ""
    assert assistant.audience_conversation_id == OWNER
    assert assistant.audience_attribution_version == 1
    assert len(rows) == 2


def test_attested_completion_rejects_conflicting_assistant_channel(pipeline):
    rec, store, _ = pipeline
    with pytest.raises(CanonicalSourceConflict, match="assistant.*channel"):
        _completion(rec, assistant_channel=OTHER_CHANNEL)
    assert store.get_all_canonical_turns(OWNER) == []


def test_unattested_completion_does_not_inherit_user_channel(pipeline):
    rec, store, _ = pipeline
    _completion(rec, attested=False)
    assistant = next(row for row in store.get_all_canonical_turns(OWNER) if row.assistant_content)
    assert assistant.origin_channel_id == ""


def test_conflicting_user_source_is_rejected_before_assistant_admission(pipeline):
    rec, store, _ = pipeline
    with pytest.raises(CanonicalSourceConflict):
        _completion(rec, claim_channel=OTHER_CHANNEL)
    assert store.get_all_canonical_turns(OWNER) == []


def _seed_legacy_user(store):
    store.save_canonical_turn(
        OWNER, 0, BODY, "",
        canonical_turn_id="legacy-user",
        turn_group_number=0,
        sender="Synthetic Participant",
        sender_actor_id=f"actor:discord:{ACTOR}",
        source_message_id=MESSAGE,
        origin_channel_id=CHANNEL,
        audience_conversation_id=OWNER,
        audience_attribution_version=1,
    )


def test_completion_of_existing_user_preserves_source_channel(pipeline):
    rec, store, _ = pipeline
    _seed_legacy_user(store)
    result = _completion(rec)
    assert result.merge_mode == "source_exact_legacy_user_adopt"
    rows = store.get_all_canonical_turns(OWNER)
    assert len(rows) == 2
    assert rows[0].canonical_turn_id == "legacy-user"
    assert rows[1].origin_channel_id == CHANNEL
    assert rows[1].sender_actor_id == rows[1].sender == ""


def test_historical_resend_reports_stored_channel_without_repair(pipeline):
    rec, store, _ = pipeline
    _seed_legacy_user(store)
    store.save_canonical_turn(
        OWNER, 1, "", ANSWER,
        canonical_turn_id="legacy-assistant",
        turn_group_number=0,
        audience_conversation_id=OWNER,
        audience_attribution_version=1,
    )
    adopted = _completion(rec)
    assert adopted.merge_mode == "source_exact_legacy_pair_adopt"
    replay = _completion(rec)
    assert replay.merge_mode == "source_exact_resend"
    assert replay.turns_written == 0
    assistant = next(row for row in replay.rows if row.assistant_content)
    assert assistant.origin_channel_id == ""
    stored = store.get_all_canonical_turns(OWNER)
    assert len(stored) == 2
    assert stored[1].canonical_turn_id == "legacy-assistant"
    assert stored[1].origin_channel_id == ""


def test_completed_turn_surface_preserves_channel_without_assistant_envelope(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    config = load_config(config_dict={
        "context_window": 10000,
        "tenant_id": "test-tenant",
        "conversation_id": OWNER,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "engine.db")}},
        "tag_generator": {"type": "keyword"},
    })
    engine = VirtualContextEngine(
        config=config, embedding_provider=SimpleNamespace(get_embed_fn=lambda: None),
    )
    try:
        engine._store.upsert_conversation(tenant_id="test-tenant", conversation_id=OWNER)
        metadata = build_user_turn_metadata(
            sender_name="Synthetic Participant",
            sender_actor_id=f"actor:discord:{ACTOR}",
            source_message_id=MESSAGE,
            origin_channel_id=CHANNEL,
            source_conversation_key="agent:test-agent:discord:guild:100000000000000005",
            source_attestation=_claim(),
        )
        outcome = engine.persist_completed_turn(
            [],
            source_audience_conversation_id=OWNER,
            user_turn_metadata=metadata,
            completed_user_message=Message(role="user", content=BODY),
            completed_assistant_message=Message(role="assistant", content=ANSWER),
            expected_owner_conversation_id=OWNER,
            expected_conversation_generation=engine._store.get_conversation_generation(OWNER),
            expected_lifecycle_epoch=engine._engine_state.lifecycle_epoch,
            return_outcome=True,
        )
        assert outcome["status"] == "accepted", outcome
        rows = engine._store.get_all_canonical_turns(OWNER)
        assert len(rows) == 2
        assistant = next(row for row in rows if row.assistant_content)
        assert assistant.origin_channel_id == CHANNEL
        assert assistant.audience_conversation_id == OWNER
        assert assistant.sender_actor_id == assistant.sender == ""
    finally:
        engine.close()
