"""BUG-076: occurrence time must come from exact source evidence."""
from __future__ import annotations

import hashlib
import sqlite3
from uuid import UUID

import pytest

from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import VirtualContextConfig, normalize_source_attestation


pytestmark = pytest.mark.regression("BUG-076")
OCCURRED = "2026-08-20T12:34:56.789Z"


def _claim(body, occurred_at=None):
    digest = hashlib.sha256(body.encode()).hexdigest()
    claim = dict(version=1, agent_scope_id="agent", platform="discord",
                 account_id="account", message_id="111111111111111111", channel_id="channel-1",
                 guild_id="guild-1", author_id="person-1", transport_body_sha256=digest,
                 canonical_body_sha256=digest, projection_version="adapter-v1",
                 reply_target_message_id="")
    if occurred_at is not None:
        claim["occurred_at"] = occurred_at
    return claim


@pytest.fixture
def source_store(tmp_path):
    store = SQLiteStore(tmp_path / "event-times.db")
    store.activate_conversation("owner")
    store.upsert_conversation(tenant_id="tenant", conversation_id="owner")
    config = VirtualContextConfig(conversation_id="owner")
    semantic = SemanticSearchManager(store, config)
    semantic._embed_fn = None
    yield store, IngestReconciler(store=store, semantic=semantic)
    store.close()


def _ingest(store, reconciler, occurred_at=None):
    reconciler.ingest_single(
        "owner", user_content="Use the agreed title for thirty days.",
        assistant_content="Agreed, for thirty days.",
        user_origin_channel_id="channel-1", user_sender_actor_id="actor:discord:person-1",
        user_reply_edge=dict(source_message_id="111111111111111111", audience_conversation_id="owner",
                             audience_attribution_version=1),
        user_source_claim=_claim("Use the agreed title for thirty days.", occurred_at),
        expected_conversation_generation=store.get_conversation_generation("owner"),
        expected_lifecycle_epoch=1,
    )
    with store._relational_connection() as conn:
        return {key: str(value) if isinstance(value, UUID) else value
                for key, value in dict(conn.execute("SELECT * FROM canonical_message_sources").fetchone()).items()}


def _times(store, source, tenant="tenant", owner="owner"):
    return store.get_canonical_source_event_times(
        [(owner, source["canonical_turn_id"])], tenant_id=tenant,
    )


def test_source_occurrence_claim_preserves_adapter_utc_format():
    claim = _claim("source", OCCURRED)
    assert normalize_source_attestation(claim) == claim


@pytest.mark.parametrize("invalid", ["2026-08-20", "2026-08-20T12:34:56", "2026-08-20T12:34:56.789+01:00", "2026-02-30T12:34:56.789Z", 123, None])
def test_source_occurrence_claim_rejects_invalid_optional_time(invalid):
    claim = _claim("source")
    claim["occurred_at"] = invalid
    assert normalize_source_attestation(claim) == {}


def test_source_occurrence_admission_has_no_ingestion_date_fallback(source_store):
    store, rec = source_store
    source = _ingest(store, rec)
    assert _times(store, source) == {}
    before = dict(source)
    _ingest(store, rec, OCCURRED)
    assert _times(store, source) == {("owner", source["canonical_turn_id"]): OCCURRED}
    assert dict(store._get_conn().execute("SELECT * FROM canonical_message_sources").fetchone()) == before
    assert _times(store, source, tenant="another-tenant") == {}
    assert _times(store, source, owner="another-owner") == {}


def test_source_occurrence_replay_is_optional_immutable_and_atomic(source_store):
    store, rec = source_store
    source = _ingest(store, rec, OCCURRED)
    assert _times(store, source) == {("owner", source["canonical_turn_id"]): OCCURRED}
    before = [dict(row) for row in store._get_conn().execute("SELECT * FROM canonical_turns ORDER BY canonical_turn_id")]
    _ingest(store, rec)
    _ingest(store, rec, OCCURRED)
    with pytest.raises(CanonicalSourceConflict):
        _ingest(store, rec, "2026-08-21T12:34:56.789Z")
    assert _times(store, source) == {("owner", source["canonical_turn_id"]): OCCURRED}
    assert dict(store._get_conn().execute("SELECT * FROM canonical_message_sources").fetchone()) == source
    assert [dict(row) for row in store._get_conn().execute("SELECT * FROM canonical_turns ORDER BY canonical_turn_id")] == before
    with pytest.raises(sqlite3.IntegrityError):
        store._get_conn().execute("UPDATE canonical_source_event_times SET occurred_at=?", ("2026-08-21T12:34:56.789Z",))


def test_source_occurrence_read_hides_stale_membership_and_deleted_lifecycle(source_store):
    store, rec = source_store
    source = _ingest(store, rec, OCCURRED)
    conn = store._get_conn()
    conn.execute("UPDATE conversation_lifecycle SET deleted=1 WHERE conversation_id='owner'")
    assert _times(store, source) == {}
    conn.execute("UPDATE conversation_lifecycle SET deleted=0 WHERE conversation_id='owner'")
    conn.execute("DROP TRIGGER trg_source_event_time_update")
    conn.execute("UPDATE canonical_source_event_times SET source_fingerprint='stale'")
    assert _times(store, source) == {}


def test_source_occurrence_insert_failure_rolls_back_pair_and_membership(source_store, monkeypatch):
    from virtual_context.storage import source_event_times
    store, rec = source_store
    original = source_event_times.record_source_event_time

    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise CanonicalSourceConflict("synthetic event insertion failure")

    monkeypatch.setattr(source_event_times, "record_source_event_time", fail_after_insert)
    with pytest.raises(CanonicalSourceConflict, match="synthetic event"):
        _ingest(store, rec, OCCURRED)
    for table in ("canonical_turns", "canonical_message_sources", "canonical_source_event_times"):
        assert store._get_conn().execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_source_occurrence_same_name_noop_guard_fails_schema_assertion(source_store):
    from virtual_context.storage.source_event_times import assert_source_event_time_schema
    store, _ = source_store
    conn = store._get_conn()
    conn.execute("DROP TRIGGER trg_source_event_time_update")
    conn.execute("CREATE TRIGGER trg_source_event_time_update BEFORE UPDATE ON canonical_source_event_times BEGIN SELECT 1; END")
    with pytest.raises(RuntimeError, match="unguarded"):
        assert_source_event_time_schema(conn, "sqlite")


def test_source_occurrence_enrichment_invalidates_inflight_card_once(source_store):
    store, rec = source_store
    _ingest(store, rec)
    store.upsert_actor_profile_from_turn("owner", "actor:discord:person-1", seen_at=OCCURRED)
    conn = store._get_conn()
    conn.execute("UPDATE actor_profiles SET card_dirty=0,card_build_marker='old-build'")
    _ingest(store, rec, OCCURRED)
    profile = dict(conn.execute("SELECT * FROM actor_profiles").fetchone())
    assert profile["card_dirty"] == 1 and profile["card_build_marker"] == ""
    conn.execute("UPDATE actor_profiles SET card_dirty=0,card_build_marker='new-build'")
    before = dict(conn.execute("SELECT * FROM actor_profiles").fetchone())
    _ingest(store, rec, OCCURRED)
    assert dict(conn.execute("SELECT * FROM actor_profiles").fetchone()) == before


@pytest.mark.parametrize("legacy_metadata", ["missing_audience", "inherited_actor"])
@pytest.mark.parametrize("initial_occurrence", [None, OCCURRED])
def test_optional_occurrence_preserves_exact_legacy_replay(source_store, legacy_metadata, initial_occurrence):
    store, rec = source_store
    store.save_canonical_turn(
        "owner", 0, "Use the agreed title for thirty days.", "",
        canonical_turn_id="10000000-0000-0000-0000-000000000001", turn_group_number=0,
        origin_channel_id="channel-1", sender_actor_id="actor:discord:person-1",
        source_message_id="111111111111111111", audience_conversation_id="owner",
        audience_attribution_version=1,
    )
    store.save_canonical_turn(
        "owner", 1, "", "Agreed, for thirty days.",
        canonical_turn_id="10000000-0000-0000-0000-000000000002", turn_group_number=0,
        sender_actor_id="actor:discord:person-1" if legacy_metadata == "inherited_actor" else "",
        audience_conversation_id="owner" if legacy_metadata == "inherited_actor" else "",
        audience_attribution_version=1 if legacy_metadata == "inherited_actor" else 0,
    )
    source = _ingest(store, rec, initial_occurrence)
    with store._relational_connection() as conn:
        before = [dict(r) for r in conn.execute("SELECT * FROM canonical_turns ORDER BY canonical_turn_id")]
    assert _ingest(store, rec, OCCURRED) == source
    assert _times(store, source) == {}
    with store._relational_connection() as conn:
        assert [dict(r) for r in conn.execute("SELECT * FROM canonical_turns ORDER BY canonical_turn_id")] == before


def test_prior_occurrence_conflict_is_fatal_before_legacy_refusal(source_store, monkeypatch):
    from virtual_context.core.exceptions import SourceEventTimeUnavailable
    from virtual_context.storage import source_event_times

    store, rec = source_store
    _ingest(store, rec, OCCURRED)

    def legacy_metadata(*args, **kwargs):
        raise SourceEventTimeUnavailable("legacy metadata")

    monkeypatch.setattr(source_event_times, "_exact_source", legacy_metadata)
    with pytest.raises(CanonicalSourceConflict, match="timestamp conflicts"):
        _ingest(store, rec, "2026-08-21T12:34:56.789Z")


def test_source_occurrence_preserves_existing_audience_and_channel_receipts(tmp_path):
    from tests.test_assistant_channel_enrichment import historical, stack
    from tests.test_audience_reassignment import OWNER, TARGET, TENANT

    base = stack.__wrapped__(tmp_path)
    store, rec = next(base)
    try:
        store, rec, audience_manifest, assistant_id = historical.__wrapped__((store, rec))
        conn = store._get_conn()
        source = dict(conn.execute("SELECT * FROM canonical_message_sources").fetchone())
        key = (OWNER, source["canonical_turn_id"])
        user = store.get_canonical_turn_rows_by_id([key], internal_validation=True)[key]
        user.source_claim = {"version": 1, **{k: source[k] for k in _claim("unused") if k != "version"}, "occurred_at": OCCURRED}
        before = [dict(row) for row in conn.execute("SELECT * FROM canonical_audience_reassignments ORDER BY canonical_turn_id")]
        store.attest_canonical_source_event_time(user)
        manifest = store.plan_assistant_channel_enrichment(
            OWNER, tenant_id=TENANT, audience_conversation_id=TARGET,
            expected_lifecycle_epoch=1, operation_id="source-event-channel-control",
            assistant_canonical_turn_ids=[assistant_id],
        )
        store.enrich_assistant_channels(manifest, dry_run=False)
        assert store.get_canonical_source_event_times([key], tenant_id=TENANT) == {key: OCCURRED}
        assert store.reassign_audience(audience_manifest)["already_applied"]
        assert [dict(row) for row in conn.execute("SELECT * FROM canonical_audience_reassignments ORDER BY canonical_turn_id")] == before
        assert dict(conn.execute("SELECT * FROM canonical_message_sources").fetchone()) == source
    finally:
        base.close()
