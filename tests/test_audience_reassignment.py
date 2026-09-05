"""BUG-070: audited audience changes preserve exact source admission.

All identifiers and transcript text in these fixtures are synthetic. Audience
identifiers deliberately have no relationship to transport channel identifiers.
"""

from __future__ import annotations

import copy
import hashlib
import sqlite3
from pathlib import Path

import pytest

from virtual_context.config import VirtualContextConfig
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import StorageConfig, TagGeneratorConfig


pytestmark = pytest.mark.regression("BUG-070")

OWNER = "opaque-owner-070"
TENANT = "synthetic-tenant-070"
ORIGINAL = "audience-before-070"
TARGET = "audience-after-070"
OTHER = "audience-unrelated-070"
CHANNEL = "1000000000000000070"
GUILD = "2000000000000000070"
ACTOR = "3000000000000000070"
MESSAGE = "4000000000000000070"
BODY = "A synthetic source observation."
ANSWER = "A synthetic source response."
REFUSAL = (ValueError, RuntimeError, CanonicalSourceConflict)


def _seed_durable_card(store):
    from virtual_context.types import ActorCardEntry, ActorCardEntrySource

    actor = f"actor:discord:{ACTOR}"
    store.upsert_actor_profile_from_turn(
        OWNER, actor, "Synthetic contributor", seen_at="2026-01-01T00:00:00+00:00",
    )
    turn = next(row for row in store.get_all_canonical_turns(OWNER) if row.user_content)
    entry = ActorCardEntry(
        id="synthetic-durable-card-070", kind="communication_pref",
        body="Prefers concise responses.", confidence=0.7,
        audience_scope="cross_context", sensitivity="normal",
    )
    source = ActorCardEntrySource(
        entry_id=entry.id, tenant_id=TENANT, owner_conversation_id=OWNER,
        audience_conversation_id=ORIGINAL, audience_channel_id=CHANNEL,
        canonical_turn_id=turn.canonical_turn_id,
    )
    assert store.replace_actor_card(
        TENANT, actor, [(entry, [source])], input_hash="synthetic-before-070",
        expected_source_epochs={OWNER: 1, ORIGINAL: 1},
    ) == 1
    return actor


@pytest.fixture
def stack(tmp_path: Path):
    store = SQLiteStore(tmp_path / "audience-reassignment.db")
    for conversation_id in (OWNER, ORIGINAL, TARGET, OTHER):
        store.activate_conversation(conversation_id)
        store.upsert_conversation(tenant_id=TENANT, conversation_id=conversation_id)
    for alias in (ORIGINAL, TARGET, OTHER):
        store.save_conversation_alias(alias, OWNER)
    store._get_conn().execute(
        "UPDATE conversations SET phase = 'active' WHERE conversation_id = ?",
        (OWNER,),
    )
    store._get_conn().commit()
    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(backend="sqlite"),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    reconciler = IngestReconciler(store=store, semantic=semantic)
    _ingest(reconciler)
    yield store, reconciler
    store.close()


def _ingest(
    reconciler: IngestReconciler,
    *,
    audience: str = ORIGINAL,
    message_id: str = MESSAGE,
    body: str = BODY,
    owner: str = OWNER,
):
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return reconciler.ingest_single(
        owner,
        user_content=body,
        assistant_content=ANSWER,
        user_origin_channel_id=CHANNEL,
        user_sender_actor_id=f"actor:discord:{ACTOR}",
        user_reply_edge={
            "source_message_id": message_id,
            "audience_conversation_id": audience,
            "audience_attribution_version": 1,
        },
        user_source_claim={
            "version": 1,
            "agent_scope_id": "synthetic-agent",
            "platform": "discord",
            "account_id": "synthetic-account",
            "message_id": message_id,
            "channel_id": CHANNEL,
            "guild_id": GUILD,
            "author_id": ACTOR,
            "transport_body_sha256": digest,
            "canonical_body_sha256": digest,
            "projection_version": "openclaw-current-user-v1",
            "reply_target_message_id": "",
        },
        expected_conversation_generation=(
            reconciler._store.get_conversation_generation(owner)
        ),
        expected_lifecycle_epoch=1,
    )


def _plan(store: SQLiteStore, **overrides) -> dict:
    arguments = {
        "tenant_id": TENANT,
        "expected_lifecycle_epoch": store.get_lifecycle_epoch(OWNER),
        "operation_id": "synthetic-reassignment-070",
    }
    arguments.update(overrides)
    return store.plan_audience_reassignment(
        OWNER, ORIGINAL, TARGET, **arguments,
    )


def _canonical(store: SQLiteStore) -> dict[str, dict]:
    return {
        row["canonical_turn_id"]: dict(row)
        for row in store._get_conn().execute(
            "SELECT * FROM canonical_turns ORDER BY canonical_turn_id"
        )
    }


def _ledger(store: SQLiteStore) -> list[dict]:
    return [
        dict(row) for row in store._get_conn().execute(
            "SELECT * FROM canonical_message_sources ORDER BY message_id"
        )
    ]


def _dump(store: SQLiteStore) -> tuple[str, ...]:
    return tuple(store._get_conn().iterdump())


def _ledger_evidence(store: SQLiteStore) -> list[dict]:
    """Replay may refresh observation time; original evidence stays immutable."""
    return [
        {key: value for key, value in row.items() if key != "observed_at"}
        for row in _ledger(store)
    ]


def test_plan_and_default_dry_run_are_read_only_and_pin_complete_pair(stack):
    store, _ = stack
    before = _dump(store)
    manifest = _plan(store)
    assert manifest["version"] == 1
    assert manifest["operation_id"] == "synthetic-reassignment-070"
    assert manifest["tenant_id"] == TENANT
    assert manifest["owner_conversation_id"] == OWNER
    assert manifest["from_audience"] == ORIGINAL
    assert manifest["to_audience"] == TARGET
    assert manifest["expected_lifecycle_epoch"] == 1
    assert len(manifest["rows"]) == 2
    assert len(manifest["manifest_digest"]) == 64
    result = store.reassign_audience(manifest)
    assert result["selected"] == 2
    assert result["updated"] == 0
    assert result["dry_run"] is True
    assert _dump(store) == before


def test_scope_change_invalidates_without_erasing_durable_card_evidence(stack):
    store, _ = stack
    actor = _seed_durable_card(store)
    before = store.list_actor_card_carryovers(TENANT, actor)
    result = store.reassign_audience(_plan(store), dry_run=False)
    assert result["cards_invalidated"] == 1
    assert store.list_actor_card_carryovers(TENANT, actor) == before
    profile = store.get_actor_profile(TENANT, actor)
    assert profile.card_dirty and profile.card_invalid
    assert store.get_actor_card(
        TENANT, actor, owner_conversation_id=OWNER, audience_conversation_id=TARGET,
    ) is None


@pytest.mark.parametrize("deleted_alias", [ORIGINAL, TARGET])
def test_deleted_audience_lifecycle_refuses_plan_and_apply(stack, deleted_alias):
    store, _ = stack
    manifest = _plan(store)
    store._get_conn().execute(
        "UPDATE conversation_lifecycle SET deleted=1 WHERE conversation_id=?", (deleted_alias,),
    )
    before = _dump(store)
    with pytest.raises(ValueError, match="lifecycle.*deleted"):
        _plan(store)
    with pytest.raises(ValueError, match="lifecycle.*deleted"):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before


def test_apply_moves_exact_audience_and_preserves_ledger_and_source_bytes(stack):
    store, reconciler = stack
    _ingest(reconciler, audience=OTHER, message_id="4000000000000000071")
    store.activate_conversation("opaque-other-owner-070")
    store.upsert_conversation(
        tenant_id="synthetic-other-tenant-070",
        conversation_id="opaque-other-owner-070",
    )
    store.save_canonical_turn(
        "opaque-other-owner-070", 0, "Unrelated tenant source.", "",
        canonical_turn_id="synthetic-other-owner-row-070",
        audience_conversation_id=ORIGINAL,
        audience_attribution_version=1,
    )
    before = _canonical(store)
    ledger_before = _ledger(store)
    manifest = _plan(store)
    result = store.reassign_audience(manifest, dry_run=False)
    assert result["selected"] == result["updated"] == 2
    assert result["dry_run"] is False
    after = _canonical(store)
    assert after.keys() == before.keys()
    for row_id, original in before.items():
        row = after[row_id]
        selected = (
            original["conversation_id"] == OWNER
            and original["audience_conversation_id"] == ORIGINAL
        )
        assert row["audience_conversation_id"] == (
            TARGET if selected else original["audience_conversation_id"]
        )
        for field in (
            "conversation_id", "canonical_turn_id", "turn_group_number",
            "source_message_id", "user_content", "assistant_content",
            "user_raw_content", "assistant_raw_content", "turn_hash",
            "hash_version", "origin_channel_id", "origin_channel_label",
            "sender_actor_id", "reply_target_message_id",
        ):
            assert row[field] == original[field], (row_id, field)
    assert _ledger(store) == ledger_before
    assert ledger_before[0]["audience_conversation_id"] == ORIGINAL


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_unattested_selected_row_refuses_entire_plan_and_apply(stack, role):
    store, _ = stack
    manifest = _plan(store)
    store.save_canonical_turn(
        OWNER, 4, "Synthetic unverified source." if role == "user" else "",
        "Synthetic unverified response." if role == "assistant" else "",
        audience_conversation_id=ORIGINAL, audience_attribution_version=1,
    )
    before = _dump(store)
    with pytest.raises(ValueError, match="unattested canonical row"):
        _plan(store)
    with pytest.raises(ValueError, match="unattested canonical row"):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before


def _seed_historical_unattested_transfer(conn, *, dialect, receipt_role="user"):
    """Represent an older receipt with no source ledger; never create via API."""
    from virtual_context.storage.audience_reassignment import (
        canonical_row_fingerprint, source_membership_fingerprint,
    )

    p = "%s" if dialect == "postgres" else "?"
    conn.execute("DELETE FROM canonical_message_sources")
    if receipt_role == "user":
        conn.execute("DELETE FROM canonical_turns WHERE assistant_content <> ''")
    conn.execute(
        f"UPDATE canonical_turns SET audience_conversation_id={p}", (TARGET,),
    )
    body_column = "user_content" if receipt_role == "user" else "assistant_content"
    row = dict(conn.execute(
        f"SELECT * FROM canonical_turns WHERE {body_column} <> ''",
    ).fetchone())
    operation = {
        "operation_id": "synthetic-historical-receipt-070", "tenant_id": TENANT,
        "owner_conversation_id": OWNER, "from_audience": ORIGINAL,
        "to_audience": TARGET, "expected_lifecycle_epoch": 1,
        "manifest_digest": "f" * 64, "row_count": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    receipt = {
        **{key: value for key, value in operation.items()
           if key not in {"expected_lifecycle_epoch", "row_count"}},
        "canonical_turn_id": row["canonical_turn_id"],
        "from_attribution_version": 1, "to_attribution_version": 1,
        "turn_hash": row["turn_hash"],
        "source_fingerprint": source_membership_fingerprint([]),
        "row_fingerprint": canonical_row_fingerprint(row),
    }
    manifest = _resign_manifest({
        "version": 1,
        **{key: value for key, value in operation.items()
           if key not in {"manifest_digest", "row_count", "created_at"}},
        "rows": [{key: value for key, value in receipt.items()
                  if key not in {"operation_id", "tenant_id", "owner_conversation_id",
                                 "manifest_digest", "created_at"}}],
    })
    operation["manifest_digest"] = receipt["manifest_digest"] = manifest["manifest_digest"]
    for table, values in (("audience_reassignment_operations", operation),
                          ("canonical_audience_reassignments", receipt)):
        conn.execute(
            f"INSERT INTO {table} ({','.join(values)}) "
            f"VALUES ({','.join(p for _ in values)})", tuple(values.values()),
        )


@pytest.mark.parametrize("receipt_role", ["user", "assistant"])
def test_historical_unattested_receipt_refuses_adoption_before_pair_commit(stack, receipt_role):
    store, reconciler = stack
    _seed_historical_unattested_transfer(
        store._get_conn(), dialect="sqlite", receipt_role=receipt_role,
    )
    before = _dump(store)
    with pytest.raises(CanonicalSourceConflict, match="audience-reassigned legacy row"):
        _ingest(reconciler, audience=TARGET)
    assert _dump(store) == before


def test_historical_unattested_receipt_cannot_replan_as_supported_operation(stack):
    store, _ = stack
    _seed_historical_unattested_transfer(store._get_conn(), dialect="sqlite")
    before = _dump(store)
    with pytest.raises(ValueError, match="unattested canonical row"):
        _plan(store, operation_id="synthetic-historical-receipt-070")
    assert _dump(store) == before


def test_exact_replay_accepts_original_and_target_but_rejects_alien_audience(stack):
    store, reconciler = stack
    store.reassign_audience(_plan(store), dry_run=False)
    expected_rows = _canonical(store)
    expected_ledger = _ledger_evidence(store)
    assert _ingest(reconciler, audience=ORIGINAL).turns_written == 0
    assert _ingest(reconciler, audience=TARGET).turns_written == 0
    with pytest.raises(CanonicalSourceConflict):
        _ingest(reconciler, audience=OTHER)
    assert _canonical(store).keys() == expected_rows.keys()
    assert all(
        row["audience_conversation_id"] == TARGET
        for row in _canonical(store).values()
    )
    assert _ledger_evidence(store) == expected_ledger


def test_audited_move_does_not_allow_unaudited_sql_rewrites(stack):
    store, _ = stack
    store.reassign_audience(_plan(store), dry_run=False)
    before = _canonical(store)
    ledger_before = _ledger(store)
    user = next(row for row in before.values() if row["user_content"])
    assistant = next(row for row in before.values() if row["assistant_content"])
    for row_id, field, value in (
        (user["canonical_turn_id"], "audience_conversation_id", OTHER),
        (assistant["canonical_turn_id"], "audience_conversation_id", OTHER),
        (user["canonical_turn_id"], "user_content", "Unverified edit."),
        (assistant["canonical_turn_id"], "assistant_content", "Unverified response."),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            store._get_conn().execute(
                f"UPDATE canonical_turns SET {field} = ? WHERE canonical_turn_id = ?",
                (value, row_id),
            )
        store._get_conn().rollback()
    assert _canonical(store) == before
    assert _ledger(store) == ledger_before


def test_stale_lifecycle_epoch_refuses_plan_and_apply_without_partial_move(stack):
    store, _ = stack
    manifest = _plan(store)
    store._get_conn().execute(
        "UPDATE conversations SET lifecycle_epoch = 2 WHERE conversation_id = ?",
        (OWNER,),
    )
    store._get_conn().commit()
    before = _dump(store)
    with pytest.raises(REFUSAL):
        _plan(store, expected_lifecycle_epoch=1)
    with pytest.raises(REFUSAL):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before


def test_cross_tenant_or_tampered_manifest_cannot_authorize_a_move(stack):
    store, _ = stack
    manifest = _plan(store)
    before = _dump(store)
    with pytest.raises(REFUSAL):
        _plan(store, tenant_id="synthetic-alien-tenant")
    for field, value in (
        ("tenant_id", "synthetic-alien-tenant"),
        ("to_audience", OTHER),
        ("owner_conversation_id", "opaque-alien-owner"),
        ("rows", manifest["rows"][:-1]),
    ):
        tampered = copy.deepcopy(manifest)
        tampered[field] = value
        with pytest.raises(REFUSAL):
            store.reassign_audience(tampered, dry_run=False)
    assert _dump(store) == before


def _resign_manifest(manifest):
    import json

    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return manifest


@pytest.mark.parametrize("change", ["tenant", "alias", "pair"])
def test_recomputed_digest_cannot_bypass_live_scope_or_pair_fences(stack, change):
    store, _ = stack
    manifest = _plan(store)
    if change == "tenant":
        manifest["tenant_id"] = "synthetic-alien-tenant"
        expected = "cross tenants"
    elif change == "alias":
        # The named audience exists in the tenant but belongs to another owner.
        store._get_conn().execute(
            "UPDATE conversation_aliases SET target_id=? WHERE alias_id=?",
            (OTHER, TARGET),
        )
        manifest["operation_id"] = "synthetic-retargeted-alias-070"
        expected = "direct aliases"
    else:
        manifest["rows"] = manifest["rows"][:-1]
        expected = "stale or incomplete"
    _resign_manifest(manifest)
    before = _dump(store)
    with pytest.raises(ValueError, match=expected):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before


def test_source_set_change_after_plan_refuses_entire_apply(stack):
    store, reconciler = stack
    manifest = _plan(store)
    _ingest(
        reconciler,
        message_id="4000000000000000072",
        body="Synthetic source arriving after planning.",
    )
    before = _dump(store)
    with pytest.raises(REFUSAL):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before
    assert all(
        row["audience_conversation_id"] == ORIGINAL
        for row in _canonical(store).values()
    )


def test_exact_apply_replay_is_idempotent_and_second_move_is_refused(stack):
    store, _ = stack
    manifest = _plan(store)
    assert store.reassign_audience(manifest, dry_run=False)["updated"] == 2
    before = _dump(store)
    replay = store.reassign_audience(manifest, dry_run=False)
    assert replay["selected"] == 2
    assert replay["updated"] == 0
    assert replay["dry_run"] is False
    assert _dump(store) == before
    with pytest.raises(REFUSAL):
        second = store.plan_audience_reassignment(
            OWNER, TARGET, OTHER,
            tenant_id=TENANT,
            expected_lifecycle_epoch=1,
            operation_id="synthetic-second-reassignment-070",
        )
        store.reassign_audience(second, dry_run=False)
    assert _dump(store) == before


@pytest.mark.parametrize(
    ("assistant_audience", "assistant_version", "expected_updated"),
    [("", 0, 2), (TARGET, 1, 1), (ORIGINAL, 0, 2), (TARGET, 0, 2)],
    ids=["blank-legacy-assistant", "already-target-assistant",
         "original-audience-legacy-assistant", "target-audience-legacy-assistant"],
)
def test_complete_pair_pins_assistant_with_different_existing_scope(
    stack, assistant_audience, assistant_version, expected_updated,
):
    store, reconciler = stack
    assistant = next(
        row for row in _canonical(store).values() if row["assistant_content"]
    )
    conn = store._get_conn()
    conn.execute(
        "UPDATE canonical_turns SET audience_conversation_id = ?, "
        "audience_attribution_version = ? WHERE canonical_turn_id = ?",
        (assistant_audience, assistant_version, assistant["canonical_turn_id"]),
    )
    conn.commit()
    ledger = _ledger(store)
    manifest = _plan(store)
    pinned = next(
        row for row in manifest["rows"]
        if row["canonical_turn_id"] == assistant["canonical_turn_id"]
    )
    assert pinned["from_audience"] == assistant_audience
    assert pinned["from_attribution_version"] == assistant_version
    assert len(manifest["rows"]) == 2
    result = store.reassign_audience(manifest, dry_run=False)
    assert result["selected"] == 2
    assert result["updated"] == expected_updated
    assert all(
        row["audience_conversation_id"] == TARGET
        and row["audience_attribution_version"] == 1
        for row in _canonical(store).values()
    )
    assert _ledger(store) == ledger
    assert _ingest(reconciler, audience=TARGET).turns_written == 0


def test_live_receipts_are_immutable_and_follow_canonical_delete_cascade(stack):
    store, _ = stack
    store.reassign_audience(_plan(store), dry_run=False)
    conn = store._get_conn()
    before = _dump(store)
    for statement in (
        "UPDATE canonical_audience_reassignments SET to_audience = 'unverified'",
        "DELETE FROM canonical_audience_reassignments",
        "UPDATE audience_reassignment_operations SET to_audience = 'unverified'",
        "DELETE FROM audience_reassignment_operations",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(statement)
        conn.rollback()
    assert _dump(store) == before
    conn.execute("DELETE FROM canonical_turns WHERE conversation_id = ?", (OWNER,))
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_audience_reassignments"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_message_sources"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM audience_reassignment_operations"
    ).fetchone()[0] == 1


@pytest.mark.parametrize("failure", ["caller-rollback", "second-row-error"])
def test_reassignment_audit_and_scope_changes_roll_back_together(stack, failure):
    store, _ = stack
    manifest = _plan(store)
    conn = store._get_conn()
    if failure == "second-row-error":
        ordered = sorted(_canonical(store))
        conn.execute(
            "CREATE TEMP TRIGGER reject_second_audience_change "
            "BEFORE UPDATE OF audience_conversation_id ON canonical_turns "
            f"WHEN OLD.canonical_turn_id = '{ordered[1]}' "
            "BEGIN SELECT RAISE(ABORT, 'synthetic second-row failure'); END"
        )
        conn.commit()
    before = _dump(store)
    if failure == "caller-rollback":
        conn.execute("BEGIN IMMEDIATE")
        assert store.reassign_audience(manifest, dry_run=False)["updated"] == 2
        assert all(
            row["audience_conversation_id"] == TARGET
            for row in _canonical(store).values()
        )
        conn.rollback()
    else:
        with pytest.raises(sqlite3.IntegrityError, match="synthetic second-row failure"):
            store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_audience_reassignments"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM audience_reassignment_operations"
    ).fetchone()[0] == 0


@pytest.mark.parametrize("work", ["raw", "ingestion", "compaction", "merge"])
def test_active_work_refuses_planning_and_previously_planned_apply(stack, work):
    store, _ = stack
    manifest = _plan(store)
    if work == "raw":
        store._get_conn().execute(
            "UPDATE conversations SET pending_raw_payload_entries = 1 "
            "WHERE conversation_id = ?", (OWNER,),
        )
        store._get_conn().commit()
    elif work == "ingestion":
        store.upsert_ingestion_episode(
            conversation_id=TARGET, lifecycle_epoch=1,
            worker_id="synthetic-worker", raw_payload_entries=1,
        )
    elif work == "compaction":
        store.start_compaction_operation(
            conversation_id=ORIGINAL, lifecycle_epoch=1,
            worker_id="synthetic-worker", phase_count=1,
            phase_name="synthetic-phase", operation_id="synthetic-compaction-070",
        )
    else:
        reservation = store.try_reserve_merge_audit_in_progress(
            merge_id="synthetic-merge-reservation-070", tenant_id=TENANT,
            source_conversation_id=OWNER, target_conversation_id=OTHER,
            source_label_at_merge="synthetic-source",
        )
        assert reservation.status == "reserved"
    before = _dump(store)
    with pytest.raises(REFUSAL):
        _plan(store)
    with pytest.raises(REFUSAL):
        store.reassign_audience(manifest, dry_run=False)
    assert _dump(store) == before


def test_ordinary_owner_merge_preserves_reassigned_exact_source_replay(stack):
    store, reconciler = stack
    store.reassign_audience(_plan(store), dry_run=False)
    ledger = _ledger_evidence(store)
    canonical_ids = set(_canonical(store))
    destination = "opaque-later-owner-070"
    store.activate_conversation(destination)
    store.upsert_conversation(tenant_id=TENANT, conversation_id=destination)
    store._get_conn().execute(
        "UPDATE conversations SET phase = 'active' WHERE conversation_id = ?",
        (destination,),
    )
    store._get_conn().commit()
    merge_id = "synthetic-later-owner-merge-070"
    reservation = store.try_reserve_merge_audit_in_progress(
        merge_id=merge_id, tenant_id=TENANT,
        source_conversation_id=OWNER, target_conversation_id=destination,
        source_label_at_merge="synthetic-source",
    )
    assert reservation.status == "reserved"
    store.merge_conversation_data(
        merge_id=merge_id, tenant_id=TENANT,
        source_conversation_id=OWNER, target_conversation_id=destination,
        sort_key_offset=1000.0, request_turn_offset=10,
        expected_target_lifecycle_epoch=1, source_label_at_merge="synthetic-source",
    )
    assert set(_canonical(store)) == canonical_ids
    assert all(
        row["conversation_id"] == destination
        and row["audience_conversation_id"] == TARGET
        for row in _canonical(store).values()
    )
    assert _ingest(reconciler, owner=destination, audience=ORIGINAL).turns_written == 0
    assert _ingest(reconciler, owner=destination, audience=TARGET).turns_written == 0
    with pytest.raises(CanonicalSourceConflict):
        _ingest(reconciler, owner=destination, audience=OTHER)
    assert _ledger_evidence(store) == ledger
    assert set(_canonical(store)) == canonical_ids
