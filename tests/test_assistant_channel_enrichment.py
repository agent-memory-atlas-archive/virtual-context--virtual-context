"""BUG-073: append-only channel enrichment preserves earlier audience proof."""
from __future__ import annotations

import copy
import sqlite3

import pytest

from test_audience_reassignment import (
    CHANNEL, ORIGINAL, OTHER, OWNER, TARGET, TENANT, _canonical, _dump,
    _ingest, _ledger_evidence, _plan, stack as base_stack,
)
from virtual_context.core.quote_search import find_quote
from virtual_context.types import SpeakerRetrievalContext

pytestmark = pytest.mark.regression("BUG-073")


@pytest.fixture
def stack(tmp_path):
    yield from base_stack.__wrapped__(tmp_path)


@pytest.fixture
def historical(stack):
    store, rec = stack
    conn = store._get_conn()
    assistant = next(row for row in _canonical(store).values() if row["assistant_content"])
    conn.execute(
        "UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?",
        (assistant["canonical_turn_id"],),
    )
    conn.commit()
    audience_manifest = _plan(store)
    store.reassign_audience(audience_manifest, dry_run=False)
    return store, rec, audience_manifest, assistant["canonical_turn_id"]


def _repair():
    from virtual_context.storage import assistant_channel_enrichment
    return assistant_channel_enrichment


def _enrichment_plan(store, assistant_id, **overrides):
    args = dict(
        tenant_id=TENANT, audience_conversation_id=TARGET,
        expected_lifecycle_epoch=1, operation_id="synthetic-channel-enrichment",
        assistant_canonical_turn_ids=[assistant_id], dialect="sqlite",
    )
    args.update(overrides)
    return _repair().plan_assistant_channel_enrichment(store, OWNER, **args)


def _apply(store, manifest, **kwargs):
    return _repair().enrich_assistant_channels(store, manifest, dialect="sqlite", **kwargs)


def _old_receipts(store):
    return [dict(row) for row in store._get_conn().execute(
        "SELECT * FROM canonical_audience_reassignments ORDER BY canonical_turn_id"
    )]


def test_enrichment_preserves_old_manifest_and_restores_assistant_quote(historical):
    store, rec, audience_manifest, assistant_id = historical
    before = _canonical(store)
    receipts = _old_receipts(store)
    sources = _ledger_evidence(store)
    context = SpeakerRetrievalContext(
        tenant_id=TENANT, owner_conversation_id=OWNER,
        audience_conversation_id=TARGET, audience_channel_scope="conversation",
        request_origin_channel_id="1000000000000000099",
    )
    before_quote = find_quote(
        store, rec._semantic, "synthetic source response", conversation_id=OWNER,
        speaker_context=context,
    )
    assert before_quote["found"] is False
    manifest = _enrichment_plan(store, assistant_id)
    assert _apply(store, manifest, dry_run=False)["updated"] == 1
    assert store.reassign_audience(audience_manifest)["already_applied"]
    assert _old_receipts(store) == receipts
    assert _ledger_evidence(store) == sources
    after = _canonical(store)
    assert after[assistant_id]["origin_channel_id"] == CHANNEL
    for row_id, old in before.items():
        for field in old:
            if row_id == assistant_id and field == "origin_channel_id":
                continue
            assert after[row_id][field] == old[field], (row_id, field)
    assert _ingest(rec, audience=ORIGINAL).turns_written == 0
    assert _ingest(rec, audience=TARGET).turns_written == 0
    with pytest.raises(ValueError):
        _enrichment_plan(store, assistant_id, audience_conversation_id=OTHER)
    result = find_quote(
        store, rec._semantic, "synthetic source response", conversation_id=OWNER,
        speaker_context=context,
    )
    assert result["found"] is True
    assert any(row.get("matched_side") == "assistant" for row in result["results"])


def test_plan_default_dry_run_and_replay_are_exact_and_read_only(historical):
    store, _, _, assistant_id = historical
    before = _dump(store)
    manifest = _enrichment_plan(store, assistant_id)
    assert _dump(store) == before
    assert _apply(store, manifest)["would_update"] == 1
    assert _dump(store) == before
    _apply(store, manifest, dry_run=False)
    after = _dump(store)
    assert _enrichment_plan(store, assistant_id) == manifest
    report = _apply(store, manifest, dry_run=False)
    assert report["already_applied"] and report["updated"] == report["would_update"] == 0
    assert _dump(store) == after


def test_enrichment_without_prior_audience_receipt(stack):
    store, _ = stack
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    store._get_conn().execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    store._get_conn().commit()
    manifest = _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    assert all(manifest["rows"][0][name] == "" for name in (
        "prior_audience_operation_id", "prior_audience_manifest_digest", "prior_audience_row_fingerprint",
    ))
    assert _apply(store, manifest, dry_run=False)["updated"] == 1
    _repair().verify_assistant_channel_enrichment(store._get_conn(), assistant_id, dialect="sqlite")


@pytest.mark.parametrize("field,value", [
    ("to_channel_id", "9999999999999999999"),
    ("from_channel_id", CHANNEL),
    ("user_canonical_turn_id", "unrelated-user"),
    ("source_fingerprint", "a" * 64),
    ("user_row_fingerprint", "b" * 64),
    ("before_row_fingerprint", "c" * 64),
    ("after_row_fingerprint", "d" * 64),
    ("prior_audience_operation_id", ""),
    ("prior_audience_manifest_digest", ""),
    ("prior_audience_row_fingerprint", ""),
])
def test_resigned_malicious_manifest_fails_without_writes(historical, field, value):
    store, _, _, assistant_id = historical
    manifest = _enrichment_plan(store, assistant_id)
    manifest["rows"][0][field] = value
    manifest["manifest_digest"] = _repair()._digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    before = _dump(store)
    with pytest.raises(ValueError):
        _apply(store, manifest, dry_run=False)
    assert _dump(store) == before


@pytest.mark.parametrize("overrides", [
    {"tenant_id": "unrelated-tenant"},
    {"audience_conversation_id": OTHER},
    {"expected_lifecycle_epoch": 2},
    {"expected_lifecycle_epoch": True},
    {"assistant_canonical_turn_ids": []},
])
def test_wrong_authority_or_empty_selection_refuses_without_writes(historical, overrides):
    store, _, _, assistant_id = historical
    before = _dump(store)
    with pytest.raises(ValueError):
        _enrichment_plan(store, assistant_id, **overrides)
    assert _dump(store) == before


def test_failed_update_rolls_back_operation_and_receipts(historical):
    store, _, _, assistant_id = historical
    manifest = _enrichment_plan(store, assistant_id)
    conn = store._get_conn()
    conn.execute("""CREATE TRIGGER synthetic_enrichment_write_failure
        BEFORE UPDATE OF origin_channel_id ON canonical_turns
        BEGIN SELECT RAISE(ABORT, 'synthetic repair failure'); END""")
    conn.commit()
    before = _dump(store)
    with pytest.raises(sqlite3.IntegrityError, match="synthetic repair failure"):
        _apply(store, manifest, dry_run=False)
    assert _dump(store) == before


@pytest.mark.parametrize("table", [
    "assistant_channel_enrichment_operations", "canonical_assistant_channel_enrichments",
])
def test_authorizations_are_immutable_and_live_deletes_refuse(historical, table):
    store, _, _, assistant_id = historical
    _apply(store, _enrichment_plan(store, assistant_id), dry_run=False)
    conn = store._get_conn()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(f"UPDATE {table} SET manifest_digest='changed'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(f"DELETE FROM {table}")
    conn.rollback()


def test_new_operation_cannot_reauthorize_enriched_row(historical):
    store, _, _, assistant_id = historical
    _apply(store, _enrichment_plan(store, assistant_id), dry_run=False)
    before = _dump(store)
    with pytest.raises(ValueError, match="already has"):
        _enrichment_plan(store, assistant_id, operation_id="another-operation")
    assert _dump(store) == before


def test_duplicate_selection_and_nonboolean_apply_are_refused(historical):
    store, _, _, assistant_id = historical
    with pytest.raises(ValueError, match="duplicate"):
        _enrichment_plan(store, assistant_id, assistant_canonical_turn_ids=[assistant_id, assistant_id])
    manifest = _enrichment_plan(store, assistant_id)
    before = _dump(store)
    with pytest.raises(ValueError, match="boolean"):
        _apply(store, manifest, dry_run="false")
    changed = copy.deepcopy(manifest)
    changed["version"] = True
    with pytest.raises(ValueError, match="version"):
        _apply(store, changed)
    assert _dump(store) == before


def test_half_applied_receipt_cannot_use_original_audience_fingerprint(historical):
    from virtual_context.storage.channel_enrichment_guards import ensure_receipted_channel_guard

    store, _, audience_manifest, assistant_id = historical
    _apply(store, _enrichment_plan(store, assistant_id), dry_run=False)
    conn = store._get_conn()
    # Simulate recovery of a corrupt half-applied backup. The original body
    # fingerprint still matches the earlier audience receipt, but a newer
    # authorization exists and must be verified rather than ignored.
    conn.execute("DROP TRIGGER trg_guard_receipted_assistant_channel_update")
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    ensure_receipted_channel_guard(conn, "sqlite")
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="changed source evidence"):
        store.reassign_audience(audience_manifest)
    with pytest.raises(ValueError, match="changed source evidence"):
        _repair().verify_assistant_channel_enrichment(conn, assistant_id, dialect="sqlite")
    assert _dump(store) == before


@pytest.mark.parametrize("name", [
    "trg_guard_receipted_assistant_channel_update",
    "trg_invalidate_actor_card_turn_source_update",
])
@pytest.mark.parametrize("stale", [False, True])
def test_missing_or_stale_guards_refuse_plan_and_apply_without_migration(historical, name, stale):
    from virtual_context.storage.channel_enrichment_guards import authorized_channel_transition_sql

    store, _, _, assistant_id = historical
    manifest = _enrichment_plan(store, assistant_id)
    conn = store._get_conn()
    original = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()["sql"]
    conn.execute(f"DROP TRIGGER {name}")
    if stale:
        if name == "trg_guard_receipted_assistant_channel_update":
            changed = original.replace("SELECT RAISE(ABORT, 'receipted channel enrichment requires authorization')", "SELECT 1")
        else:
            changed = original.replace(authorized_channel_transition_sql("sqlite"), "0")
        assert changed != original
        conn.execute(changed)
    conn.commit()
    before = _dump(store)
    with pytest.raises(RuntimeError, match="guard|trigger"):
        _enrichment_plan(store, assistant_id)
    with pytest.raises(RuntimeError, match="guard|trigger"):
        _apply(store, manifest, dry_run=False)
    assert _dump(store) == before


def _two_historical_pairs(stack):
    store, rec = stack
    _ingest(rec, message_id="4000000000000000088", body="Another synthetic source observation.")
    conn = store._get_conn()
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE assistant_content<>''")
    conn.commit()
    store.reassign_audience(_plan(store), dry_run=False)
    assistant_ids = sorted(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    assert len(assistant_ids) == 2
    return store, assistant_ids


def test_prior_operation_checks_unselected_pair_before_apply(stack):
    store, ids = _two_historical_pairs(stack)
    manifest = _enrichment_plan(store, ids[0])
    conn = store._get_conn()
    conn.execute("UPDATE canonical_turns SET sender='Changed synthetic sender' WHERE canonical_turn_id=?", (ids[1],))
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="changed source evidence"):
        _enrichment_plan(store, ids[0])
    with pytest.raises(ValueError, match="changed source evidence"):
        _apply(store, manifest, dry_run=False)
    assert _dump(store) == before


def test_prior_operation_checked_after_apply_and_transaction_rolls_back(stack):
    store, ids = _two_historical_pairs(stack)
    manifest = _enrichment_plan(store, ids[0])
    conn = store._get_conn()
    conn.execute(f"""CREATE TRIGGER synthetic_unselected_pair_corruption
        AFTER UPDATE OF origin_channel_id ON canonical_turns
        BEGIN UPDATE canonical_turns SET sender='Changed synthetic sender'
              WHERE canonical_turn_id='{ids[1]}'; END""")
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="changed source evidence"):
        _apply(store, manifest, dry_run=False)
    assert _dump(store) == before


@pytest.mark.parametrize("field,value", [
    ("sender", " "), ("sender_actor_id", " "), ("source_message_id", " "),
    ("reply_target_message_id", " "), ("reply_subject_actor_id", " "),
    ("reply_subject_label", " "), ("reply_target_body", " "),
    ("reply_attribution_version", 1), ("turn_group_number", -1),
])
def test_nonblank_assistant_identity_or_invalid_group_refuses(stack, field, value):
    store, _ = stack
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn = store._get_conn()
    conn.execute(f"UPDATE canonical_turns SET origin_channel_id='',{field}=? WHERE canonical_turn_id=?", (value, assistant_id))
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="identity|attribution|projection"):
        _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    assert _dump(store) == before


def test_original_source_ordinal_is_not_current_pair_identity(stack):
    store, _ = stack
    conn = store._get_conn()
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    conn.commit()
    source = dict(conn.execute("SELECT * FROM canonical_message_sources").fetchone())
    # Exact membership IDs/hashes survive canonical ordinal assignment and
    # owner merges; the immutable admission ordinal may still be its default.
    assert source["turn_group_number"] == -1
    assert all(row["turn_group_number"] >= 0 for row in _canonical(store).values())
    manifest = _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    assert _apply(store, manifest, dry_run=False)["updated"] == 1
    assert dict(conn.execute("SELECT * FROM canonical_message_sources").fetchone()) == source


def test_channel_enrichment_then_audience_reassignment_preserves_both_replays(stack):
    store, rec = stack
    conn = store._get_conn()
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    conn.commit()
    channel_plan = _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    _apply(store, channel_plan, dry_run=False)
    before_receipts = [dict(row) for row in conn.execute("SELECT * FROM canonical_assistant_channel_enrichments")]
    source = _ledger_evidence(store)

    audience_plan = _plan(store)
    assert store.reassign_audience(audience_plan)["would_update"] == 2
    assert store.reassign_audience(audience_plan, dry_run=False)["updated"] == 2
    assert [dict(row) for row in conn.execute("SELECT * FROM canonical_assistant_channel_enrichments")] == before_receipts
    assert _ledger_evidence(store) == source
    assert _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL) == channel_plan
    assert _apply(store, channel_plan, dry_run=False)["already_applied"]
    assert store.reassign_audience(audience_plan, dry_run=False)["already_applied"]
    _repair().verify_assistant_channel_enrichment(conn, assistant_id, dialect="sqlite")
    assert _ingest(rec, audience=ORIGINAL).turns_written == 0
    assert _ingest(rec, audience=TARGET).turns_written == 0


def test_unproved_assistant_audience_requires_separate_attribution(stack):
    store, _ = stack
    conn = store._get_conn()
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn.execute("UPDATE canonical_turns SET origin_channel_id='',audience_attribution_version=0 WHERE canonical_turn_id=?", (assistant_id,))
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="unproved audience attribution.*separate"):
        _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    assert _dump(store) == before


def test_successor_audience_must_start_from_proved_attribution(stack):
    store, _ = stack
    conn = store._get_conn()
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    conn.commit()
    _apply(store, _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL), dry_run=False)
    store.reassign_audience(_plan(store), dry_run=False)
    name = "trg_audience_reassignment_update"
    original = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()["sql"]
    conn.execute(f"DROP TRIGGER {name}")
    conn.execute("UPDATE canonical_audience_reassignments SET from_attribution_version=0 WHERE canonical_turn_id=?", (assistant_id,))
    conn.execute(original)
    conn.commit()
    with pytest.raises(ValueError, match="successor audience"):
        _repair().verify_assistant_channel_enrichment(conn, assistant_id, dialect="sqlite")


def test_unreachable_card_authorization_text_is_not_guard_capability(historical):
    store, _, _, assistant_id = historical
    conn = store._get_conn()
    name = "trg_invalidate_actor_card_turn_source_update"
    original = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()["sql"]
    conn.execute(f"DROP TRIGGER {name}")
    conn.execute(f"CREATE TRIGGER {name} BEFORE UPDATE ON canonical_turns BEGIN SELECT 1; /* {original} */ END")
    conn.commit()
    before = _dump(store)
    with pytest.raises(RuntimeError, match="card.*guard"):
        _enrichment_plan(store, assistant_id)
    assert _dump(store) == before


def test_successor_audience_lifecycle_is_fenced_during_old_channel_replay(stack):
    store, _ = stack
    conn = store._get_conn()
    assistant_id = next(row["canonical_turn_id"] for row in _canonical(store).values() if row["assistant_content"])
    conn.execute("UPDATE canonical_turns SET origin_channel_id='' WHERE canonical_turn_id=?", (assistant_id,))
    conn.commit()
    channel_plan = _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    _apply(store, channel_plan, dry_run=False)
    store.reassign_audience(_plan(store), dry_run=False)
    conn.execute("UPDATE conversation_lifecycle SET deleted=1 WHERE conversation_id=?", (TARGET,))
    conn.commit()
    before = _dump(store)
    with pytest.raises(ValueError, match="lifecycle is deleted"):
        _enrichment_plan(store, assistant_id, audience_conversation_id=ORIGINAL)
    with pytest.raises(ValueError, match="lifecycle is deleted"):
        _apply(store, channel_plan)
    assert _dump(store) == before
