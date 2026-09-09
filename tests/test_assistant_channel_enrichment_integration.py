"""BUG-073: guarded maintenance and immutable card evidence retention."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from tests.test_audience_reassignment import (
    CHANNEL,
    ORIGINAL,
    OWNER,
    TARGET,
    TENANT,
    _seed_durable_card,
    stack as audience_stack,
)
from virtual_context.core.composite_store import CompositeStore

pytestmark = pytest.mark.regression("BUG-073")


@pytest.fixture
def stack(tmp_path):
    yield from audience_stack.__wrapped__(tmp_path)


def _blank_assistant(store):
    conn = store._get_conn()
    row = conn.execute("SELECT * FROM canonical_turns WHERE assistant_content <> ''").fetchone()
    conn.execute(
        "UPDATE canonical_turns SET origin_channel_id = '' WHERE canonical_turn_id = ?",
        (row["canonical_turn_id"],),
    )
    return row["canonical_turn_id"]


def _plan(store, assistant_id, audience=ORIGINAL):
    return store.plan_assistant_channel_enrichment(
        OWNER,
        tenant_id=TENANT,
        audience_conversation_id=audience,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-channel-enrichment",
        assistant_canonical_turn_ids=[assistant_id],
    )


def test_generic_channel_fill_refuses_receipt_bound_row(stack):
    store, _ = stack
    assistant_id = _blank_assistant(store)
    old = store.plan_audience_reassignment(
        OWNER,
        ORIGINAL,
        TARGET,
        tenant_id=TENANT,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-prior-audience",
    )
    store.reassign_audience(old, dry_run=False)
    before = tuple(store._get_conn().iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="requires authorization"):
        store.update_canonical_turn_channels_if_empty(
            OWNER, {assistant_id: (CHANNEL, "")}, expected_lifecycle_epoch=1
        )
    assert tuple(store._get_conn().iterdump()) == before
    manifest = _plan(store, assistant_id, TARGET)
    assert store.enrich_assistant_channels(manifest, dry_run=False)["updated"] == 1
    assert store.reassign_audience(old, dry_run=False)["updated"] == 0
    assert store.enrich_assistant_channels(manifest, dry_run=False)["updated"] == 0


def test_unreceipted_channel_fill_retains_existing_maintenance_behavior(stack):
    store, _ = stack
    assistant_id = _blank_assistant(store)
    assert (
        store.update_canonical_turn_channels_if_empty(
            OWNER, {assistant_id: (CHANNEL, "")}, expected_lifecycle_epoch=1
        )
        == 1
    )
    assert (
        store._get_conn()
        .execute("SELECT COUNT(*) FROM canonical_assistant_channel_enrichments")
        .fetchone()[0]
        == 0
    )


def test_authorized_channel_change_hides_card_but_retains_legacy_claims(stack):
    store, _ = stack
    assistant_id = _blank_assistant(store)
    actor = _seed_durable_card(store)
    conn = store._get_conn()
    # Model a historical durable claim that cites an assistant row. Modern
    # admission need not create it; administrative repair must retain evidence.
    conn.execute(
        "UPDATE actor_card_turn_sources SET canonical_turn_id = ?, audience_channel_id = ? WHERE tenant_id = ?",
        (assistant_id, "", TENANT),
    )
    conn.execute(
        "UPDATE actor_profiles SET card_invalid=0,card_dirty=0,card_build_marker='synthetic-pending' WHERE tenant_id=? AND actor_id=?",
        (TENANT, actor),
    )
    entries = [dict(r) for r in conn.execute("SELECT * FROM actor_card_entries")]
    sources = [dict(r) for r in conn.execute("SELECT * FROM actor_card_turn_sources")]
    manifest = _plan(store, assistant_id)
    assert store.enrich_assistant_channels(manifest, dry_run=False)["updated"] == 1
    assert [dict(r) for r in conn.execute("SELECT * FROM actor_card_entries")] == entries
    assert [dict(r) for r in conn.execute("SELECT * FROM actor_card_turn_sources")] == sources
    profile = conn.execute(
        "SELECT card_dirty,card_invalid,card_build_marker FROM actor_profiles WHERE tenant_id=? AND actor_id=?",
        (TENANT, actor),
    ).fetchone()
    assert tuple(profile) == (1, 1, "")
    assert (
        store.get_actor_card(
            TENANT, actor, owner_conversation_id=OWNER, audience_conversation_id=ORIGINAL
        )
        is None
    )


def test_unauthorized_channel_change_keeps_defensive_card_deletion(stack):
    store, _ = stack
    assistant_id = _blank_assistant(store)
    _seed_durable_card(store)
    conn = store._get_conn()
    conn.execute(
        "UPDATE actor_card_turn_sources SET canonical_turn_id = ? WHERE tenant_id = ?",
        (assistant_id, TENANT),
    )
    assert store.update_canonical_turn_channels_if_empty(OWNER, {assistant_id: (CHANNEL, "")}) == 1
    assert conn.execute("SELECT COUNT(*) FROM actor_card_entries").fetchone()[0] == 0


def test_composite_forwards_exact_optional_administration():
    calls = []
    canonical = SimpleNamespace(
        plan_assistant_channel_enrichment=lambda *a, **k: (
            calls.append(("plan", a, k)) or {"planned": True}
        ),
        enrich_assistant_channels=lambda *a, **k: (
            calls.append(("apply", a, k)) or {"dry_run": k["dry_run"]}
        ),
    )
    other = SimpleNamespace()
    store = CompositeStore(
        segments=canonical, facts=other, fact_links=other, state=other, search=other
    )
    kwargs = dict(
        tenant_id=TENANT,
        audience_conversation_id=TARGET,
        expected_lifecycle_epoch=1,
        operation_id="synthetic",
        assistant_canonical_turn_ids=["opaque-id"],
    )
    assert store.plan_assistant_channel_enrichment(OWNER, **kwargs) == {"planned": True}
    manifest = {"opaque": "manifest"}
    assert store.enrich_assistant_channels(manifest) == {"dry_run": True}
    assert calls == [("plan", (OWNER,), kwargs), ("apply", (manifest,), {"dry_run": True})]
    store._segments = other
    with pytest.raises(NotImplementedError):
        store.plan_assistant_channel_enrichment(OWNER, **kwargs)
    with pytest.raises(NotImplementedError):
        store.enrich_assistant_channels(manifest)


def test_bulk_receipted_ids_are_owner_scoped_read_only_and_chunked(stack):
    from uuid import uuid4

    store, _ = stack
    assistant_id = _blank_assistant(store)
    ids = [row.canonical_turn_id for row in store.get_all_canonical_turns(OWNER)]
    requested = [str(uuid4()) for _ in range(510)] + ids + ids
    assert store.get_receipted_canonical_turn_ids(OWNER, requested) == set()
    store.enrich_assistant_channels(_plan(store, assistant_id), dry_run=False)
    before = tuple(store._get_conn().iterdump())
    assert store.get_receipted_canonical_turn_ids(OWNER, requested) == {assistant_id}
    assert store.get_receipted_canonical_turn_ids("synthetic-other-owner", requested) == set()
    assert store.get_receipted_canonical_turn_ids(OWNER, []) == set()
    assert tuple(store._get_conn().iterdump()) == before
    audience = store.plan_audience_reassignment(
        OWNER,
        ORIGINAL,
        TARGET,
        tenant_id=TENANT,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-later-audience-bulk-read",
    )
    store.reassign_audience(audience, dry_run=False)
    before = tuple(store._get_conn().iterdump())
    assert store.get_receipted_canonical_turn_ids(OWNER, requested) == set(ids)
    assert tuple(store._get_conn().iterdump()) == before


def test_composite_forwards_receipt_lookup_to_canonical_store():
    calls = []
    canonical = SimpleNamespace(
        get_receipted_canonical_turn_ids=lambda owner, ids: calls.append((owner, ids)) or {ids[0]},
    )
    other = SimpleNamespace()
    store = CompositeStore(
        segments=canonical, facts=other, fact_links=other, state=other, search=other
    )
    ids = ["synthetic-canonical-id"]
    assert store.get_receipted_canonical_turn_ids(OWNER, ids) == set(ids)
    assert calls == [(OWNER, ids)]
    store._segments = other
    assert store.get_receipted_canonical_turn_ids(OWNER, ids) == set()
