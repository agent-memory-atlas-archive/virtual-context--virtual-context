"""BUG-070: real database parity for audited audience reassignment.

Destructive setup is permitted only in an explicitly disposable database.
Fixtures reuse only synthetic transcript and source identifiers.
"""
from __future__ import annotations

import copy

import psycopg
import pytest

from tests.pg_helpers import pg_dsn
from tests.test_audience_reassignment import (
    ORIGINAL, OTHER, OWNER, TARGET, TENANT, _ingest, _seed_durable_card,
    _seed_historical_unattested_transfer, _resign_manifest,
)
from virtual_context.config import VirtualContextConfig
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.postgres import PostgresStore
from virtual_context.types import StorageConfig, TagGeneratorConfig


def _disposable_dsn():
    dsn = pg_dsn()
    if not dsn:
        return None
    name = psycopg.conninfo.conninfo_to_dict(dsn).get("dbname", "")
    return dsn if name.startswith("vc_disposable_") else None


pytestmark = [
    pytest.mark.regression("BUG-070"),
    pytest.mark.skipif(not _disposable_dsn(), reason="requires vc_disposable_* database"),
]


@pytest.fixture
def stack():
    dsn = _disposable_dsn()
    assert dsn, "refusing a non-disposable database"
    store = PostgresStore(dsn)
    with store.pool.connection() as conn:
        conn.execute("TRUNCATE canonical_turns, conversations, conversation_lifecycle CASCADE")
        conn.execute("DELETE FROM audience_reassignment_operations")
    for conversation_id in (OWNER, ORIGINAL, TARGET, OTHER):
        store.activate_conversation(conversation_id)
        store.upsert_conversation(tenant_id=TENANT, conversation_id=conversation_id)
    for alias in (ORIGINAL, TARGET, OTHER):
        store.save_conversation_alias(alias, OWNER)
    with store.pool.connection() as conn:
        conn.execute("UPDATE conversations SET phase = 'active' WHERE conversation_id = %s", (OWNER,))
    config = VirtualContextConfig(
        conversation_id=OWNER,
        storage=StorageConfig(backend="postgres", postgres_dsn=dsn),
        tag_generator=TagGeneratorConfig(type="keyword"),
    )
    semantic = SemanticSearchManager(store=store, config=config)
    semantic._embed_fn = None
    reconciler = IngestReconciler(store=store, semantic=semantic)
    _ingest(reconciler)
    yield store, reconciler
    store.close()


def _plan(store):
    return store.plan_audience_reassignment(
        OWNER, ORIGINAL, TARGET, tenant_id=TENANT,
        expected_lifecycle_epoch=1, operation_id="synthetic-pg-reassignment-070",
    )


def _snapshot(store):
    with store.pool.connection() as conn:
        rows = conn.execute("SELECT * FROM canonical_turns ORDER BY canonical_turn_id").fetchall()
        ledger = conn.execute("SELECT * FROM canonical_message_sources ORDER BY message_id").fetchall()
    return rows, ledger


def test_pg_reassignment_preserves_receipts_and_exact_replay(stack):
    store, reconciler = stack
    before, ledger = _snapshot(store)
    manifest = _plan(store)
    assert store.reassign_audience(manifest)["updated"] == 0
    assert _snapshot(store) == (before, ledger)
    result = store.reassign_audience(manifest, dry_run=False)
    assert result["selected"] == result["updated"] == 2
    assert result["remaining_source_rows"] == 0
    after, new_ledger = _snapshot(store)
    assert ledger == new_ledger
    for old, new in zip(before, after):
        assert new["audience_conversation_id"] == TARGET
        for name, value in old.items():
            if name not in {"audience_conversation_id", "updated_at"}:
                assert new[name] == value, name
    for audience in (ORIGINAL, TARGET):
        assert _ingest(reconciler, audience=audience).turns_written == 0
    with pytest.raises(CanonicalSourceConflict):
        _ingest(reconciler, audience=OTHER)
    assert store.reassign_audience(manifest, dry_run=False)["updated"] == 0
    assert _plan(store) == manifest
    assert len(_snapshot(store)[0]) == 2


def test_pg_reassignment_guards_both_rows_and_audit_deletion(stack):
    store, _ = stack
    store.reassign_audience(_plan(store), dry_run=False)
    rows, _ = _snapshot(store)
    for row in rows:
        for field, value in (("audience_conversation_id", OTHER), ("turn_hash", "changed")):
            with store.pool.connection() as conn, pytest.raises(psycopg.IntegrityError):
                conn.execute(
                    f"UPDATE canonical_turns SET {field} = %s WHERE canonical_turn_id = %s",
                    (value, row["canonical_turn_id"]),
                )
    with store.pool.connection() as conn:
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute("DELETE FROM canonical_audience_reassignments")
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute("UPDATE canonical_audience_reassignments SET to_audience = 'forged'")
        conn.execute("DELETE FROM canonical_turns")
        assert conn.execute("SELECT count(*) AS n FROM canonical_audience_reassignments").fetchone()["n"] == 0


def test_pg_stale_manifest_and_epoch_write_nothing(stack):
    store, reconciler = stack
    manifest = _plan(store)
    _ingest(reconciler, message_id="4000000000000000071", body="Another synthetic source.")
    before = _snapshot(store)
    with pytest.raises(ValueError):
        store.reassign_audience(manifest, dry_run=False)
    manifest = _plan(store)
    stale = copy.deepcopy(manifest)
    stale["expected_lifecycle_epoch"] += 1
    with pytest.raises(ValueError):
        store.reassign_audience(stale, dry_run=False)
    assert _snapshot(store) == before
    with store.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM canonical_audience_reassignments").fetchone()["n"] == 0


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_pg_unattested_selected_row_refuses_entire_plan_and_apply(stack, role):
    store, _ = stack
    manifest = _plan(store)
    store.save_canonical_turn(
        OWNER, 4, "Synthetic unverified source." if role == "user" else "",
        "Synthetic unverified response." if role == "assistant" else "",
        audience_conversation_id=ORIGINAL, audience_attribution_version=1,
    )
    before = _snapshot(store)
    with pytest.raises(ValueError, match="unattested canonical row"):
        _plan(store)
    with pytest.raises(ValueError, match="unattested canonical row"):
        store.reassign_audience(manifest, dry_run=False)
    assert _snapshot(store) == before


@pytest.mark.parametrize("receipt_role", ["user", "assistant"])
def test_pg_historical_unattested_receipt_refuses_adoption_before_pair_commit(stack, receipt_role):
    store, reconciler = stack
    with store.pool.connection() as conn:
        _seed_historical_unattested_transfer(
            conn, dialect="postgres", receipt_role=receipt_role,
        )
    before = _snapshot(store)
    with pytest.raises(CanonicalSourceConflict, match="audience-reassigned legacy row"):
        _ingest(reconciler, audience=TARGET)
    assert _snapshot(store) == before


@pytest.mark.parametrize("trigger", [
    "trg_guard_attested_canonical_turn_update",
    "trg_invalidate_actor_card_turn_source",
])
def test_pg_no_bootstrap_refuses_stale_scope_trigger_body(stack, trigger):
    store, _ = stack
    from virtual_context.storage.audience_reassignment import assert_audience_reassignment_schema

    # Roll back the altered function so each fixture keeps the current schema.
    with store.pool.connection() as conn:
        with pytest.raises(RuntimeError, match="trigger capability"):
            with conn.transaction():
                function = "vc_guard_attested_canonical_turn_update" if "guard" in trigger else "vc_invalidate_actor_card_turn_source"
                conn.execute(f"CREATE OR REPLACE FUNCTION {function}() RETURNS trigger "
                             "AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql")
                assert_audience_reassignment_schema(conn, "postgres")


@pytest.mark.parametrize("change", ["tenant", "alias", "pair"])
def test_pg_recomputed_digest_cannot_bypass_live_scope_or_pair_fences(stack, change):
    store, _ = stack
    manifest = _plan(store)
    if change == "tenant":
        manifest["tenant_id"] = "synthetic-alien-tenant"
        expected = "cross tenants"
    elif change == "alias":
        with store.pool.connection() as conn:
            conn.execute("UPDATE conversation_aliases SET target_id=%s WHERE alias_id=%s",
                         (OTHER, TARGET))
        manifest["operation_id"] = "synthetic-retargeted-alias-070"
        expected = "direct aliases"
    else:
        manifest["rows"] = manifest["rows"][:-1]
        expected = "stale or incomplete"
    _resign_manifest(manifest)
    before = _snapshot(store)
    with pytest.raises(ValueError, match=expected):
        store.reassign_audience(manifest, dry_run=False)
    assert _snapshot(store) == before


def test_pg_scope_change_retains_durable_claims_but_blocks_serving(stack):
    store, _ = stack
    actor = _seed_durable_card(store)
    before = store.list_actor_card_carryovers(TENANT, actor)
    result = store.reassign_audience(_plan(store), dry_run=False)
    assert result["cards_invalidated"] == 1
    assert store.list_actor_card_carryovers(TENANT, actor) == before
    assert store.get_actor_profile(TENANT, actor).card_invalid
    assert store.get_actor_card(
        TENANT, actor, owner_conversation_id=OWNER, audience_conversation_id=TARGET,
    ) is None


@pytest.mark.parametrize("audience", ["", ORIGINAL, TARGET])
def test_pg_exact_pair_promotes_legacy_assistant_attribution(stack, audience):
    store, reconciler = stack
    with store.pool.connection() as conn:
        conn.execute(
            "UPDATE canonical_turns SET audience_conversation_id=%s, "
            "audience_attribution_version=0 WHERE assistant_content<>''", (audience,),
        )
    _, ledger = _snapshot(store)
    result = store.reassign_audience(_plan(store), dry_run=False)
    assert result["updated"] == 2
    for replay_audience in (ORIGINAL, TARGET):
        assert _ingest(reconciler, audience=replay_audience).turns_written == 0
    rows, after_ledger = _snapshot(store)
    assert after_ledger == ledger
    assert all(row["audience_conversation_id"] == TARGET
               and row["audience_attribution_version"] == 1 for row in rows)
