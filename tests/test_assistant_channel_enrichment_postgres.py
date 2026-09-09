"""BUG-073: real disposable PostgreSQL channel enrichment parity."""

from __future__ import annotations

import psycopg
import pytest

from tests.test_audience_reassignment import (
    CHANNEL,
    ORIGINAL,
    OTHER,
    OWNER,
    TARGET,
    TENANT,
    _ingest,
    _seed_durable_card,
)
from tests.test_audience_reassignment_postgres import _disposable_dsn, stack as audience_stack
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.quote_search import find_quote
from virtual_context.storage.actor_card_transition_guards import (
    postgres_actor_card_turn_source_function_body,
)
from virtual_context.storage.assistant_channel_enrichment import (
    assert_assistant_channel_enrichment_schema,
)
from virtual_context.storage.channel_enrichment_guards import receipted_channel_guard_function_body
from virtual_context.types import SpeakerRetrievalContext

pytestmark = [
    pytest.mark.regression("BUG-073"),
    pytest.mark.skipif(not _disposable_dsn(), reason="requires vc_disposable_* database"),
]


@pytest.fixture
def stack():
    generator = audience_stack.__wrapped__()
    store, reconciler = next(generator)
    try:
        with store.pool.connection() as conn:
            conn.execute("DELETE FROM assistant_channel_enrichment_operations")
        yield store, reconciler
    finally:
        generator.close()


def _blank(store):
    with store.pool.connection() as conn:
        rows = conn.execute(
            "SELECT canonical_turn_id FROM canonical_turns WHERE assistant_content <> '' ORDER BY canonical_turn_id"
        ).fetchall()
        conn.execute(
            "UPDATE canonical_turns SET origin_channel_id='' WHERE assistant_content <> ''"
        )
    return [str(row["canonical_turn_id"]) for row in rows]


def _plan(store, ids, audience=ORIGINAL):
    return store.plan_assistant_channel_enrichment(
        OWNER,
        tenant_id=TENANT,
        audience_conversation_id=audience,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-pg-channel-enrichment",
        assistant_canonical_turn_ids=ids,
    )


def _snapshot(store):
    with store.pool.connection() as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in (
                "canonical_turns",
                "canonical_message_sources",
                "canonical_audience_reassignments",
                "assistant_channel_enrichment_operations",
                "canonical_assistant_channel_enrichments",
            )
        }


def test_pg_old_manifest_exact_source_and_assistant_retrieval_survive_enrichment(stack):
    store, reconciler = stack
    ids = _blank(store)
    prior = store.plan_audience_reassignment(
        OWNER,
        ORIGINAL,
        TARGET,
        tenant_id=TENANT,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-prior-pg-audience",
    )
    store.reassign_audience(prior, dry_run=False)
    before = _snapshot(store)
    with pytest.raises(psycopg.IntegrityError, match="requires authorization"):
        store.update_canonical_turn_channels_if_empty(
            OWNER, {ids[0]: (CHANNEL, "")}, expected_lifecycle_epoch=1
        )
    assert _snapshot(store) == before
    manifest = _plan(store, ids, TARGET)
    assert store.enrich_assistant_channels(manifest)["updated"] == 0
    assert _snapshot(store) == before
    assert store.enrich_assistant_channels(manifest, dry_run=False)["updated"] == 1
    after = _snapshot(store)
    assert after["canonical_message_sources"] == before["canonical_message_sources"]
    assert after["canonical_audience_reassignments"] == before["canonical_audience_reassignments"]
    for old, new in zip(before["canonical_turns"], after["canonical_turns"]):
        for field in old:
            if str(old["canonical_turn_id"]) in ids and field == "origin_channel_id":
                continue
            assert old[field] == new[field], field
    assert store.reassign_audience(prior, dry_run=False)["already_applied"] is True
    assert store.enrich_assistant_channels(manifest, dry_run=False)["already_applied"] is True
    for audience in (ORIGINAL, TARGET):
        assert _ingest(reconciler, audience=audience).turns_written == 0
    with pytest.raises(CanonicalSourceConflict):
        _ingest(reconciler, audience=OTHER)
    context = SpeakerRetrievalContext(
        tenant_id=TENANT,
        owner_conversation_id=OWNER,
        audience_conversation_id=TARGET,
        audience_channel_scope="conversation",
        request_origin_channel_id="1000000000000000099",
    )
    result = find_quote(
        store,
        reconciler._semantic,
        "synthetic source response",
        conversation_id=OWNER,
        speaker_context=context,
    )
    assert any(row.get("matched_side") == "assistant" for row in result["results"])


def test_pg_authorized_repair_keeps_claims_but_hides_card(stack):
    store, _ = stack
    ids = _blank(store)
    actor = _seed_durable_card(store)
    with store.pool.connection() as conn:
        conn.execute(
            "UPDATE actor_card_turn_sources SET canonical_turn_id=%s,audience_channel_id=%s WHERE tenant_id=%s",
            (ids[0], "", TENANT),
        )
        conn.execute(
            "UPDATE actor_profiles SET card_invalid=0,card_dirty=0,card_build_marker='synthetic-pending' WHERE tenant_id=%s AND actor_id=%s",
            (TENANT, actor),
        )
        before_entries = conn.execute("SELECT * FROM actor_card_entries ORDER BY id").fetchall()
        before_sources = conn.execute(
            "SELECT * FROM actor_card_turn_sources ORDER BY entry_id"
        ).fetchall()
    assert store.enrich_assistant_channels(_plan(store, ids), dry_run=False)["updated"] == 1
    with store.pool.connection() as conn:
        assert (
            conn.execute("SELECT * FROM actor_card_entries ORDER BY id").fetchall()
            == before_entries
        )
        assert (
            conn.execute("SELECT * FROM actor_card_turn_sources ORDER BY entry_id").fetchall()
            == before_sources
        )
        row = conn.execute(
            "SELECT card_dirty,card_invalid,card_build_marker FROM actor_profiles WHERE tenant_id=%s AND actor_id=%s",
            (TENANT, actor),
        ).fetchone()
        assert row == {"card_dirty": 1, "card_invalid": 1, "card_build_marker": ""}
    assert (
        store.get_actor_card(
            TENANT, actor, owner_conversation_id=OWNER, audience_conversation_id=ORIGINAL
        )
        is None
    )


def test_pg_receipts_immutable_and_canonical_deletion_cascades(stack):
    store, _ = stack
    ids = _blank(store)
    store.enrich_assistant_channels(_plan(store, ids), dry_run=False)
    for table in (
        "assistant_channel_enrichment_operations",
        "canonical_assistant_channel_enrichments",
    ):
        for sql in (f"UPDATE {table} SET manifest_digest='forged'", f"DELETE FROM {table}"):
            with store.pool.connection() as conn, pytest.raises(psycopg.errors.RaiseException):
                conn.execute(sql)
    with store.pool.connection() as conn:
        conn.execute("DELETE FROM canonical_turns WHERE conversation_id=%s", (OWNER,))
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM canonical_assistant_channel_enrichments"
            ).fetchone()["n"]
            == 0
        )


def test_pg_second_update_failure_rolls_back_channels_and_receipts(stack):
    store, reconciler = stack
    _ingest(reconciler, message_id="4000000000000000073", body="Another synthetic channel source.")
    ids = _blank(store)
    manifest = _plan(store, ids)
    assert len(ids) == 2
    before = _snapshot(store)
    with store.pool.connection() as conn:
        conn.execute(
            "CREATE FUNCTION synthetic_fail_second_channel() RETURNS trigger AS $$ BEGIN IF OLD.canonical_turn_id=%s::uuid THEN RAISE EXCEPTION 'synthetic second update'; END IF; RETURN NEW; END; $$ LANGUAGE plpgsql".replace(
                "%s", "'" + ids[1] + "'"
            )
        )
        conn.execute(
            "CREATE TRIGGER synthetic_fail_second_channel BEFORE UPDATE OF origin_channel_id ON canonical_turns FOR EACH ROW EXECUTE FUNCTION synthetic_fail_second_channel()"
        )
    try:
        with pytest.raises(psycopg.errors.RaiseException, match="synthetic second update"):
            store.enrich_assistant_channels(manifest, dry_run=False)
        assert _snapshot(store) == before
    finally:
        with store.pool.connection() as conn:
            conn.execute("DROP TRIGGER synthetic_fail_second_channel ON canonical_turns")
            conn.execute("DROP FUNCTION synthetic_fail_second_channel()")


@pytest.mark.parametrize(
    "function,unreachable_guard",
    [
        ("vc_invalidate_actor_card_turn_source", False),
        ("vc_invalidate_actor_card_turn_source", True),
        ("vc_guard_receipted_assistant_channel_update", False),
        ("vc_guard_receipted_assistant_channel_update", True),
    ],
)
def test_pg_stale_guard_body_refuses_without_schema_rewrite(stack, function, unreachable_guard):
    store, _ = stack
    body = "BEGIN RETURN NEW; END;"
    if unreachable_guard:
        # A substring check must not mistake an unreachable, otherwise exact
        # authorization predicate for an operative refusal guard.
        expected_body = (
            postgres_actor_card_turn_source_function_body()
            if function == "vc_invalidate_actor_card_turn_source"
            else receipted_channel_guard_function_body()
        )
        body = f"BEGIN RETURN NEW; IF FALSE THEN {expected_body} END IF; END;"
    with store.pool.connection() as conn:
        with pytest.raises(RuntimeError):
            with conn.transaction():
                conn.execute(
                    f"CREATE OR REPLACE FUNCTION {function}() RETURNS trigger AS $$ {body} $$ LANGUAGE plpgsql"
                )
                assert_assistant_channel_enrichment_schema(conn, "postgres")


def test_pg_unreceipted_maintenance_channel_fill_remains_available(stack):
    store, _ = stack
    ids = _blank(store)
    assert (
        store.update_canonical_turn_channels_if_empty(
            OWNER, {ids[0]: (CHANNEL, "")}, expected_lifecycle_epoch=1
        )
        == 1
    )
    with store.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM canonical_assistant_channel_enrichments"
            ).fetchone()["n"]
            == 0
        )


def test_pg_channel_enrichment_then_audience_reassignment_composes(stack):
    store, reconciler = stack
    ids = _blank(store)
    channel_manifest = _plan(store, ids)
    assert store.enrich_assistant_channels(channel_manifest, dry_run=False)["updated"] == 1
    with store.pool.connection() as conn:
        channel_receipt = conn.execute(
            "SELECT * FROM canonical_assistant_channel_enrichments"
        ).fetchall()
    audience_manifest = store.plan_audience_reassignment(
        OWNER,
        ORIGINAL,
        TARGET,
        tenant_id=TENANT,
        expected_lifecycle_epoch=1,
        operation_id="synthetic-later-pg-audience",
    )
    assert store.reassign_audience(audience_manifest, dry_run=False)["updated"] == 2
    with store.pool.connection() as conn:
        assert (
            conn.execute("SELECT * FROM canonical_assistant_channel_enrichments").fetchall()
            == channel_receipt
        )
    assert (
        store.enrich_assistant_channels(channel_manifest, dry_run=False)["already_applied"] is True
    )
    assert store.reassign_audience(audience_manifest, dry_run=False)["already_applied"] is True
    for audience in (ORIGINAL, TARGET):
        assert _ingest(reconciler, audience=audience).turns_written == 0
    with pytest.raises(CanonicalSourceConflict):
        _ingest(reconciler, audience=OTHER)


@pytest.mark.parametrize("mode", ["audience", "enrichment", "both"])
def test_pg_channel_backfill_skips_receipts_before_limit_and_preserves_evidence(stack, mode):
    import json
    from uuid import uuid4
    from virtual_context.core.composite_store import CompositeStore
    from virtual_context.engine import VirtualContextEngine

    store, _ = stack
    assistant_id = _blank(store)[0]
    if mode in {"audience", "both"}:
        manifest = store.plan_audience_reassignment(
            OWNER,
            ORIGINAL,
            TARGET,
            tenant_id=TENANT,
            expected_lifecycle_epoch=1,
            operation_id="synthetic-backfill-pg-audience",
        )
        store.reassign_audience(manifest, dry_run=False)
    if mode in {"enrichment", "both"}:
        store.enrich_assistant_channels(
            _plan(store, [assistant_id], TARGET if mode == "both" else ORIGINAL), dry_run=False
        )
    later_id = str(uuid4())
    body = "A later synthetic eligible message."
    envelope = (
        "Conversation info (guild):\n```json\n"
        + json.dumps(
            {"chat_id": "channel:7000000000000000001", "group_channel": "#other-synthetic"}
        )
        + "\n```\n"
        + body
    )
    store.save_canonical_turn(
        OWNER,
        -1,
        body,
        "",
        canonical_turn_id=later_id,
        sort_key=100000.0,
        user_raw_content=json.dumps([{"type": "text", "text": envelope}]),
    )
    engine = VirtualContextEngine.__new__(VirtualContextEngine)
    engine._store = CompositeStore(
        segments=store, facts=store, fact_links=store, state=store, search=store
    )
    with store.pool.connection() as conn:
        before = conn.execute(
            "SELECT * FROM canonical_turns WHERE canonical_turn_id=%s", (assistant_id,)
        ).fetchone()
    snapshot = _snapshot(store)
    preview = engine.backfill_channels(OWNER, dry_run=True, limit=1)
    assert _snapshot(store) == snapshot
    result = engine.backfill_channels(OWNER, limit=1)
    assert preview["updated"] == result["updated"] == 1
    expected_protected = 2 if mode in {"audience", "both"} else 1
    assert preview["skipped_receipted"] == result["skipped_receipted"] == expected_protected
    after = _snapshot(store)
    for table in snapshot:
        if table == "canonical_turns":
            assert [row for row in after[table] if str(row["canonical_turn_id"]) != later_id] == [
                row for row in snapshot[table] if str(row["canonical_turn_id"]) != later_id
            ]
        else:
            assert after[table] == snapshot[table]
    with store.pool.connection() as conn:
        assert (
            conn.execute(
                "SELECT * FROM canonical_turns WHERE canonical_turn_id=%s", (assistant_id,)
            ).fetchone()
            == before
        )
        later = conn.execute(
            "SELECT origin_channel_id,origin_channel_label FROM canonical_turns WHERE canonical_turn_id=%s",
            (later_id,),
        ).fetchone()
        assert later == {
            "origin_channel_id": "7000000000000000001",
            "origin_channel_label": "#other-synthetic",
        }
