"""BUG-076: source occurrence parity on an explicitly disposable PostgreSQL DB."""
from __future__ import annotations

import psycopg
import pytest

from tests.test_audience_reassignment_postgres import _disposable_dsn
from tests.test_source_event_times import OCCURRED, _ingest, _times
from virtual_context.core.exceptions import CanonicalSourceConflict
from virtual_context.core.ingest_reconciler import IngestReconciler
from virtual_context.core.semantic_search import SemanticSearchManager
from virtual_context.storage.postgres import PostgresStore
from virtual_context.types import VirtualContextConfig

pytestmark = [pytest.mark.regression("BUG-076"), pytest.mark.skipif(
    not _disposable_dsn(), reason="requires vc_disposable_* database",
)]


@pytest.fixture
def source_store():
    dsn = _disposable_dsn()
    assert dsn, "refusing non-disposable database"
    store = PostgresStore(dsn)
    try:
        with store.pool.connection() as conn:
            conn.execute("TRUNCATE canonical_turns, conversations, conversation_lifecycle CASCADE")
        store.activate_conversation("owner")
        store.upsert_conversation(tenant_id="tenant", conversation_id="owner")
        semantic = SemanticSearchManager(store, VirtualContextConfig(conversation_id="owner"))
        semantic._embed_fn = None
        yield store, IngestReconciler(store=store, semantic=semantic)
    finally:
        store.close()


def _snapshot(store):
    with store.pool.connection() as conn:
        return {table: conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("canonical_turns", "canonical_message_sources", "canonical_source_event_times")}


def test_pg_source_occurrence_new_admission_and_exact_replay(source_store):
    store, rec = source_store
    source = _ingest(store, rec, OCCURRED)
    assert _times(store, source) == {("owner", source["canonical_turn_id"]): OCCURRED}
    before = _snapshot(store)
    _ingest(store, rec)
    _ingest(store, rec, OCCURRED)
    with pytest.raises(CanonicalSourceConflict):
        _ingest(store, rec, "2026-08-21T12:34:56.789Z")
    assert _snapshot(store) == before
    with store.pool.connection() as conn, pytest.raises(psycopg.errors.RaiseException):
        conn.execute("UPDATE canonical_source_event_times SET occurred_at='2026-08-21T12:34:56.789Z'")
    assert _times(store, source, tenant="wrong-tenant") == {}


def test_pg_source_occurrence_later_enrichment_preserves_ledger(source_store):
    store, rec = source_store
    source = _ingest(store, rec)
    assert _times(store, source) == {}
    before = _snapshot(store)
    _ingest(store, rec, OCCURRED)
    after = _snapshot(store)
    assert after["canonical_turns"] == before["canonical_turns"]
    assert after["canonical_message_sources"] == before["canonical_message_sources"]
    assert len(after["canonical_source_event_times"]) == 1
    assert _times(store, source) == {("owner", source["canonical_turn_id"]): OCCURRED}


def test_pg_source_occurrence_noop_guard_function_fails_schema_assertion(source_store):
    from virtual_context.storage.source_event_times import assert_source_event_time_schema
    store, _ = source_store
    with store.pool.connection() as conn:
        conn.execute("CREATE OR REPLACE FUNCTION guard_source_event_time() RETURNS trigger AS $$ BEGIN RETURN OLD; END; $$ LANGUAGE plpgsql")
        with pytest.raises(RuntimeError, match="unguarded"):
            assert_source_event_time_schema(conn, "postgres")


@pytest.mark.parametrize("legacy_metadata", ["missing_audience", "inherited_actor"])
@pytest.mark.parametrize("initial_occurrence", [None, OCCURRED])
def test_pg_optional_occurrence_preserves_exact_legacy_replay(source_store, legacy_metadata, initial_occurrence):
    from tests.test_source_event_times import test_optional_occurrence_preserves_exact_legacy_replay
    test_optional_occurrence_preserves_exact_legacy_replay(source_store, legacy_metadata, initial_occurrence)
