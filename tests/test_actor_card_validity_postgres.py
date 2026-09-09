"""BUG-075: validity storage parity on an explicitly disposable PostgreSQL DB."""

import pytest

from tests.test_actor_card_validity import (
    CardWorld, _check_boundaries, _check_dirty_and_invalid,
    _check_invalid_write_atomic, _check_roundtrip, _check_scope, _check_existing_bounds_immutable,
)
from tests.test_audience_reassignment_postgres import _disposable_dsn
from virtual_context.storage.postgres import PostgresStore

pytestmark = [
    pytest.mark.regression("BUG-075"),
    pytest.mark.skipif(not _disposable_dsn(), reason="requires vc_disposable_* database"),
]


@pytest.fixture
def world(monkeypatch):
    dsn = _disposable_dsn()
    assert dsn, "refusing a non-disposable database"
    store = PostgresStore(dsn)
    try:
        with store.pool.connection() as conn:
            assert conn.execute("SELECT current_database() AS name").fetchone()["name"].startswith("vc_disposable_")
        yield CardWorld(store, monkeypatch)
    finally:
        store.close()


def test_pg_serve_window_starts_inclusively_and_expires_exclusively(world, monkeypatch):
    _check_boundaries(world, monkeypatch)


def test_pg_invalid_window_does_not_supersede_last_good_card(world):
    _check_invalid_write_atomic(world)


def test_pg_existing_entry_cannot_extend_or_remove_its_validity(world, monkeypatch):
    _check_existing_bounds_immutable(world, monkeypatch)


def test_pg_offset_roundtrip_and_scoped_carryover_do_not_extend_expiry(world, monkeypatch):
    _check_roundtrip(world, monkeypatch)


def test_pg_expiry_applies_to_dirty_last_good_card_and_invalid_bounds_fail_closed(world, monkeypatch):
    _check_dirty_and_invalid(world, monkeypatch)


def test_pg_bounded_scoped_preference_does_not_cross_audience(world):
    _check_scope(world)


def test_pg_existing_schema_adds_nullable_validity_without_rewriting_card(world):
    assert world.save([world.entry()]) == 1
    before = world.snapshot()
    with world.connection() as conn:
        conn.execute("ALTER TABLE actor_card_entries DROP COLUMN valid_from")
        conn.execute("ALTER TABLE actor_card_entries DROP COLUMN expires_at")
    world.store._ensure_actor_card_schema()
    assert world.snapshot() == before
    assert world.card().entries[0].valid_from is None
    assert world.card().entries[0].expires_at is None
    world.store._ensure_actor_card_schema()
    assert world.snapshot() == before
