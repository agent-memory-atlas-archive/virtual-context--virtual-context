"""BUG-075: grounded card influence has a deterministic validity window."""
from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3
import uuid

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import ActorCardEntry, ActorCardEntrySource

pytestmark = pytest.mark.regression("BUG-075")
NOW = datetime(2040, 6, 1, 12, tzinfo=timezone.utc)
START = "2040-06-01T12:00:00+00:00"
END = "2040-07-01T12:00:00+00:00"


class FrozenDatetime(datetime):
    instant = NOW

    @classmethod
    def now(cls, tz=None):
        return cls.instant.astimezone(tz) if tz else cls.instant.replace(tzinfo=None)


class CardWorld:
    def __init__(self, store, monkeypatch):
        self.store = store
        self.pg = hasattr(store, "pool")
        module = "virtual_context.storage.postgres" if self.pg else "virtual_context.storage.sqlite"
        monkeypatch.setattr(module + ".datetime", FrozenDatetime)
        monkeypatch.setattr(FrozenDatetime, "instant", NOW)
        self.tenant = "validity-" + uuid.uuid4().hex
        self.owner = "conversation-" + uuid.uuid4().hex
        self.actor = "actor:discord:" + uuid.uuid4().hex
        self.turn = str(uuid.uuid4())
        with self.connection() as conn:
            conn.execute(self.sql("""INSERT INTO conversations
                (conversation_id,tenant_id,lifecycle_epoch,phase,created_at,updated_at)
                VALUES (?, ?, 1, 'active', ?, ?)"""),
                (self.owner, self.tenant, START, START))
            conn.execute(self.sql("""INSERT INTO canonical_turns
                (canonical_turn_id,conversation_id,turn_hash,sort_key,user_content,
                 assistant_content,sender_actor_id,audience_conversation_id,
                 audience_attribution_version,origin_channel_id)
                VALUES (?, ?, ?, 1, ?, '', ?, ?, 1, 'channel-a')"""),
                (self.turn, self.owner, self.turn,
                 "Please use a playful form of address during the agreed period.",
                 self.actor, self.owner))
        store.upsert_actor_profile_from_turn(self.owner, self.actor, "Member A", seen_at=START)

    def sql(self, text):
        return text.replace("?", "%s") if self.pg else text

    @contextmanager
    def connection(self):
        if self.pg:
            with self.store.pool.connection() as conn:
                yield conn
        else:
            conn = self.store._get_conn()
            yield conn
            conn.commit()

    def entry(self, *, valid_from=None, expires_at=None, scope="same_conversation", eid=None):
        entry = ActorCardEntry(
            id=eid or "entry-" + uuid.uuid4().hex,
            tenant_id=self.tenant, actor_id=self.actor, kind="communication_pref",
            body="Wants playful forms of address during the agreed period.",
            confidence=0.7, audience_scope=scope,
        )
        # Dynamic assignment also exercises the pre-change writer, which
        # silently discarded these fields instead of enforcing their bounds.
        entry.valid_from = valid_from
        entry.expires_at = expires_at
        return entry

    def save(self, entries):
        return self.store.replace_actor_card(
            self.tenant, self.actor,
            [(entry, [ActorCardEntrySource(
                entry_id=entry.id, tenant_id=self.tenant,
                owner_conversation_id=self.owner, audience_conversation_id=self.owner,
                audience_channel_id="channel-a", canonical_turn_id=self.turn,
            )]) for entry in entries],
            input_hash="validity-input", expected_source_epochs={self.owner: 1},
        )

    def card(self):
        return self.store.get_actor_card(
            self.tenant, self.actor, owner_conversation_id=self.owner,
            audience_conversation_id=self.owner, audience_channel_id="channel-b",
        )

    def carryovers(self):
        return self.store.list_actor_card_carryovers(self.tenant, self.actor)

    def snapshot(self):
        with self.connection() as conn:
            entries = conn.execute(self.sql("""SELECT id,body,superseded_by,updated_at,valid_from,expires_at
                FROM actor_card_entries WHERE tenant_id=? ORDER BY id"""), (self.tenant,)).fetchall()
            profile = conn.execute(self.sql("""SELECT card_dirty,card_input_hash,card_built_at
                FROM actor_profiles WHERE tenant_id=? AND actor_id=?"""),
                (self.tenant, self.actor)).fetchone()
        return [dict(row) for row in entries], dict(profile)


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = SQLiteStore(db_path=str(tmp_path / "validity.db"))
    yield CardWorld(store, monkeypatch)
    store.close()


def _check_boundaries(world, monkeypatch):
    entry = world.entry(valid_from=START, expires_at=END)
    assert world.save([entry]) == 1
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 6, 1, 11, 59, 59, tzinfo=timezone.utc))
    assert world.card() is None
    monkeypatch.setattr(FrozenDatetime, "instant", NOW)
    assert [e.id for e in world.card().entries] == [entry.id]
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 7, 1, 12, tzinfo=timezone.utc))
    assert world.card() is None


def test_serve_window_starts_inclusively_and_expires_exclusively(world, monkeypatch):
    _check_boundaries(world, monkeypatch)


def _check_invalid_write_atomic(world):
    assert world.save([world.entry()]) == 1
    before = world.snapshot()
    assert world.save([world.entry(), world.entry(valid_from=END, expires_at=START)]) == 0
    assert world.snapshot() == before


def test_invalid_window_does_not_supersede_last_good_card(world):
    _check_invalid_write_atomic(world)


def _check_existing_bounds_immutable(world, monkeypatch):
    entry = world.entry(valid_from=START, expires_at=END)
    assert world.save([entry]) == 1
    # An equivalent offset representation is still the same interval.
    entry.valid_from = "2040-06-01T08:00:00-04:00"
    assert world.save([entry]) == 1
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 7, 1, 12, tzinfo=timezone.utc))
    assert world.card() is None
    before = world.snapshot()
    entry.expires_at = "2040-08-01T12:00:00Z"
    assert world.save([entry]) == 0
    assert world.snapshot() == before
    assert world.card() is None
    entry.valid_from = entry.expires_at = None
    assert world.save([entry]) == 0
    assert world.snapshot() == before
    legacy = world.entry()
    assert world.save([legacy]) == 1
    before = world.snapshot()
    legacy.expires_at = "2040-08-01T12:00:00Z"
    assert world.save([legacy]) == 0
    assert world.snapshot() == before


def test_existing_entry_cannot_extend_or_remove_its_validity(world, monkeypatch):
    _check_existing_bounds_immutable(world, monkeypatch)


def _check_roundtrip(world, monkeypatch):
    entry = world.entry(valid_from="2040-06-01T08:00:00-04:00", expires_at="2040-07-01T14:00:00+02:00")
    assert world.save([entry]) == 1
    served = world.card().entries[0]
    assert served.valid_from == "2040-06-01T12:00:00.000000+00:00"
    assert served.expires_at == "2040-07-01T12:00:00.000000+00:00"
    carried = world.carryovers()
    assert len(carried) == 1
    assert carried[0][0].audience_scope == "same_conversation"
    assert carried[0][0].expires_at == served.expires_at
    assert carried[0][1][0].audience_conversation_id == world.owner
    # A future start survives rebuilding while it remains hidden on serving.
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 5, 1, tzinfo=timezone.utc))
    assert world.card() is None
    assert world.carryovers()[0][0].valid_from == served.valid_from
    assert world.save([carried[0][0]]) == 1
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 7, 1, 12, tzinfo=timezone.utc))
    assert world.card() is None
    assert world.carryovers() == []


def test_offset_roundtrip_and_scoped_carryover_do_not_extend_expiry(world, monkeypatch):
    _check_roundtrip(world, monkeypatch)


def _check_dirty_and_invalid(world, monkeypatch):
    bounded, unbounded = world.entry(valid_from=START, expires_at=END), world.entry(scope="cross_context")
    assert world.save([bounded, unbounded]) == 2
    assert world.store.mark_actor_card_dirty(world.tenant, world.actor)
    assert len(world.card().entries) == 2
    monkeypatch.setattr(FrozenDatetime, "instant", datetime(2040, 7, 1, 12, tzinfo=timezone.utc))
    assert [e.id for e in world.card().entries] == [unbounded.id]
    with world.connection() as conn:
        conn.execute(world.sql("UPDATE actor_card_entries SET expires_at='not-a-time' WHERE id=?"), (unbounded.id,))
    assert world.card() is None
    assert world.carryovers() == []


def test_expiry_applies_to_dirty_last_good_card_and_malformed_bounds_fail_closed(world, monkeypatch):
    _check_dirty_and_invalid(world, monkeypatch)


@pytest.mark.parametrize("start,end", [
    ("2040-06-01", END), ("2040-06-01T12:00:00", END),
    (True, END), (START, ""), (START, START), (END, START),
])
def test_invalid_bounds_reject_before_any_card_mutation(world, start, end):
    assert world.save([world.entry()]) == 1
    before = world.snapshot()
    assert world.save([world.entry(valid_from=start, expires_at=end)]) == 0
    assert world.snapshot() == before


def test_optional_validity_defaults_leave_legacy_entries_unbounded():
    entry = ActorCardEntry()
    assert entry.valid_from is None and entry.expires_at is None


def _check_scope(world):
    entry = world.entry(valid_from=START, expires_at=END)
    assert world.save([entry]) == 1
    other = "conversation-" + uuid.uuid4().hex
    with world.connection() as conn:
        conn.execute(world.sql("""INSERT INTO conversations
            (conversation_id,tenant_id,lifecycle_epoch,phase,created_at,updated_at)
            VALUES (?, ?, 1, 'active', ?, ?)"""), (other, world.tenant, START, START))
    assert world.store.get_actor_card(
        world.tenant, world.actor, owner_conversation_id=world.owner,
        audience_conversation_id=other, audience_channel_id="channel-a",
    ) is None
    assert world.card().entries[0].id == entry.id
    assert world.carryovers()[0][1][0].audience_conversation_id == world.owner


def test_bounded_scoped_preference_does_not_cross_audience(world):
    _check_scope(world)


def test_existing_sqlite_schema_adds_nullable_validity_without_rewriting_card(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    store = SQLiteStore(db_path=str(path))
    world = CardWorld(store, monkeypatch)
    assert world.save([world.entry()]) == 1
    before = world.snapshot()
    store.close()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE actor_card_entries DROP COLUMN valid_from")
        conn.execute("ALTER TABLE actor_card_entries DROP COLUMN expires_at")
    store = SQLiteStore(db_path=str(path))
    world.store = store
    try:
        assert world.snapshot() == before
        assert world.card().entries[0].valid_from is None
        assert world.card().entries[0].expires_at is None
        store._ensure_actor_card_schema(store._get_conn())
        assert world.snapshot() == before
    finally:
        store.close()
