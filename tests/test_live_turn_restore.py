"""A provider-mode engine can restore only the live turns from canonical rows.

When the shared session state already holds the turn index and markers,
loading every canonical row to rebuild what the session replaces is wasted
work with a large memory peak. The live (uncompacted) and pending turns are
the only part the session does not carry.
"""

from __future__ import annotations

import sqlite3

import fakeredis
import pytest

from virtual_context.engine import VirtualContextEngine
from virtual_context.proxy.session_state import SessionState, SessionStateProvider
from virtual_context.types import Message, VirtualContextConfig

CONV = "conv-live"


def _engine(tmp_path):
    config = VirtualContextConfig()
    config.conversation_id = CONV
    config.storage.backend = "sqlite"
    config.storage.sqlite_path = str(tmp_path / "store.db")
    return VirtualContextEngine(config=config)


def _seed(tmp_path, turns=5, compacted=3):
    engine = _engine(tmp_path)
    history: list[Message] = []
    for i in range(turns):
        history += [Message(role="user", content=f"question {i}"), Message(role="assistant", content=f"answer {i}")]
        engine.on_turn_complete(history)
    with sqlite3.connect(tmp_path / "store.db") as db:
        db.execute("UPDATE canonical_turns SET tagged_at = created_at WHERE tagged_at IS NULL")
        rows = db.execute("SELECT canonical_turn_id, turn_group_number FROM canonical_turns ORDER BY sort_key").fetchall()
        done = {tid for tid, group in rows if group is not None and group < compacted}
        db.executemany("UPDATE canonical_turns SET compacted_at = created_at WHERE canonical_turn_id = ?",
                       [(tid,) for tid in done])
        db.execute("DELETE FROM engine_state")


@pytest.mark.regression("BUG-090")
def test_live_restore_matches_the_full_restore(tmp_path):
    _seed(tmp_path)
    full = _engine(tmp_path)
    full._restore_from_canonical_rows(CONV)
    light = _engine(tmp_path)
    light._restored_conversation_history, light._restored_pending_turns = [], []
    assert light.restore_live_turns_from_canonical_rows(CONV) is True
    assert light._restored_conversation_history == full._restored_conversation_history
    assert light._restored_pending_turns == full._restored_pending_turns
    assert [t for t, *_ in light._restored_conversation_history] == [3, 4]


def test_the_provider_reports_whether_a_session_is_stored():
    provider = SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)
    assert provider.has_state(CONV) is False
    provider.save(CONV, SessionState())
    assert provider.has_state(CONV) is True
