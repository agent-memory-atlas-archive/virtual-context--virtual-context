"""A conversation with tagged turns but no saved engine state keeps its tags.

Resetting a conversation's derived data deletes its engine state and keeps
its tagged canonical turns. The next engine must rebuild its turn-tag index
from those turns; otherwise compaction finds every turn missing from the
index and tags the whole history again from scratch.
"""

from __future__ import annotations

import pytest

from virtual_context.engine import VirtualContextEngine
from virtual_context.types import Message, VirtualContextConfig


def _engine(tmp_path):
    config = VirtualContextConfig()
    config.conversation_id = "conv-restore"
    config.storage.backend = "sqlite"
    config.storage.sqlite_path = str(tmp_path / "store.db")
    return VirtualContextEngine(config=config)


@pytest.mark.regression("BUG-088")
def test_the_index_is_rebuilt_from_tagged_turns_when_engine_state_is_gone(tmp_path):
    engine = _engine(tmp_path)
    history: list[Message] = []
    for text in ("the database migration plan", "fix the deploy bug in git", "api error on deploy"):
        history += [Message(role="user", content=text), Message(role="assistant", content=f"noted: {text}")]
        engine.on_turn_complete(history)
    tagged = {e.turn_number: list(e.tags) for e in engine._turn_tag_index.entries}
    assert tagged

    import sqlite3
    with sqlite3.connect(tmp_path / "store.db") as db:
        # Rows admitted through a transport are stamped when tagged; this
        # in-process path writes the tags without the stamp.
        db.execute("UPDATE canonical_turns SET tagged_at = created_at WHERE tagged_at IS NULL")
        db.execute("DELETE FROM engine_state WHERE conversation_id = ?", ("conv-restore",))

    fresh = _engine(tmp_path)
    restored = {e.turn_number: list(e.tags) for e in fresh._turn_tag_index.entries}
    assert restored == tagged
    # Compaction's segmenter looks turns up in the same index.
    assert fresh._segmenter._turn_tag_index is fresh._turn_tag_index
    assert fresh._retriever._turn_tag_index is fresh._turn_tag_index


@pytest.mark.regression("BUG-088")
def test_components_built_before_a_saved_state_restore_see_the_restored_index(tmp_path):
    engine = _engine(tmp_path)
    history: list[Message] = []
    for text in ("the database migration plan", "fix the deploy bug in git"):
        history += [Message(role="user", content=text), Message(role="assistant", content=f"noted: {text}")]
        engine.on_turn_complete(history)
    import sqlite3
    with sqlite3.connect(tmp_path / "store.db") as db:
        db.execute("UPDATE canonical_turns SET tagged_at = created_at WHERE tagged_at IS NULL")

    fresh = _engine(tmp_path)
    assert fresh._turn_tag_index.entries
    assert fresh._segmenter._turn_tag_index is fresh._turn_tag_index
    assert fresh._retriever._turn_tag_index is fresh._turn_tag_index
