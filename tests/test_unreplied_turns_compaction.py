"""An unreplied message is history: it compacts, and a compacted one does not end the prefix."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from virtual_context.engine import VirtualContextEngine
from virtual_context.storage.sqlite import SQLiteStore

CONV = "conv-unreplied"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_group(store: SQLiteStore, group: int, *, user: str, assistant: str | None) -> None:
    now = _now()
    store.save_canonical_turn(
        CONV, -1, user, "", turn_group_number=group, canonical_turn_id=f"u{group:03d}",
        sort_key=float(group * 10), turn_hash=f"h-u{group}", hash_version=1, tagged_at=now,
        primary_tag="topic", tags=["topic"], created_at=now, updated_at=now, first_seen_at=now, last_seen_at=now,
    )
    if assistant is not None:
        store.save_canonical_turn(
            CONV, -1, "", assistant, turn_group_number=group, canonical_turn_id=f"a{group:03d}",
            sort_key=float(group * 10 + 1), turn_hash=f"h-a{group}", hash_version=1, tagged_at=now,
            primary_tag="topic", tags=["topic"], created_at=now, updated_at=now, first_seen_at=now, last_seen_at=now,
        )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteStore:
    s = SQLiteStore(tmp_path / "unreplied.db")
    s.upsert_conversation(tenant_id="t", conversation_id=CONV)
    for group in range(8):
        _seed_group(s, group, user=f"member {group} says hi", assistant=None if group in (3, 6) else f"reply {group}")
    return s


@pytest.mark.regression("BUG-079")
def test_unreplied_message_outside_protected_tail_is_compactable(store):
    pending = store.get_uncompacted_canonical_turns(CONV, protected_recent_turns=2)
    groups = [row.turn_group_number for row in pending]
    assert groups == [0, 1, 2, 3, 4, 5], groups
    lone = next(row for row in pending if row.turn_group_number == 3)
    assert lone.user_content == "member 3 says hi" and not (lone.assistant_content or "").strip()
    assert store.get_uncompacted_canonical_turns(CONV, protected_recent_turns=8) == []


@pytest.mark.regression("BUG-079")
def test_store_watermark_counts_a_compacted_unreplied_turn_as_one_message(store):
    conn = store._get_conn()
    conn.execute("UPDATE canonical_turns SET compacted_at='2026-09-22' WHERE turn_group_number <= 5")
    # groups 0..5 = five pairs and one lone message: 5*2 + 1
    assert store.get_compaction_watermark(CONV) == (11, 5)
    conn.execute("UPDATE canonical_turns SET compacted_at=NULL WHERE turn_group_number = 3")
    assert store.get_compaction_watermark(CONV) == (6, 2)


@pytest.mark.regression("BUG-079")
def test_engine_prefix_counts_a_compacted_unreplied_turn_as_one_message():
    def row(user="", assistant="", compacted="2026-09-22"):
        return SimpleNamespace(user_content=user, assistant_content=assistant, compacted_at=compacted)
    paired = [
        (0, [row(user="q0"), row(assistant="a0")]),
        (1, [row(user="unreplied")]),
        (2, [row(user="q2"), row(assistant="a2")]),
        (3, [row(user="q3", compacted=None), row(assistant="a3", compacted=None)]),
    ]
    assert VirtualContextEngine._canonical_prefix_watermark(paired) == (5, 2)
    paired[1][1][0].compacted_at = None
    assert VirtualContextEngine._canonical_prefix_watermark(paired) == (2, 0)
