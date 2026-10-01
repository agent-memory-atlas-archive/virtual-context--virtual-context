"""Admin edits change the conversation record and keep an immutable audit trail."""

from __future__ import annotations

import json
import sqlite3

import pytest

from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import Fact, SegmentMetadata, StoredSegment

CONV = "conv-record"


def _id(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "store.db")
    yield s
    s.close()


def _turn(store, n, tags, user="", assistant=""):
    store.save_canonical_turn(
        CONV, n, user or f"user {n} about {' '.join(tags)}", assistant or f"reply {n}",
        canonical_turn_id=_id(n), turn_group_number=n, sort_key=float(n + 1),
        primary_tag=tags[0] if tags else "_general", tags=list(tags),
    )


def _segment(store, ref, primary, tags, turn_ids):
    store.store_segment(StoredSegment(
        ref=ref, conversation_id=CONV, primary_tag=primary, tags=list(tags),
        summary=f"summary of {primary}", full_text="text",
        metadata=SegmentMetadata(canonical_turn_ids=list(turn_ids), source_mapping_complete=True),
    ))
    store.store_facts([Fact(subject="user", verb="likes", object=primary, segment_ref=ref,
                            conversation_id=CONV, tags=[primary])])


def _rows(store):
    conn = store._get_conn()
    return {r["canonical_turn_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM canonical_turns WHERE conversation_id=?", (CONV,))}


def test_forget_strips_the_tag_and_removes_turns_left_without_a_topic(store):
    _turn(store, 0, ["docker"])
    _turn(store, 1, ["docker", "python"])
    _turn(store, 2, ["python"])
    _segment(store, "seg-docker", "docker", ["docker", "python"], [_id(0), _id(1)])
    _segment(store, "seg-python", "python", ["python", "docker"], [_id(1), _id(2)])

    report = store.forget_topic_records(CONV, ["docker"], actor="admin", reason="asked")

    rows = _rows(store)
    assert _id(0) not in rows
    assert json.loads(rows[_id(1)]["tags_json"]) == ["python"]
    assert rows[_id(1)]["primary_tag"] == "python"
    assert report["turns_removed"] == 1 and report["turns_untagged"] == 1
    assert store.get_segment("seg-docker") is None
    assert store.get_facts_by_segment("seg-docker") == []
    assert "docker" not in store.get_segment("seg-python").tags
    edits = {e["canonical_turn_id"]: e for e in store.get_turn_edits(CONV)}
    assert edits[_id(0)]["action"] == "remove"
    assert edits[_id(0)]["before_user_content"] == "user 0 about docker"
    assert edits[_id(1)]["action"] == "forget_untag"
    assert json.loads(edits[_id(1)]["before_tags_json"]) == ["docker", "python"]
    op = store.get_record_edit_operation(report["operation_id"])
    assert op["status"] == "pending" and op["forgotten"] == ["docker"]
    assert {seg["ref"] for seg in op["segments"]} == {"seg-docker", "seg-python"}
    assert "docker" not in op["tags"]


def test_forget_of_an_unknown_topic_changes_nothing(store):
    _turn(store, 0, ["python"])
    report = store.forget_topic_records(CONV, ["docker"], actor="admin")
    assert report["turns_removed"] == report["turns_untagged"] == 0
    assert store.get_record_edit_operation(report["operation_id"])["status"] == "completed"
    assert store.get_turn_edits(CONV) == []


def test_edit_keeps_the_original_and_retires_its_hash(store):
    _turn(store, 0, ["python"], user="I live in Boston", assistant="noted")
    _segment(store, "seg-python", "python", ["python"], [_id(0)])
    old_hash = _rows(store)[_id(0)]["turn_hash"]

    report = store.edit_canonical_turn(CONV, _id(0), user_content="I live in Denver", actor="admin")

    row = _rows(store)[_id(0)]
    assert row["user_content"] == "I live in Denver" and row["assistant_content"] == "noted"
    assert row["turn_hash"] != old_hash
    edit = store.get_turn_edits(CONV, canonical_turn_id=_id(0))[0]
    assert (edit["before_user_content"], edit["after_user_content"]) == ("I live in Boston", "I live in Denver")
    assert edit["actor"] == "admin"
    assert store.get_retired_turns(CONV) == {old_hash: {
        "turn_hash": row["turn_hash"], "user_content": "I live in Denver", "assistant_content": "noted",
    }}
    op = store.get_record_edit_operation(report["operation_id"])
    assert op["retag"] == [_id(0)] and op["segments"][0]["ref"] == "seg-python"


def test_remove_takes_the_whole_exchange_and_retires_its_hashes(store):
    _turn(store, 0, ["python"])
    store.save_canonical_turn(CONV, 0, "", "assistant half", canonical_turn_id=_id(9),
                              turn_group_number=0, sort_key=1.5, tags=["python"])
    _turn(store, 1, ["python"])
    hashes = {r["turn_hash"] for k, r in _rows(store).items() if k in (_id(0), _id(9))}

    report = store.remove_canonical_turn(CONV, _id(0), actor="admin")

    assert set(_rows(store)) == {_id(1)}
    assert report["turns_removed"] == 2
    assert store.get_retired_turns(CONV) == {h: None for h in hashes}


def test_a_second_edit_waits_for_the_first_to_finish(store):
    _turn(store, 0, ["python"])
    _segment(store, "seg-python", "python", ["python"], [_id(0)])
    first = store.edit_canonical_turn(CONV, _id(0), user_content="changed", actor="admin")
    with pytest.raises(RuntimeError):
        store.remove_canonical_turn(CONV, _id(0), actor="admin")
    store.update_record_edit_operation(first["operation_id"], status="completed")
    store.remove_canonical_turn(CONV, _id(0), actor="admin")


def test_audit_rows_cannot_be_changed_and_go_with_the_conversation(store):
    _turn(store, 0, ["python"])
    store.remove_canonical_turn(CONV, _id(0), actor="admin")
    conn = store._get_conn()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        conn.execute("UPDATE turn_edits SET before_user_content='x'")
    conn.rollback()
    store.delete_conversation(CONV)
    assert store.get_turn_edits(CONV) == []
    assert store.list_record_edit_operations(CONV) == []
