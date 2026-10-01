"""Record edits rebuild the segments, facts and topic summaries derived from changed turns."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

from virtual_context.core.canonical_turns import compute_turn_hash_from_raw
from virtual_context.types import (
    CompactionResult, Fact, SegmentMetadata, StoredSegment, TagSummary, TurnTagEntry,
)

CONV = "conv-record-edit"


def _id(n: int) -> str:
    return f"00000000-0000-0000-0000-{n:012d}"


class _Compactor:
    """Summaries and facts are the segment's own text, so a test can see what was rebuilt."""

    model_name = "stub"

    def compact(self, segments, **_kwargs):
        results = []
        for seg in segments:
            text = " | ".join(m.content for m in seg.messages)
            results.append(CompactionResult(
                segment_id=seg.id, primary_tag=seg.primary_tag, tags=list(seg.tags),
                summary=f"summary: {text}", summary_tokens=5, full_text=text, original_tokens=10,
                messages=[{"role": m.role, "content": m.content, "metadata": m.metadata} for m in seg.messages],
                metadata=SegmentMetadata(), compression_ratio=0.5,
                facts=[Fact(subject="user", verb="said", object=text, tags=[seg.primary_tag])],
            ))
        return results

    def compact_tag_summaries(self, *, cover_tags, max_turn, **_kwargs):
        return [TagSummary(tag=tag, summary=f"topic {tag}", summary_tokens=2, covers_through_turn=max_turn)
                for tag in cover_tags]


@pytest.fixture
def engine(tmp_path):
    from virtual_context.config import load_config
    from virtual_context.engine import VirtualContextEngine

    config = load_config(config_dict={
        "context_window": 10000,
        "storage": {"backend": "sqlite", "sqlite": {"path": str(tmp_path / "store.db")}},
        "tag_generator": {"type": "keyword"},
        "conversation_id": CONV,
    })
    engine = VirtualContextEngine(config=config)
    engine._compactor = engine._compaction._compactor = _Compactor()
    return engine


def _turn(engine, n, tags, user):
    engine._store.save_canonical_turn(
        CONV, n, user, f"reply {n}", canonical_turn_id=_id(n), turn_group_number=n,
        sort_key=float(n + 1), primary_tag=tags[0], tags=list(tags),
        tagged_at="2026-10-01T00:00:00+00:00", compacted_at="2026-10-01T00:00:00+00:00",
    )
    engine._turn_tag_index.append(TurnTagEntry(
        turn_number=n, canonical_turn_id=_id(n), tags=list(tags), primary_tag=tags[0],
    ))


def _segment(engine, ref, primary, tags, turns):
    engine._store.store_segment(StoredSegment(
        ref=ref, conversation_id=CONV, primary_tag=primary, tags=list(tags),
        summary=f"old summary of {primary}", full_text="old",
        metadata=SegmentMetadata(canonical_turn_ids=[_id(n) for n in turns], source_mapping_complete=True),
    ))
    engine._store.store_facts([Fact(subject="user", verb="said", object=f"old {primary}",
                                    segment_ref=ref, conversation_id=CONV, tags=[primary])])
    engine._store.save_tag_summary(TagSummary(tag=primary, summary=f"old topic {primary}"),
                                   conversation_id=CONV)


def test_forget_rebuilds_the_surviving_turns_and_drops_the_topic(engine):
    _turn(engine, 0, ["docker"], "my docker secret")
    _turn(engine, 1, ["docker", "python"], "docker and python together")
    _turn(engine, 2, ["python"], "plain python")
    _segment(engine, "seg-docker", "docker", ["docker", "python"], [0, 1])
    _segment(engine, "seg-python", "python", ["python"], [2])
    editor = engine.record_editor

    report = editor.forget_topic("Docker", actor="admin")
    assert report["found"] and report["turns_removed"] == 1
    assert engine._turn_tag_index.get_tags_for_canonical_turn(_id(0)) is None
    assert engine._turn_tag_index.get_tags_for_canonical_turn(_id(1)).tags == ["python"]

    op = editor.process(report["operation_id"])

    assert op["status"] == "completed" and not op["failed"]
    rebuilt = engine._store.get_segment("seg-docker")
    assert rebuilt.primary_tag == "python" and "docker" not in rebuilt.tags
    assert "secret" not in rebuilt.summary and "docker and python together" in rebuilt.summary
    facts = engine._store.get_facts_by_segment("seg-docker")
    assert [f.object for f in facts] == ["docker and python together | reply 1"]
    assert engine._store.get_tag_summary("docker", conversation_id=CONV) is None
    assert engine._store.get_tag_summary("python", conversation_id=CONV).summary == "topic python"
    assert editor.process(report["operation_id"])["status"] == "completed"


def test_forget_of_an_unknown_topic_reports_not_found(engine):
    _turn(engine, 0, ["python"], "plain python")
    assert engine.record_editor.forget_topic("docker", actor="admin")["found"] is False


def test_edit_rebuilds_the_segment_from_the_new_text(engine):
    _turn(engine, 0, ["python"], "I live in Boston")
    _segment(engine, "seg-python", "python", ["python"], [0])
    editor = engine.record_editor

    report = editor.edit_turn(_id(0), user_content="I live in Denver", actor="admin", reason="typo")
    op = editor.process(report["operation_id"])

    assert op["status"] == "completed"
    segment = engine._store.get_segment("seg-python")
    assert "Denver" in segment.summary and "Boston" not in segment.summary
    edit = engine._store.get_turn_edits(CONV, canonical_turn_id=_id(0))[0]
    assert edit["before_user_content"] == "I live in Boston" and edit["reason"] == "typo"


def test_removing_the_last_turn_of_a_segment_deletes_it(engine):
    _turn(engine, 0, ["python"], "only turn")
    _segment(engine, "seg-python", "python", ["python"], [0])

    report = engine.record_editor.remove_turn(_id(0), actor="admin")
    engine.record_editor.process(report["operation_id"])

    assert engine._store.get_segment("seg-python") is None
    assert engine._store.get_facts_by_segment("seg-python") == []
    assert engine._store.get_tag_summary("python", conversation_id=CONV) is None


def test_host_replay_does_not_restore_a_removed_turn_or_the_text_before_an_edit(engine):
    reconciler = engine._ingest_reconciler
    _turn(engine, 0, ["python"], "keep me")
    _turn(engine, 1, ["python"], "remove me")
    _turn(engine, 2, ["python"], "old words")
    engine.record_editor.remove_turn(_id(1), actor="admin")
    engine.record_editor.edit_turn(_id(2), user_content="new words", actor="admin")

    def replayed(user, assistant):
        row = reconciler._prepare_message_row(CONV, role="user", content=user)
        row.assistant_content = assistant
        row.turn_hash = compute_turn_hash_from_raw(user, assistant)[0]
        return row

    replay = [replayed("remove me", "reply 1"), replayed("old words", "reply 2"),
              replayed("brand new", "")]
    kept = reconciler._apply_record_edits(CONV, replay)

    assert [r.user_content for r in kept] == ["new words", "brand new"]
    stored = {r.turn_hash for r in engine._store.get_all_canonical_turns(CONV)}
    assert kept[0].turn_hash in stored


def test_vcforget_changes_the_record_and_queues_the_rebuild(engine):
    from types import SimpleNamespace

    from virtual_context.proxy.handlers import _handle_vcforget

    _turn(engine, 0, ["docker"], "docker only")
    _segment(engine, "seg-docker", "docker", ["docker"], [0])
    queued = []
    state = SimpleNamespace(engine=engine, metrics=None, submit_record_edit=queued.append)

    text = _handle_vcforget("docker", state, actor="actor:discord:42")

    assert text.startswith("Forgot 'docker': 1 turn(s) removed")
    assert engine.record_editor.status(queued[0])[0]["actor"] == "actor:discord:42"
    assert _handle_vcforget("nothing", state).startswith("Topic 'nothing' not found")
