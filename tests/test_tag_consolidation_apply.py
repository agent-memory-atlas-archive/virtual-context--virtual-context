"""Applying tag consolidation: set-based segment-tag writes with provenance, plans, reverts, strict judgment."""
import json
from datetime import datetime, timezone
from unittest import mock

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import JudgmentUnavailable, build_runtime, judge_tag_consolidation
from virtual_context.core.conversation_store import ConversationStoreView
from virtual_context.core.tag_consolidator import ConsolidationGroup, consolidate_tags, revert_consolidation
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import JudgmentConfig, SegmentMetadata, StoredSegment


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _segment(ref, tags, conversation_id="conv-A"):
    now = datetime.now(timezone.utc)
    return StoredSegment(ref=ref, conversation_id=conversation_id, primary_tag=tags[0], tags=list(tags), summary="s",
                         summary_tokens=1, full_tokens=2, full_text="t", metadata=SegmentMetadata(), created_at=now,
                         start_timestamp=now, end_timestamp=now)


def _store(tmp_sqlite_db):
    store = SQLiteStore(db_path=tmp_sqlite_db)
    store.store_segment(_segment("s1", ["dosing-advice"]))
    store.store_segment(_segment("s2", ["dosing-accuracy"]))
    store.store_segment(_segment("s3", ["dosing-accuracy", "dosing-advice"]))
    store.store_segment(_segment("s5", ["dosing-advice"]))  # advice outranks accuracy as canonical
    store.store_segment(_segment("s4", ["dosing-accuracy"], conversation_id="conv-B"))
    return store


def test_store_adds_canonical_to_alias_segments_set_based_and_scoped(tmp_sqlite_db):
    store = _store(tmp_sqlite_db)
    try:
        added = store.add_tag_to_segments_with_tags("dosing-advice", ["dosing-accuracy"], conversation_id="conv-A")
        assert sorted(added) == ["s2"]  # s3 already carries the canonical; s4 is another conversation
        assert sorted(store.get_segment("s2").tags) == ["dosing-accuracy", "dosing-advice"]
        assert store.get_segment("s4").tags == ["dosing-accuracy"]
        assert store.add_tag_to_segments_with_tags("dosing-advice", ["dosing-accuracy"], conversation_id="conv-A") == []
        assert store.remove_tag_from_segments("dosing-advice", ["s2"]) == 1
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
        store.set_tag_alias("dosing-accuracy", "dosing-advice", conversation_id="conv-A")
        assert store.delete_tag_alias("dosing-accuracy", conversation_id="conv-A") == 1
        assert store.get_tag_aliases(conversation_id="conv-A") == {}
    finally:
        store.close()


def _rt(answers_fn, threshold=0.5, mode="jev"):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "jev-t", "answers": answers_fn(body), "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode="legacy", seams={"tag_consolidation": mode}, noul_threshold=threshold),
                         environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


def _answers(same: dict[frozenset, float]):
    def fn(body):
        out = {}
        for key in body["questions"]:
            pair = frozenset(body["state"]["pairs"][key.split("__", 1)[1]])
            out[key] = {"type": "noul", "noul": same.get(pair, 0.05)}
        return out
    return fn


def test_strict_consolidation_raises_instead_of_falling_back_to_legacy():
    def broken(request):
        return httpx.Response(500, text="down")
    http = httpx.Client(transport=httpx.MockTransport(broken))
    rt = build_runtime(JudgmentConfig(mode="legacy", seams={"tag_consolidation": "jev"}), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    legacy = mock.MagicMock(return_value=[{"canonical": "x", "aliases": ["y"], "reason": "llm"}])
    with pytest.raises(JudgmentUnavailable):
        judge_tag_consolidation(["dosing-advice", "dosing-accuracy"], legacy=legacy, runtime=rt, canonical_rank={}, strict=True)
    legacy.assert_not_called()


def test_conflicting_group_is_dropped_not_merged():
    # A~B and B~C accepted, A~C explicitly rejected: single-link would merge all three
    rt, _ = _rt(_answers({frozenset(("dosing-a", "dosing-b")): 0.9, frozenset(("dosing-b", "dosing-c")): 0.9,
                          frozenset(("dosing-a", "dosing-c")): 0.05, frozenset(("squat-x", "squat-y")): 0.9}))
    groups = judge_tag_consolidation(["dosing-a", "dosing-b", "dosing-c", "squat-x", "squat-y"], legacy=lambda: [],
                                     runtime=rt, canonical_rank={"squat-x": 3})
    assert [(g["canonical"], g["aliases"]) for g in groups] == [("squat-x", ["squat-y"])]


def test_apply_from_a_plan_records_provenance_and_reverts(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)  # the engine's scoped view: writes stay in conv-A
    try:
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="plan")]
        result = consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert result.aliases_written == 1 and result.segment_tags_added == 1
        assert result.applied == [{"canonical": "dosing-advice", "aliases_written": ["dosing-accuracy"], "segment_refs": ["s2"]}]
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice"}
        assert "dosing-advice" in store.get_segment("s2").tags
        undone = revert_consolidation(store, result.applied)
        assert undone == {"aliases_deleted": 1, "segment_tags_removed": 1}
        assert store.get_tag_aliases(conversation_id="conv-A") == {}
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
        assert store.get_segment("s4").tags == ["dosing-accuracy"]  # other conversation untouched throughout
    finally:
        raw.close()


def test_dry_run_returns_groups_without_writes(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        rt, _ = _rt(_answers({frozenset(("dosing-accuracy", "dosing-advice")): 0.9}))
        result = consolidate_tags(store, llm=None, dry_run=True, judgment_runtime=rt, strict=True)
        assert [(g.canonical, g.aliases) for g in result.groups] == [("dosing-advice", ["dosing-accuracy"])]
        assert result.applied == [] and store.get_tag_aliases(conversation_id="conv-A") == {}
    finally:
        raw.close()
