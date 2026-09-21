"""Consolidation apply safety: alias ownership, scoped reverts, verbatim plans, closure judging, fenced writes."""
import json
from datetime import datetime, timezone
from unittest import mock

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.conversation_store import ConversationStoreView, StaleConversationWriteError
from virtual_context.core.judgment import build_runtime, candidate_tag_pairs, judge_tag_consolidation
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
    store.store_segment(_segment("s5", ["dosing-advice"]))
    store.store_segment(_segment("s4", ["dosing-accuracy"], conversation_id="conv-B"))
    return store


def test_tag_removal_and_alias_creation_are_conversation_scoped(tmp_sqlite_db):
    store = _store(tmp_sqlite_db)
    try:
        # s4 belongs to conv-B: a conv-A revert must not touch it
        assert store.remove_tag_from_segments("dosing-accuracy", ["s4"], conversation_id="conv-A") == 0
        assert store.get_segment("s4").tags == ["dosing-accuracy"]
        assert store.create_tag_alias_if_absent("dosing-accuracy", "dosing-advice", conversation_id="conv-A") is True
        assert store.create_tag_alias_if_absent("dosing-accuracy", "other-topic", conversation_id="conv-A") is False
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice"}
    finally:
        store.close()


def test_alias_owned_by_another_canonical_is_skipped_not_overwritten(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-accuracy", "other-topic", conversation_id="conv-A")
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="plan")]
        result = consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert result.aliases_written == 0 and result.segment_tags_added == 0
        assert result.skipped == [{"alias": "dosing-accuracy", "canonical": "dosing-advice", "existing": "other-topic"}]
        assert result.applied == [{"canonical": "dosing-advice", "aliases_written": [], "segment_refs": []}]
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "other-topic"}
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
    finally:
        raw.close()


def test_alias_already_pointing_at_the_canonical_is_backfilled_but_not_claimed(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-accuracy", "dosing-advice", conversation_id="conv-A")
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="plan")]
        result = consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert result.aliases_written == 0 and result.segment_tags_added == 1
        assert result.applied == [{"canonical": "dosing-advice", "aliases_written": [], "segment_refs": ["s2"]}]
        # revert removes only what this run wrote: the pre-existing alias survives
        assert revert_consolidation(store, result.applied) == {"aliases_deleted": 0, "segment_tags_removed": 1}
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice"}
    finally:
        raw.close()


def test_plan_groups_apply_verbatim_and_overlaps_are_refused(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        overlapping = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason=""),
                       ConsolidationGroup(canonical="dosing-accuracy", aliases=["dosing-notes"], reason="")]
        with pytest.raises(ValueError, match="overlap"):
            consolidate_tags(store, llm=None, dry_run=False, groups=overlapping)
        assert store.get_tag_aliases(conversation_id="conv-A") == {}
    finally:
        raw.close()


def test_progress_callback_receives_each_group_as_it_commits(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.store_segment(_segment("s6", ["squat-form"]))
        raw.store_segment(_segment("s7", ["squat-technique"]))
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason=""),
                ConsolidationGroup(canonical="squat-form", aliases=["squat-technique"], reason="")]
        seen = []
        consolidate_tags(store, llm=None, dry_run=False, groups=plan, on_group_applied=seen.append)
        assert [e["canonical"] for e in seen] == ["dosing-advice", "squat-form"]
        assert seen[1]["segment_refs"] == ["s7"]
    finally:
        raw.close()


def test_view_fences_consolidation_writes_for_stale_generations():
    raw = mock.MagicMock()
    raw.is_conversation_generation_current.return_value = False
    view = ConversationStoreView(raw, "conv-A", 3)
    for name in ("create_tag_alias_if_absent", "delete_tag_alias", "add_tag_to_segments_with_tags", "remove_tag_from_segments"):
        with pytest.raises(StaleConversationWriteError):
            getattr(view, name)("x", ["y"], conversation_id="conv-A")
    raw.create_tag_alias_if_absent.assert_not_called()


def _jev(answers: dict[frozenset, float]):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        out = {}
        for key in body["questions"]:
            pair = frozenset(body["state"]["pairs"][key.split("__", 1)[1]])
            out[key] = {"type": "noul", "noul": answers.get(pair, 0.05)}
        return httpx.Response(200, json={"model": "jev-t", "answers": out, "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode="legacy", seams={"tag_consolidation": "jev"}, noul_threshold=0.5),
                         environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


def test_transitive_group_pairs_missing_from_candidates_are_judged_before_acceptance():
    # limit=2 keeps (a,b) and (a,z); (b,z) is never a candidate, so the chain must be closed by a second call
    tags = ["dosing-a", "dosing-b", "dosing-z"]
    assert candidate_tag_pairs(tags, limit=2) == [("dosing-a", "dosing-b"), ("dosing-a", "dosing-z")]
    accepted = {frozenset(("dosing-a", "dosing-b")): 0.9, frozenset(("dosing-a", "dosing-z")): 0.9}

    rt, seen = _jev({**accepted, frozenset(("dosing-b", "dosing-z")): 0.05})
    groups = judge_tag_consolidation(tags, legacy=lambda: [], runtime=rt, canonical_rank={"dosing-a": 5}, max_pairs=2)
    assert groups == []  # the closure pair contradicted the chain: vetoed
    assert len(seen) == 2 and list(seen[1]["state"]["pairs"].values()) == [["dosing-b", "dosing-z"]]

    judgment.reset()
    rt, seen = _jev({**accepted, frozenset(("dosing-b", "dosing-z")): 0.9})
    groups = judge_tag_consolidation(tags, legacy=lambda: [], runtime=rt, canonical_rank={"dosing-a": 5}, max_pairs=2)
    assert [(g["canonical"], sorted(g["aliases"])) for g in groups] == [("dosing-a", ["dosing-b", "dosing-z"])]


def test_strict_runs_fail_closed_when_embeddings_are_unavailable():
    def broken(texts):
        raise ConnectionError("sidecar down")
    assert candidate_tag_pairs(["dosing-a", "dosing-b"], embed_fn=broken) == [("dosing-a", "dosing-b")]
    with pytest.raises(ConnectionError):
        candidate_tag_pairs(["dosing-a", "dosing-b"], embed_fn=broken, strict_embed=True)
    rt, _ = _jev({})
    with pytest.raises(ConnectionError):
        judge_tag_consolidation(["dosing-a", "dosing-b"], legacy=lambda: [], runtime=rt, canonical_rank={},
                                embed_fn=broken, strict=True)
