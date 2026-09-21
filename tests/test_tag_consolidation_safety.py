"""Consolidation apply safety: alias ownership, scoped reverts, verbatim plans, closure judging, fenced writes."""
import json
from datetime import datetime, timezone
from unittest import mock

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.conversation_store import ConversationStoreView, StaleConversationWriteError
from virtual_context.core.judgment import build_runtime, candidate_tag_pairs, judge_tag_consolidation
from virtual_context.core.tag_consolidator import (
    ConsolidationApplyError, ConsolidationGroup, consolidate_tags, revert_consolidation,
)
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


def test_judged_apply_skips_an_alias_owned_by_another_canonical(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-accuracy", "other-topic", conversation_id="conv-A")
        rt, _ = _jev({frozenset(("dosing-accuracy", "dosing-advice")): 0.9})
        result = consolidate_tags(store, llm=None, dry_run=False, judgment_runtime=rt, strict=True)
        assert [(g.canonical, g.aliases) for g in result.groups] == [("dosing-advice", ["dosing-accuracy"])]
        assert result.aliases_written == 0 and result.segment_tags_added == 0
        assert result.skipped == [{"alias": "dosing-accuracy", "canonical": "dosing-advice", "existing": "other-topic"}]
        assert result.applied == [{"canonical": "dosing-advice", "aliases_written": [], "aliases_rewritten": [],
                                   "segment_refs": []}]
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "other-topic"}
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
    finally:
        raw.close()


def test_reviewed_plan_is_refused_before_any_write_when_the_store_drifted(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-accuracy", "other-topic", conversation_id="conv-A")
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="plan")]
        with pytest.raises(ValueError, match="already maps to 'other-topic'"):
            consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        raw.delete_tag_alias("dosing-accuracy", conversation_id="conv-A")
        raw.set_tag_alias("dosing-advice", "other-topic", conversation_id="conv-A")
        with pytest.raises(ValueError, match="is already an alias"):
            consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-advice": "other-topic"}
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
    finally:
        raw.close()


def test_failure_inside_a_group_still_records_the_aliases_it_committed(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.store_segment(_segment("s8", ["dosing-notes"]))
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy", "dosing-notes"], reason="")]
        seen = []

        def boom(*a, **k):
            raise RuntimeError("connection reset")

        raw.add_tag_to_segments_with_tags = boom
        with pytest.raises(ConsolidationApplyError, match="connection reset") as failure:
            consolidate_tags(store, llm=None, dry_run=False, groups=plan, on_group_applied=seen.append)
        assert failure.value.applied == seen  # the record also travels on the error, callback or not
        # both alias rows committed before the backfill failed, and both are in the record
        assert seen == [{"canonical": "dosing-advice", "aliases_written": ["dosing-accuracy", "dosing-notes"],
                         "aliases_rewritten": [], "segment_refs": []}]
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice",
                                                                    "dosing-notes": "dosing-advice"}
        assert revert_consolidation(store, seen) == {"aliases_deleted": 2, "segment_tags_removed": 0}
        assert store.get_tag_aliases(conversation_id="conv-A") == {}
    finally:
        raw.close()


def test_concurrent_claim_during_an_exact_apply_is_an_error_with_provenance(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="")]
        raw.create_tag_alias_if_absent = lambda alias, canonical, conversation_id="": False
        seen = []
        with pytest.raises(ConsolidationApplyError, match="claimed concurrently"):
            consolidate_tags(store, llm=None, dry_run=False, groups=plan, on_group_applied=seen.append)
        assert seen == [{"canonical": "dosing-advice", "aliases_written": [], "aliases_rewritten": [], "segment_refs": []}]
        assert store.get_segment("s2").tags == ["dosing-accuracy"]
    finally:
        raw.close()


def test_apply_flattens_alias_chains_and_revert_restores_them(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.store_segment(_segment("s8", ["dosing-notes"]))
        raw.set_tag_alias("dosing-notes", "dosing-accuracy", conversation_id="conv-A")  # pre-existing hop
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="")]
        result = consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice",
                                                                    "dosing-notes": "dosing-advice"}
        entry = result.applied[0]
        assert entry["aliases_written"] == ["dosing-accuracy"]
        assert entry["aliases_rewritten"] == [["dosing-notes", "dosing-accuracy"]]
        assert sorted(entry["segment_refs"]) == ["s2", "s8"]
        assert "dosing-advice" in store.get_segment("s8").tags
        undone = revert_consolidation(store, result.applied)
        assert undone == {"aliases_deleted": 1, "segment_tags_removed": 2, "aliases_restored": 1}
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-notes": "dosing-accuracy"}
        assert store.get_segment("s8").tags == ["dosing-notes"]
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
        assert result.applied == [{"canonical": "dosing-advice", "aliases_written": [], "aliases_rewritten": [],
                                   "segment_refs": ["s2"]}]
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
    for name in ("create_tag_alias_if_absent", "rebind_tag_alias", "delete_tag_alias",
                 "add_tag_to_segments_with_tags", "remove_tag_from_segments"):
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


def test_rebind_is_a_compare_and_swap(tmp_sqlite_db):
    store = _store(tmp_sqlite_db)
    try:
        store.set_tag_alias("dosing-notes", "dosing-accuracy", conversation_id="conv-A")
        assert store.rebind_tag_alias("dosing-notes", "other", "dosing-advice", conversation_id="conv-A") is False
        assert store.rebind_tag_alias("dosing-notes", "dosing-accuracy", "dosing-advice", conversation_id="conv-A") is True
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-notes": "dosing-advice"}
    finally:
        store.close()


def test_apply_error_without_a_callback_still_carries_the_record(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="")]
        raw.add_tag_to_segments_with_tags = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reset"))
        with pytest.raises(ConsolidationApplyError) as failure:
            consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert failure.value.applied[0]["aliases_written"] == ["dosing-accuracy"]
        assert revert_consolidation(store, failure.value.applied)["aliases_deleted"] == 1
        assert store.get_tag_aliases(conversation_id="conv-A") == {}
    finally:
        raw.close()


def test_chain_rewrite_that_lost_a_race_stops_an_exact_apply(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-notes", "dosing-accuracy", conversation_id="conv-A")
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="")]
        real = raw.rebind_tag_alias

        def raced(alias, expected, new, conversation_id=""):
            raw.set_tag_alias(alias, "squat-form", conversation_id=conversation_id)  # a live writer moved it
            return real(alias, expected, new, conversation_id=conversation_id)

        raw.rebind_tag_alias = raced
        with pytest.raises(ConsolidationApplyError, match="changed concurrently") as failure:
            consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        assert failure.value.applied == [{"canonical": "dosing-advice", "aliases_written": [],
                                          "aliases_rewritten": [], "segment_refs": []}]
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-notes": "squat-form"}
    finally:
        raw.close()


def test_revert_leaves_a_rewrite_alone_when_a_later_writer_changed_it(tmp_sqlite_db):
    raw = _store(tmp_sqlite_db)
    store = ConversationStoreView(raw, "conv-A", 0)
    try:
        raw.set_tag_alias("dosing-notes", "dosing-accuracy", conversation_id="conv-A")
        plan = [ConsolidationGroup(canonical="dosing-advice", aliases=["dosing-accuracy"], reason="")]
        result = consolidate_tags(store, llm=None, dry_run=False, groups=plan)
        raw.set_tag_alias("dosing-notes", "squat-form", conversation_id="conv-A")  # changed after the apply
        undone = revert_consolidation(store, result.applied)
        assert undone["aliases_restore_skipped"] == 1 and "aliases_restored" not in undone
        assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-notes": "squat-form"}
    finally:
        raw.close()
