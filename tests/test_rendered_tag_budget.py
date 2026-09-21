"""The retriever's tag budget bounds the rendered tag sections the assembler admits."""
from datetime import datetime, timezone

from virtual_context.core.assembler import ContextAssembler
from virtual_context.types import (
    AssemblerConfig, DepthLevel, RetrievalResult, SegmentMetadata, StoredSegment, StoredSummary,
    WorkingSetEntry,
)

from tests.test_retriever import _make_retriever


def _summary(i: int, tag: str) -> StoredSummary:
    now = datetime.now(timezone.utc)
    return StoredSummary(
        ref=f"ref-{i}", primary_tag=tag, tags=[tag], summary="x" * 400, summary_tokens=100,
        full_tokens=400, metadata=SegmentMetadata(), created_at=now, start_timestamp=now,
        end_timestamp=now,
    )


def _assembler() -> ContextAssembler:
    return ContextAssembler(config=AssemblerConfig(core_context_max_tokens=1000, tag_context_max_tokens=5000))


def _tags_tokens(result) -> int:
    return result.budget_breakdown["tags"]


def _one_section_cost(assembler) -> int:
    rr = RetrievalResult(tags_matched=["t0"], summaries=[_summary(0, "t0")], total_tokens=100)
    return _tags_tokens(assembler.assemble(core_context="core", retrieval_result=rr, conversation_history=[], token_budget=10_000))


def _three(budget):
    tags = ["t0", "t1", "t2"]
    return RetrievalResult(tags_matched=tags, summaries=[_summary(i, t) for i, t in enumerate(tags)], total_tokens=300,
                           retrieval_scores={"t0": 3.0, "t1": 2.0, "t2": 1.0},
                           retrieval_metadata=({"tag_token_budget": budget} if budget is not None else {}))


def test_retriever_records_the_budget_it_selected_with(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db, max_budget_fraction=0.25)
    try:
        result = retriever.retrieve("What about the court filing?")
    finally:
        store.close()
    assert result.retrieval_metadata["tag_token_budget"] == 7500


def test_rendered_sections_are_capped_at_the_retriever_budget():
    assembler = _assembler()
    one = _one_section_cost(assembler)
    assert one > 0
    result = assembler.assemble(core_context="core", retrieval_result=_three(one + 1), conversation_history=[], token_budget=10_000)
    assert list(result.tag_sections) == ["t0"]
    assert _tags_tokens(result) <= one + 1
    assert result.tag_token_budget == one + 1
    assert result.tags_over_budget == 2


def test_result_without_a_budget_key_keeps_the_configured_cap():
    assembler = _assembler()
    result = assembler.assemble(core_context="core", retrieval_result=_three(None), conversation_history=[], token_budget=10_000)
    assert sorted(result.tag_sections) == ["t0", "t1", "t2"]
    assert result.tag_token_budget == 5000


def test_top_retrieved_section_is_admitted_even_over_a_scaled_budget():
    assembler = _assembler()
    result = assembler.assemble(core_context="core", retrieval_result=_three(1), conversation_history=[], token_budget=10_000)
    assert list(result.tag_sections) == ["t0"]
    assert result.tags_over_budget == 2


def test_working_set_expansion_is_not_capped_by_the_retriever_budget():
    assembler = _assembler()
    now = datetime.now(timezone.utc)
    segment = StoredSegment(
        ref="seg-1", primary_tag="expanded", tags=["expanded"], summary="s", full_text="y" * 2000,
        summary_tokens=10, full_tokens=500, metadata=SegmentMetadata(), created_at=now,
        start_timestamp=now, end_timestamp=now,
    )
    rr = RetrievalResult(tags_matched=["expanded", "t1", "t2"], summaries=[_summary(1, "t1"), _summary(2, "t2")], total_tokens=200,
                         retrieval_scores={"expanded": 3.0, "t1": 2.0, "t2": 1.0},
                         retrieval_metadata={"tag_token_budget": 1})
    result = assembler.assemble(
        core_context="core", retrieval_result=rr, conversation_history=[], token_budget=10_000,
        working_set={"expanded": WorkingSetEntry(tag="expanded", depth=DepthLevel.FULL)},
        full_segments={"expanded": [segment]},
    )
    assert "expanded" in result.tag_sections
    assert "t1" in result.tag_sections and "t2" not in result.tag_sections  # paged-in section does not consume the retrieved floor


def test_summary_floor_result_carries_the_budget(tmp_sqlite_db):
    from virtual_context.types import TagSummary

    retriever, store = _make_retriever(tmp_sqlite_db, skip_active=True)
    try:
        store.save_tag_summary(TagSummary(tag="legal", summary="Case 24-cv-1234 filing deadline.", summary_tokens=8,
                                          source_segment_refs=["legal-1"]))
        result = retriever.retrieve("What about the court filing?", current_active_tags=["legal"], post_compaction=True)
    finally:
        store.close()
    assert result.retrieval_metadata.get("summary_floor") is True
    assert result.retrieval_metadata["tag_token_budget"] == 30000


def test_unscored_alias_ride_along_cannot_take_the_guaranteed_slot():
    assembler = _assembler()
    rr = RetrievalResult(tags_matched=["primary"], summaries=[_summary(0, "alias"), _summary(1, "primary")], total_tokens=200,
                         retrieval_scores={"primary": 0.03},  # the alias ride-along has no score
                         retrieval_metadata={"tag_token_budget": 1})
    result = assembler.assemble(core_context="core", retrieval_result=rr, conversation_history=[], token_budget=10_000)
    assert list(result.tag_sections) == ["primary"]
