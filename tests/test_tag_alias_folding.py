"""Aliased tags fold into one canonical retrieval slot and never re-enter through tag reuse."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime
from virtual_context.core.tag_canonicalizer import TagCanonicalizer
from virtual_context.core.tag_generator import LLMTagGenerator
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import JudgmentConfig, SegmentMetadata, StoredSegment, TagGeneratorConfig

from conftest import MockLLMProvider, MockTagGenerator
from tests.test_retriever import _make_retriever


def _seg(ref, tags):
    now = datetime.now(timezone.utc)
    return StoredSegment(ref=ref, primary_tag=tags[0], tags=list(tags), summary=f"about {tags[0]}", summary_tokens=5,
                         full_tokens=20, full_text="t", metadata=SegmentMetadata(), created_at=now,
                         start_timestamp=now, end_timestamp=now)


def test_retriever_folds_alias_tags_into_one_canonical_slot(tmp_sqlite_db):
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        for i, tag in enumerate(["dosing-advice", "dosing-accuracy", "dosing-notes", "gardening"]):
            store.store_segment(_seg(f"d-{i}", [tag]))
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        store.set_tag_alias("dosing-notes", "dosing-advice")
        retriever.config.strategy_configs["default"].max_results = 2
        retriever.tag_generator.set_override(
            "dosing", __import__("virtual_context.types", fromlist=["TagResult"]).TagResult(
                tags=["dosing-advice", "dosing-accuracy", "dosing-notes", "gardening"], primary="dosing-advice", source="mock"))
        result = retriever.retrieve("dosing question")
    finally:
        store.close()
    # two slots: the dosing group counts once, so gardening gets the second slot
    assert set(result.retrieval_metadata["top_tags"]) == {"dosing-advice", "gardening"}
    refs = {s.ref for s in result.summaries}
    assert {"d-0", "d-1", "d-2", "d-3"} <= refs  # alias-tagged segments still come back
    assert result.retrieval_scores["dosing-accuracy"] == result.retrieval_scores["dosing-advice"]


def _jev_choice(choice):
    def handler(request):
        body = json.loads(request.content)
        out = {}
        for key, q in body["questions"].items():
            opts = list(q["criteria"])
            out[key] = {"type": "choice", "choice": choice, "confidence": 0.95,
                        "probabilities": {o: (0.95 if o == choice else 0.0) for o in opts}}
        return httpx.Response(200, json={"model": "jev-t", "answers": out, "usage": {}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode="jev"), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)


def test_tag_reuse_never_selects_an_alias_and_canonicalizes_its_pick(tmp_sqlite_db):
    judgment.reset()
    store = SQLiteStore(db_path=tmp_sqlite_db)
    try:
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        canon = TagCanonicalizer(store=store)
        canon.load()
        rt = _jev_choice("dosing-advice")
        gen = LLMTagGenerator(MockLLMProvider(response='{"tags": ["dosing-guidelines"], "primary": "dosing-guidelines"}'),
                              TagGeneratorConfig(type="llm", min_tags=1), canonicalizer=canon, judgment_runtime=rt)
        result = gen.generate_tags("how much should I dose", existing_tags=["dosing-accuracy", "dosing-advice", "gardening"])
        assert result.tags == ["dosing-advice"]
        # the alias never appears as a reuse candidate
        assert "dosing-accuracy" not in gen._reuse_candidates("dosing-guidelines", ["dosing-accuracy", "dosing-advice"], 5)
    finally:
        store.close()
        judgment.reset()


def test_alias_only_summaries_rank_with_their_canonical_topic(tmp_sqlite_db):
    from virtual_context.types import TagResult
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        store.store_segment(_seg("adv-0", ["dosing-advice"]))
        store.store_segment(_seg("acc-1", ["dosing-accuracy"]))  # alias-tagged only; never a query tag
        for i in range(4):  # a commoner tag scores below the dosing topic
            store.store_segment(_seg(f"gar-{i}", ["gardening"]))
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        retriever.config.strategy_configs["default"].max_results = 8
        retriever.tag_generator.set_override(
            "dosing", TagResult(tags=["dosing-advice", "gardening"], primary="dosing-advice", source="mock"))
        result = retriever.retrieve("dosing question")
    finally:
        store.close()
    order = [s.ref for s in result.summaries]
    assert order.index("acc-1") < min(order.index(f"gar-{i}") for i in range(4))
    assert result.retrieval_scores["dosing-accuracy"] == result.retrieval_scores["dosing-advice"] > result.retrieval_scores["gardening"]


def test_assembler_renders_alias_summaries_under_the_canonical_section(tmp_sqlite_db):
    from virtual_context.core.assembler import ContextAssembler
    from virtual_context.types import AssemblerConfig, RetrievalResult, StoredSummary

    def _summary(ref, tag):
        now = datetime.now(timezone.utc)
        return StoredSummary(ref=ref, primary_tag=tag, tags=[tag], summary="x" * 40, summary_tokens=10, full_tokens=40,
                             metadata=SegmentMetadata(), created_at=now, start_timestamp=now, end_timestamp=now)

    store = SQLiteStore(db_path=tmp_sqlite_db)
    try:
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        rr = RetrievalResult(tags_matched=["dosing-advice", "dosing-accuracy"],
                             summaries=[_summary("r0", "dosing-advice"), _summary("r1", "dosing-accuracy")], total_tokens=20)
        cfg = AssemblerConfig(core_context_max_tokens=1000, tag_context_max_tokens=5000)
        with_store = ContextAssembler(config=cfg, store=store).assemble(
            core_context="core", retrieval_result=rr, conversation_history=[], token_budget=10_000)
        without = ContextAssembler(config=cfg).assemble(
            core_context="core", retrieval_result=rr, conversation_history=[], token_budget=10_000)
    finally:
        store.close()
    assert set(without.tag_sections) == {"dosing-advice", "dosing-accuracy"}
    assert set(with_store.tag_sections) == {"dosing-advice"}
    assert with_store.presented_segment_refs == without.presented_segment_refs


def test_canonicalizer_refreshes_aliases_written_after_load(monkeypatch):
    aliases = {}
    store = SimpleNamespace(get_tag_aliases=lambda conversation_id=None: dict(aliases))
    canon = TagCanonicalizer(store, conversation_id="conv-A")
    canon.load()
    assert canon.canonicalize("dosing-accuracy") == "dosing-accuracy"
    aliases["dosing-accuracy"] = "dosing-advice"
    assert canon.canonicalize("dosing-accuracy") == "dosing-accuracy"  # cache still fresh
    monkeypatch.setattr(TagCanonicalizer, "ALIAS_REFRESH_S", 0.0001)
    import time
    time.sleep(0.001)
    assert canon.canonicalize("dosing-accuracy") == "dosing-advice"
    assert canon.get_aliases() == {"dosing-accuracy": "dosing-advice"}


def test_alias_chains_resolve_to_the_terminal_canonical(tmp_sqlite_db):
    from virtual_context.core.tag_canonicalizer import flatten_alias_map
    assert flatten_alias_map({"a": "b", "b": "c", "x": "x"}) == {"a": "c", "b": "c"}
    assert flatten_alias_map({"a": "b", "b": "a"}) == {}  # a cycle is not an alias group
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        store.set_tag_alias("dosing-notes", "dosing-accuracy")
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        assert retriever._load_alias_map() == {"dosing-notes": "dosing-advice", "dosing-accuracy": "dosing-advice"}
        canon = TagCanonicalizer(store)
        canon.load()
        assert canon.canonicalize("dosing-notes") == "dosing-advice"
    finally:
        store.close()


def test_each_topic_fetches_its_own_aliases(tmp_sqlite_db):
    from virtual_context.types import TagResult
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        for ref, tag in [("adv", "dosing-advice"), ("acc", "dosing-accuracy"), ("gar", "gardening"), ("gn", "garden-notes")]:
            store.store_segment(_seg(ref, [tag]))
        store.set_tag_alias("dosing-accuracy", "dosing-advice")
        store.set_tag_alias("garden-notes", "gardening")
        calls = []
        real = store.get_summaries_by_tags

        def recording(*args, **kwargs):
            calls.append(list(kwargs.get("tags") or (args[0] if args else [])))
            return real(*args, **kwargs)

        store.get_summaries_by_tags = recording
        retriever.tag_generator.set_override(
            "dosing", TagResult(tags=["dosing-advice", "gardening"], primary="dosing-advice", source="mock"))
        result = retriever.retrieve("dosing question")
    finally:
        store.close()
    assert ["dosing-accuracy"] in calls and ["garden-notes"] in calls  # one alias query per topic
    assert {"adv", "acc", "gar", "gn"} <= {s.ref for s in result.summaries}


def test_a_summary_behind_a_two_hop_chain_is_still_retrieved(tmp_sqlite_db):
    from virtual_context.types import TagResult
    retriever, store = _make_retriever(tmp_sqlite_db)
    try:
        store.store_segment(_seg("old-only", ["dosing-old"]))
        store.set_tag_alias("dosing-old", "dosing-mid")
        store.set_tag_alias("dosing-mid", "dosing-advice")
        retriever.tag_generator.set_override(
            "dosing", TagResult(tags=["dosing-old"], primary="dosing-old", source="mock"))
        result = retriever.retrieve("dosing question")
    finally:
        store.close()
    assert result.retrieval_metadata["top_tags"] == ["dosing-advice"]
    assert "old-only" in {s.ref for s in result.summaries}
