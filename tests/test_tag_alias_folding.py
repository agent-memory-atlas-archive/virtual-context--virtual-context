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
