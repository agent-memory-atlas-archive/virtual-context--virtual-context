"""Host integration for seams S6-S11: each call site honours the runtime it is handed."""
import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.compactor import DomainCompactor
from virtual_context.core.judgment import build_runtime
from virtual_context.core.tag_consolidator import consolidate_tags
from virtual_context.core.tag_generator import LLMTagGenerator, build_tag_generator
from virtual_context.core.tag_splitter import TagSplitter
from virtual_context.ingest.curator import FactCurator
from virtual_context.ingest.supersession import FactLinkChecker, FactSupersessionChecker
from virtual_context.types import (
    CompactorConfig, CurationConfig, Fact, JudgmentConfig, SupersessionConfig, TagGeneratorConfig,
    TagSplittingConfig, TagStats,
)

from conftest import MockLLMProvider


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _runtime(mode, answers_fn, **cfg):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"model": "jev-t", "answers": answers_fn(body),
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    rt = build_runtime(JudgmentConfig(mode=mode, **cfg), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    return rt, seen


def _choice_all(choice, confidence=0.9):
    def answers(body):
        out = {}
        for key, q in body["questions"].items():
            opts = list(q["criteria"])
            out[key] = {"type": "choice", "choice": choice, "confidence": confidence,
                        "probabilities": {o: (confidence if o == choice else 0.0) for o in opts}}
        return out
    return answers


def _noul_all(p):
    return lambda body: {k: {"type": "noul", "noul": p} for k in body["questions"]}


# --- tag reuse in the LLM tag generator --------------------------------------

_TAG_RESPONSE = '{"tags": ["data-visualization-tools", "sales-reporting"], "primary": "data-visualization-tools"}'


def _tagger(rt):
    return LLMTagGenerator(MockLLMProvider(response=_TAG_RESPONSE), TagGeneratorConfig(type="llm", min_tags=1),
                           judgment_runtime=rt)


def test_tag_generator_jev_replaces_new_tag_with_existing():
    rt, seen = _runtime("jev", _choice_all("data-visualization"))
    result = _tagger(rt).generate_tags("charts about sales", existing_tags=["data-visualization", "deployment"])
    assert result.tags[0] == "data-visualization" and result.primary == "data-visualization"
    assert "data-visualization-tools" not in result.tags
    assert seen[0]["state"]["proposed_tags"]["0"] == "data-visualization-tools"
    assert "data-visualization" in seen[0]["state"]["existing_tags"]["0"]


def test_tag_generator_shadow_keeps_tags_and_logs(caplog):
    rt, seen = _runtime("shadow", _choice_all("data-visualization"))
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        result = _tagger(rt).generate_tags("charts about sales", existing_tags=["data-visualization"])
    assert result.tags[0] == "data-visualization-tools"
    assert seen and any("JUDGMENT_SHADOW seam=tag_reuse" in r.getMessage() for r in caplog.records)


def test_tag_generator_legacy_never_calls_jev():
    rt, seen = _runtime("legacy", _choice_all("data-visualization"))
    result = _tagger(rt).generate_tags("charts about sales", existing_tags=["data-visualization"])
    assert result.tags[0] == "data-visualization-tools" and seen == []


def test_tag_generator_skips_reuse_when_no_existing_tags():
    rt, seen = _runtime("jev", _choice_all("data-visualization"))
    _tagger(rt).generate_tags("charts about sales", existing_tags=[])
    assert seen == []


def test_build_tag_generator_threads_runtime():
    rt, _ = _runtime("jev", _choice_all("x"))
    gen = build_tag_generator(TagGeneratorConfig(type="llm"), MockLLMProvider(), judgment_runtime=rt)
    assert gen._judgment_runtime is rt


# --- tag split ---------------------------------------------------------------

_SPLIT_YES = json.dumps({"splittable": True, "groups": {"cooking-pasta": [1], "car-repair": [2]}})


def _splitter(rt):
    llm = MagicMock()
    llm.complete.return_value = (_SPLIT_YES, {})
    return TagSplitter(llm=llm, config=TagSplittingConfig(enabled=True), judgment_runtime=rt), llm


def test_tag_splitter_jev_single_topic_skips_llm():
    rt, seen = _runtime("jev", _noul_all(0.1))
    splitter, llm = _splitter(rt)
    result = splitter.split("cooking", [(1, "pasta"), (2, "car repair")], set(), 10)
    assert result.splittable is False and llm.complete.call_count == 0
    assert seen[0]["state"]["tag"] == "cooking"


def test_tag_splitter_jev_multi_topic_runs_llm():
    rt, _ = _runtime("jev", _noul_all(0.9))
    splitter, llm = _splitter(rt)
    result = splitter.split("cooking", [(1, "pasta"), (2, "car repair")], set(), 10)
    assert result.splittable is True and llm.complete.call_count == 1


def test_tag_splitter_shadow_calls_llm_once(caplog):
    rt, _ = _runtime("shadow", _noul_all(0.1))
    splitter, llm = _splitter(rt)
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        result = splitter.split("cooking", [(1, "pasta"), (2, "car repair")], set(), 10)
    assert result.splittable is True and llm.complete.call_count == 1
    assert any("JUDGMENT_SHADOW seam=tag_split agree=False" in r.getMessage() for r in caplog.records)


# --- supersession ------------------------------------------------------------

def _fact(id, object, session_date=""):
    return Fact(id=id, conversation_id="c", subject="user", verb="lives-in", object=object, session_date=session_date)


def test_supersession_checker_jev_uses_relations_not_llm():
    rt, seen = _runtime("jev", _choice_all("supersedes"))
    llm = MagicMock()
    llm.complete.return_value = ("[]", {})
    checker = FactSupersessionChecker(llm_provider=llm, model="m", store=MagicMock(),
                                      config=SupersessionConfig(enabled=True), judgment_runtime=rt)
    ids = checker._check_batch(_fact("n", "Seattle", "2024/02/01"), [_fact("o", "Boston", "2023/01/01")])
    assert ids == ["o"] and llm.complete.call_count == 0
    assert seen[0]["state"]["candidates"]["o"]["session_date"] == "2023/01/01"
    assert seen[0]["state"]["new_fact_session_date"] == "2024/02/01"


def test_supersession_checker_shadow_returns_llm_answer(caplog):
    rt, _ = _runtime("shadow", _choice_all("supersedes"))
    llm = MagicMock()
    llm.complete.return_value = ("[]", {})
    checker = FactSupersessionChecker(llm_provider=llm, model="m", store=MagicMock(),
                                      config=SupersessionConfig(enabled=True), judgment_runtime=rt)
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        ids = checker._check_batch(_fact("n", "Seattle"), [_fact("o", "Boston")])
    assert ids == [] and llm.complete.call_count == 1
    assert any("JUDGMENT_SHADOW seam=supersession agree=False" in r.getMessage() for r in caplog.records)


def test_fact_link_checker_jev_builds_links():
    rt, _ = _runtime("jev", _choice_all("duplicates"))
    llm = MagicMock()
    llm.complete.return_value = ('{"superseded": [], "links": []}', {})
    checker = FactLinkChecker(llm_provider=llm, model="m", store=MagicMock(), config=SupersessionConfig(enabled=True),
                              graph_links=True, judgment_runtime=rt)
    links, superseded = checker._check_links(_fact("n", "Seattle"), [_fact("o", "Seattle, WA")])
    assert superseded == ["o"] and llm.complete.call_count == 0
    assert [(l.source_fact_id, l.target_fact_id, l.relation_type) for l in links] == [("n", "o", "same_as")]
    assert checker._supersession._judgment_runtime is rt


# --- fact curation -----------------------------------------------------------

def test_curator_jev_filters_without_llm():
    def answers(body):
        return {k: {"type": "noul", "noul": (0.9 if k.endswith("__1") else 0.05)} for k in body["questions"]}
    rt, seen = _runtime("jev", answers)
    llm = MockLLMProvider(response="0")
    curator = FactCurator(llm_provider=llm, model="m", config=CurationConfig(enabled=True), judgment_runtime=rt)
    facts = [Fact(subject="user", verb="hiked", object="Dipsea"), Fact(subject="user", verb="lives-in", object="Seattle")]
    out = curator.curate(facts, "where do I live?")
    assert out == [facts[1]] and llm.calls == []
    assert seen[0]["state"]["question"] == "where do I live?"


def test_curator_jev_keeps_no_facts_when_none_are_relevant():
    rt, _ = _runtime("jev", _noul_all(0.05))
    llm = MockLLMProvider(response="0")
    curator = FactCurator(llm_provider=llm, model="m", config=CurationConfig(enabled=True), judgment_runtime=rt)
    facts = [Fact(subject="user", verb="hiked", object="Dipsea"), Fact(subject="user", verb="lives-in", object="Seattle")]
    assert curator.curate(facts, "what is the capital of France?") == [] and llm.calls == []


def test_curator_shadow_uses_llm_and_logs(caplog):
    rt, _ = _runtime("shadow", _noul_all(0.9))
    llm = MockLLMProvider(response="0")
    curator = FactCurator(llm_provider=llm, model="m", config=CurationConfig(enabled=True), judgment_runtime=rt)
    facts = [Fact(subject="user", verb="hiked", object="Dipsea"), Fact(subject="user", verb="lives-in", object="Seattle")]
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        out = curator.curate(facts, "q")
    assert out == [facts[0]] and len(llm.calls) == 1
    assert any("JUDGMENT_SHADOW seam=fact_curation agree=False" in r.getMessage() for r in caplog.records)


# --- tag consolidation -------------------------------------------------------

def _store(tags):
    store = MagicMock()
    store.get_all_tags.return_value = [TagStats(tag=t, usage_count=n) for t, n in tags]
    store.get_tag_aliases.return_value = {}
    return store


def test_consolidate_tags_jev_groups_without_llm():
    def answers(body):
        out = {}
        for key in body["questions"]:
            pair = set(body["state"]["pairs"][key.split("__", 1)[1]])
            out[key] = {"type": "noul", "noul": 0.9 if pair == {"model-kit", "model-tanks"} else 0.1}
        return out
    rt, _ = _runtime("jev", answers)
    llm = MockLLMProvider(response='{"groups": []}')
    result = consolidate_tags(_store([("model-kit", 5), ("model-tanks", 2), ("data-model", 9)]), llm,
                              dry_run=True, judgment_runtime=rt)
    assert [(g.canonical, g.aliases) for g in result.groups] == [("model-kit", ["model-tanks"])]
    assert result.groups[0].reason.startswith("jev:") and llm.calls == []


def test_consolidate_tags_shadow_keeps_llm_groups(caplog):
    rt, _ = _runtime("shadow", _noul_all(0.1))
    llm = MockLLMProvider(response='{"groups": [{"canonical": "model-kit", "aliases": ["model-tanks"], "reason": "hobby"}]}')
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        result = consolidate_tags(_store([("model-kit", 5), ("model-tanks", 2)]), llm, dry_run=True, judgment_runtime=rt)
    assert [(g.canonical, g.aliases) for g in result.groups] == [("model-kit", ["model-tanks"])]
    assert len(llm.calls) == 1
    assert any("JUDGMENT_SHADOW seam=tag_consolidation agree=False" in r.getMessage() for r in caplog.records)


# --- summary grounding in the compactor --------------------------------------

def test_compactor_reject_reason_adds_jev_ungrounded_in_jev_mode():
    rt, seen = _runtime("jev", _noul_all(0.2))
    compactor = DomainCompactor(MockLLMProvider(), CompactorConfig(), judgment_runtime=rt)
    source = "User: I moved to Paris last spring. Assistant: Congratulations on the move."
    assert compactor._judged_summary_reject_reason("User moved to Paris.", source, None) == "jev_ungrounded"
    assert seen[0]["state"] == {"summary": "User moved to Paris.", "source": source}
    assert compactor._judged_summary_reject_reason("", source, None) == "degenerate"


def test_compactor_reject_reason_shadow_keeps_heuristic(caplog):
    rt, seen = _runtime("shadow", _noul_all(0.2))
    compactor = DomainCompactor(MockLLMProvider(), CompactorConfig(), judgment_runtime=rt)
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        assert compactor._judged_summary_reject_reason("User moved to Paris.", "User: I moved to Paris last spring, it was great.", None) is None
    assert any("JUDGMENT_SHADOW seam=summary_grounding agree=False" in r.getMessage() for r in caplog.records)


def test_compactor_reject_reason_legacy_never_calls_jev():
    rt, seen = _runtime("legacy", _noul_all(0.2))
    compactor = DomainCompactor(MockLLMProvider(), CompactorConfig(), judgment_runtime=rt)
    assert compactor._judged_summary_reject_reason("s", "a longer source text than the summary", None) is None
    assert seen == []
