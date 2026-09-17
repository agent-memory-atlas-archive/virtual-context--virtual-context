import json
import logging

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime, rerank_summaries, spearman
from virtual_context.types import JudgmentConfig, StoredSummary


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _s(ref, text, tokens=100):
    return StoredSummary(ref=ref, primary_tag=ref, tags=[ref], summary=text, summary_tokens=tokens)


def _runtime(mode, probs_by_key, **cfg):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        answers = {k: {"type": "noul", "noul": probs_by_key.get(k, 0.0)} for k in body["questions"]}
        return httpx.Response(200, json={"model": "jev-t", "answers": answers,
                                         "usage": {"input_tokens": 5, "output_tokens": 1}})
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return build_runtime(JudgmentConfig(mode=mode, **cfg), environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


def test_spearman():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_legacy_mode_returns_same_list_object_untouched():
    items = [_s("a", "A"), _s("b", "B")]
    assert rerank_summaries("q", items) is items


def test_fewer_than_two_candidates_never_calls_jev():
    rt, seen = _runtime("jev", {"c0": 0.1})
    items = [_s("a", "A")]
    assert rerank_summaries("q", items, runtime=rt) == items
    assert seen == []


def test_jev_mode_sorts_by_probability_stable_on_ties():
    rt, seen = _runtime("jev", {"c0": 0.2, "c1": 0.9, "c2": 0.9, "c3": 0.5})
    items = [_s("a", "A"), _s("b", "B"), _s("c", "C"), _s("d", "D")]
    out = rerank_summaries("what happened", items, runtime=rt)
    assert [s.ref for s in out] == ["b", "c", "d", "a"]
    body = seen[0]
    assert body["state"]["query"] == "what happened"
    assert body["state"]["candidates"] == {"c0": "A", "c1": "B", "c2": "C", "c3": "D"}
    assert set(body["questions"]) == {"c0", "c1", "c2", "c3"}
    assert "candidates.c2" in body["questions"]["c2"]["instructions"]


def test_min_probability_moves_low_candidates_to_the_end_without_dropping():
    rt, _ = _runtime("jev", {"c0": 0.05, "c1": 0.9, "c2": 0.04}, rerank_min_probability=0.1)
    items = [_s("a", "A"), _s("b", "B"), _s("c", "C")]
    out = rerank_summaries("q", items, runtime=rt)
    assert [s.ref for s in out] == ["b", "a", "c"]


def test_state_size_guard_skips_jev(caplog):
    rt, seen = _runtime("jev", {"c0": 0.9}, rerank_max_state_bytes=50)
    items = [_s("a", "x" * 100), _s("b", "y" * 100)]
    with caplog.at_level(logging.INFO):
        out = rerank_summaries("q", items, runtime=rt)
    assert [s.ref for s in out] == ["a", "b"] and seen == []
    assert any("JUDGMENT_SKIP seam=rerank reason=state_size" in r.message for r in caplog.records)


def test_shadow_keeps_legacy_order_and_logs_rank_metrics(caplog):
    rt, seen = _runtime("shadow", {"c0": 0.1, "c1": 0.9})
    items = [_s("a", "A"), _s("b", "B")]
    with caplog.at_level(logging.INFO):
        out = rerank_summaries("q", items, runtime=rt)
    assert [s.ref for s in out] == ["a", "b"] and len(seen) == 1
    line = next(r.message for r in caplog.records if "JUDGMENT_SHADOW seam=rerank" in r.message)
    assert "agree=False" in line and "spearman=-1.0" in line and "top1_legacy=a" in line and "top1_jev=b" in line


def test_jev_failure_keeps_legacy_order():
    def boom(request):
        return httpx.Response(500)
    http = httpx.Client(transport=httpx.MockTransport(boom))
    rt = build_runtime(JudgmentConfig(mode="jev"), environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    items = [_s("a", "A"), _s("b", "B")]
    assert [s.ref for s in rerank_summaries("q", items, runtime=rt)] == ["a", "b"]
