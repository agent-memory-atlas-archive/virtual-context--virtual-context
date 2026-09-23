"""Seam tag_select: the judgment model picks a turn's tags from existing topics.

Candidates are the topics whose summaries are nearest the turn plus the tags
whose names are nearest it. The model keeps the ones the turn is about and
near-synonyms collapse to the most likely one. The tagging model still reads
every turn, because it also extracts the turn's fact signals and code
references; its new tags are kept only when the judgment model says a subject
has no tag yet.
"""

from __future__ import annotations

import hashlib
import json
import logging

import httpx
import numpy as np
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime
from virtual_context.core.tag_generator import LLMTagGenerator
from tests.conftest import MockLLMProvider
from virtual_context.types import JUDGMENT_SEAMS, JudgmentConfig, TagGeneratorConfig

EXISTING = ["hcg-dosing", "hcg", "vitamin-d", "squat", "deployment", "core-work", "core"]
LLM_TAGS = json.dumps({
    "tags": ["hcg-kit-duration", "vitamin-d"], "primary": "hcg-kit-duration", "temporal": False,
    "fact_signals": [{"subject": "user", "verb": "doses", "object": "hcg", "status": "active",
                      "fact_type": "personal", "what": "500 IU twice a week"}],
})

# Hand-set name vectors: "hcg" and "hcg-dosing" are near-synonyms, as are "core" and "core-work".
_NAMED = {
    "hcg": [1.0, 0.0, 0.0, 0.0], "hcg dosing": [0.98, 0.2, 0.0, 0.0], "hcg-dosing": [0.98, 0.2, 0.0, 0.0],
    "vitamin d": [0.0, 1.0, 0.0, 0.0], "vitamin-d": [0.0, 1.0, 0.0, 0.0],
    "core": [0.0, 0.0, 1.0, 0.0], "core work": [0.0, 0.1, 0.99, 0.0], "core-work": [0.0, 0.1, 0.99, 0.0],
}


def _vec(text: str) -> list[float]:
    if text in _NAMED:
        return list(_NAMED[text])
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:4]]


def _embed(texts):
    return [_vec(t) for t in texts]


def _topics():
    tags = ["hcg-dosing", "vitamin-d", "squat", "core-work"]
    matrix = np.asarray([_vec(t) for t in tags], dtype=np.float32)
    return tags, matrix / np.linalg.norm(matrix, axis=1)[:, None]


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _runtime(mode, probs, *, new_topic=0.1, fail=False):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if fail:
            return httpx.Response(500)
        answers = {}
        for key in body["questions"]:
            if key == "new_topic":
                answers[key] = {"type": "noul", "noul": new_topic}
            else:
                tag = body["state"]["tags"][key.split("__")[1]]
                answers[key] = {"type": "noul", "noul": probs.get(tag, 0.05)}
        return httpx.Response(200, json={"model": "jev-t", "answers": answers,
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    config = JudgmentConfig(mode="legacy", seams={"tag_select": mode})
    return build_runtime(config, environ={"TYPESAFE_API_KEY": "k"}, http_client=http), seen


class _CountingLLM(MockLLMProvider):
    def __init__(self):
        super().__init__(response=LLM_TAGS)
        self.n = 0

    def complete(self, *args, **kwargs):
        self.n += 1
        return super().complete(*args, **kwargs)


def _tagger(rt):
    llm = _CountingLLM()
    tagger = LLMTagGenerator(
        llm, TagGeneratorConfig(type="llm", min_tags=1), judgment_runtime=rt,
        embed_fn_factory=lambda: _embed, load_topic_matrix=_topics,
    )
    return tagger, llm


def test_tag_select_is_a_registered_seam():
    assert "tag_select" in JUDGMENT_SEAMS


def test_live_selection_replaces_the_tag_list_and_keeps_the_rest_of_the_result():
    rt, seen = _runtime("jev", {"hcg-dosing": 0.95, "hcg": 0.9, "vitamin-d": 0.7})
    tagger, llm = _tagger(rt)
    result = tagger.generate_tags("how long does a kit of hcg last at my dose", existing_tags=EXISTING)
    # "hcg" is a near-synonym of the likelier "hcg-dosing" and collapses into it.
    assert result.tags == ["hcg-dosing", "vitamin-d"]
    assert result.primary == "hcg-dosing"
    assert llm.n == 1
    assert [f.object for f in result.fact_signals] == ["hcg"]
    offered = set(seen[0]["state"]["tags"].values())
    assert {"hcg-dosing", "vitamin-d", "squat", "core-work"} <= offered


def test_a_subject_with_no_tag_keeps_the_tagging_model_new_tag():
    rt, _ = _runtime("jev", {"hcg-dosing": 0.95}, new_topic=0.9)
    tagger, llm = _tagger(rt)
    result = tagger.generate_tags("how long does a kit last", existing_tags=EXISTING)
    assert llm.n == 1
    assert result.tags == ["hcg-dosing", "hcg-kit-duration"]


def test_shadow_keeps_the_tagging_model_answer_and_logs_the_selection(caplog):
    rt, seen = _runtime("shadow", {"hcg-dosing": 0.95})
    tagger, llm = _tagger(rt)
    with caplog.at_level(logging.INFO, logger="virtual_context.core.judgment"):
        result = tagger.generate_tags("how long does a kit last", existing_tags=EXISTING)
    assert llm.n == 1 and seen
    assert "hcg-kit-duration" in result.tags
    assert any("JUDGMENT_SHADOW seam=tag_select" in r.getMessage() and "hcg-dosing" in r.getMessage()
               for r in caplog.records)


def test_a_failed_judgment_falls_back_to_the_tagging_model():
    rt, _ = _runtime("jev", {}, fail=True)
    tagger, llm = _tagger(rt)
    result = tagger.generate_tags("how long does a kit last", existing_tags=EXISTING)
    assert llm.n == 1
    assert "hcg-kit-duration" in result.tags


def test_nothing_judged_relevant_falls_back_to_the_tagging_model():
    rt, _ = _runtime("jev", {}, new_topic=0.1)
    tagger, llm = _tagger(rt)
    result = tagger.generate_tags("how long does a kit last", existing_tags=EXISTING)
    assert llm.n == 1
    assert "hcg-kit-duration" in result.tags


def test_legacy_mode_never_asks_the_judgment_model():
    rt, seen = _runtime("legacy", {"hcg-dosing": 0.95})
    tagger, llm = _tagger(rt)
    tagger.generate_tags("how long does a kit last", existing_tags=EXISTING)
    assert seen == [] and llm.n == 1


def test_selection_keeps_at_most_the_configured_number_of_likeliest_tags():
    probs = {"squat": 0.99, "vitamin-d": 0.9, "deployment": 0.8, "core-work": 0.7}
    rt, _ = _runtime("jev", probs)
    rt.config.tag_select_max_tags = 2
    tagger, _ = _tagger(rt)
    result = tagger.generate_tags("squat and vitamin d", existing_tags=EXISTING)
    assert result.tags == ["squat", "vitamin-d"]


@pytest.mark.regression("BUG-086")
def test_no_kept_topic_and_no_new_tag_keeps_the_tagging_model_result():
    # The judgment model keeps nothing but reports an untagged subject, while
    # every tag the tagging model proposed already exists: nothing is left to select.
    rt, _ = _runtime("jev", {}, new_topic=0.9)
    tagger, llm = _tagger(rt)
    tagger.llm.response = json.dumps({"tags": ["vitamin-d", "squat"], "primary": "vitamin-d", "temporal": False})
    result = tagger.generate_tags("vitamin d and squats", existing_tags=EXISTING)
    assert result.tags == ["vitamin-d", "squat"] and result.primary == "vitamin-d"
