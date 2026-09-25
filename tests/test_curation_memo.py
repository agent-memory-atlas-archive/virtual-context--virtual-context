"""Fact curation is decided once per question and candidate set.

Every model call of a proxied tool loop resends the same user turn, so
curation received the same question and the same retrieved facts and paid
for the same judgment again on each call. The decision is now kept in shared
session state keyed by exactly those inputs, and memo entries stay alive
while a loop keeps reading them.
"""

from __future__ import annotations

from types import SimpleNamespace

import fakeredis
import pytest

from virtual_context.core.retrieval_assembler import RetrievalAssembler
from virtual_context.proxy.session_state import SessionStateProvider
from virtual_context.types import Fact


class _Curator:
    def __init__(self):
        self.calls = 0

    def curate(self, facts, question):
        self.calls += 1
        return [f for f in facts if "keep" in f.object]


def _facts(*objects):
    return [Fact(subject="s", verb="v", object=o) for o in objects]


def _assembler():
    provider = SessionStateProvider(
        redis_client=fakeredis.FakeRedis(decode_responses=False), store=None,
    )
    assembler = RetrievalAssembler.__new__(RetrievalAssembler)
    assembler._retriever = SimpleNamespace(
        _session_state_provider=provider, _conversation_id="conv",
    )
    assembler._fact_curator = _Curator()
    return assembler, provider


@pytest.mark.regression("BUG-096")
def test_the_same_question_and_facts_are_curated_once():
    assembler, _provider = _assembler()
    facts = _facts("keep a", "drop b", "keep c")
    first = assembler._curate_facts(facts, "question")
    second = assembler._curate_facts(_facts("keep a", "drop b", "keep c"), "question")
    assert [f.object for f in first] == ["keep a", "keep c"]
    assert [f.object for f in second] == ["keep a", "keep c"]
    assert assembler._fact_curator.calls == 1


@pytest.mark.regression("BUG-096")
@pytest.mark.parametrize("question,objects", [
    ("another question", ("keep a", "drop b", "keep c")),
    ("question", ("keep a", "drop b", "keep c", "keep d")),
    ("question", ("drop b", "keep a", "keep c")),
])
def test_a_different_question_or_candidate_set_is_curated_again(question, objects):
    assembler, _provider = _assembler()
    assembler._curate_facts(_facts("keep a", "drop b", "keep c"), "question")
    assembler._curate_facts(_facts(*objects), question)
    assert assembler._fact_curator.calls == 2


@pytest.mark.regression("BUG-096")
def test_without_shared_state_curation_still_runs():
    assembler, _provider = _assembler()
    assembler._retriever = SimpleNamespace(_session_state_provider=None, _conversation_id="conv")
    assembler._curate_facts(_facts("keep a"), "q")
    assembler._curate_facts(_facts("keep a"), "q")
    assert assembler._fact_curator.calls == 2


@pytest.mark.regression("BUG-096")
def test_a_memo_read_extends_its_life():
    provider = SessionStateProvider(
        redis_client=fakeredis.FakeRedis(decode_responses=False), store=None,
    )
    provider.save_retrieval_memo("conv", "k", {"x": 1}, ttl_seconds=5)
    key = provider._retrieval_memo_key("conv", "k")
    assert provider._redis.ttl(key) <= 5
    assert provider.load_retrieval_memo("conv", "k") == {"x": 1}
    assert provider._redis.ttl(key) > 5
