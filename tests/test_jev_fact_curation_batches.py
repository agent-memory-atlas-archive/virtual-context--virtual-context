"""Fact curation splits facts into token-budgeted batches sent together.

A single judgment request over about five hundred facts can exceed the
judgment service's input token limit and be rejected outright. Facts are
judged one question each, so splitting them changes no answer.
"""

from __future__ import annotations

import threading

import pytest

from virtual_context.core.judgment import JevResponse, _parse_answers, jev_fact_curation
from virtual_context.types import JudgmentConfig


class _Client:
    def __init__(self, budget, *, fail_batch=None):
        self.config = JudgmentConfig(curation_batch_tokens=budget)
        self.batches = []
        self.fail_batch = fail_batch
        self._lock = threading.Lock()
        self._second = threading.Event()

    def ask(self, *, seam, state, questions):
        with self._lock:
            self.batches.append(sorted(int(k) for k in state["facts"]))
            index = len(self.batches)
        if index == 1:
            # Returns only once another batch is in flight: batches run together.
            assert self._second.wait(5), "batches were sent one after another"
        else:
            self._second.set()
        if self.fail_batch is not None and 0 in self.batches[index - 1] and self.fail_batch == 0:
            return None
        answers = {k: {"type": "noul", "noul": 0.9 if int(k.split("__")[1]) % 2 == 0 else 0.1} for k in questions}
        return JevResponse(answers=_parse_answers(answers), model="jev", input_tokens=1, output_tokens=1, latency_ms=1.0)


FACTS = [f"user logged measurement number {i} " + "x" * 200 for i in range(40)]


@pytest.mark.regression("BUG-083")
def test_facts_over_the_budget_are_split_and_every_fact_is_judged():
    client = _Client(budget=2_000)
    out = jev_fact_curation(client, "what did I log", FACTS)
    assert len(client.batches) > 1
    assert sorted(i for batch in client.batches for i in batch) == list(range(len(FACTS)))
    assert out.value == {i: (0.9 if i % 2 == 0 else 0.1) for i in range(len(FACTS))}


@pytest.mark.regression("BUG-083")
def test_facts_within_the_budget_go_in_one_call():
    client = _Client(budget=1_000_000)
    client._second.set()
    jev_fact_curation(client, "what did I log", FACTS)
    assert client.batches == [list(range(len(FACTS)))]


@pytest.mark.regression("BUG-083")
def test_a_failed_batch_fails_the_whole_curation():
    client = _Client(budget=2_000, fail_batch=0)
    assert jev_fact_curation(client, "what did I log", FACTS) is None

