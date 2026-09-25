"""A tool loop's continuation calls keep the payload shape of the turn's first call.

The flush gate lets the first call of a turn reshape the payload when the
prompt cache is cold (collapsing chains, stubbing old outputs) and holds all
reshaping while it is warm. The first call's request makes the cache warm, so
every continuation of the same turn was sent without the reshaping the first
call applied, and the second call of each such turn missed the cache. The
first call's decision is now kept per turn and replayed by its continuations.
"""

from __future__ import annotations

import fakeredis
import pytest

from virtual_context.proxy.flush_gate import recall_turn_gate, remember_turn_gate, turn_gate_key
from virtual_context.proxy.session_state import SessionStateProvider


def _provider():
    return SessionStateProvider(redis_client=fakeredis.FakeRedis(decode_responses=False), store=None)


@pytest.mark.regression("BUG-101")
def test_a_continuation_recalls_the_first_calls_flush_boundary():
    provider = _provider()
    key = turn_gate_key("grade this workout", 12)
    assert recall_turn_gate(provider, "conv", key) is None
    remember_turn_gate(provider, "conv", key, flushed_prefix=7184)
    assert recall_turn_gate(provider, "conv", key) == 7184


@pytest.mark.regression("BUG-101")
@pytest.mark.parametrize("text,turns", [("grade this workout", 13), ("another question", 12)])
def test_another_turn_does_not_reuse_the_decision(text, turns):
    provider = _provider()
    remember_turn_gate(provider, "conv", turn_gate_key("grade this workout", 12), flushed_prefix=7184)
    assert recall_turn_gate(provider, "conv", turn_gate_key(text, turns)) is None
    assert recall_turn_gate(provider, "other", turn_gate_key("grade this workout", 12)) is None


@pytest.mark.regression("BUG-101")
def test_without_shared_state_there_is_nothing_to_recall():
    remember_turn_gate(None, "conv", "k", flushed_prefix=1)
    assert recall_turn_gate(None, "conv", "k") is None
    assert turn_gate_key("", 3) == ""
