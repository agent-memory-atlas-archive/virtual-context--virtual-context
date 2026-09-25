"""A tool loop's continuation drops the same history as the turn's first call.

The protected window of recent history left out the current turn only when
that turn held nothing but user messages. On a continuation the current turn
also carries the loop's tool calls and outputs, so it was counted as history,
the window moved back one turn, and one more old turn was dropped than on the
turn's first call, which changed the prefix the first call had cached.
"""

from __future__ import annotations

import copy

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import get_format
from virtual_context.proxy.message_filter import drop_compacted_turns


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _first_call(history_turns=10):
    items = [_msg("developer", "You are Vast.")]
    for i in range(history_turns):
        items += [_msg("user", f"question {i}"), _msg("assistant", f"answer {i}")]
    items.append(_msg("user", "grade this workout"))
    return {"model": "m", "input": items}


def _continuation(rounds):
    body = _first_call()
    for i in range(rounds):
        body["input"] += [
            {"type": "function_call", "call_id": f"c{i}", "name": "exec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": f"out {i}"},
        ]
    return body


def _drop(body):
    body = copy.deepcopy(body)
    return drop_compacted_turns(
        body, TurnTagIndex(), 1000, fmt=get_format("openai_responses"),
        protected_recent_turns=6, drop_boundary=1000,
    )


@pytest.mark.regression("BUG-103")
@pytest.mark.parametrize("rounds", [1, 3])
def test_a_continuation_drops_what_the_first_call_dropped(rounds):
    first, first_count = _drop(_first_call())
    later, later_count = _drop(_continuation(rounds))
    assert later_count == first_count
    assert later["input"][: len(first["input"])] == first["input"]
