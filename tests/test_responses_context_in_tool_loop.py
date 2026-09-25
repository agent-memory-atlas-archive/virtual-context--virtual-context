"""VC's context block stays in place while a Responses tool loop runs.

The block was appended after every input item, so each round of a tool loop
moved it behind that round's new calls and outputs. The block is the same
for every round of a turn, yet it and everything after it were re-billed on
each call. It now sits right after the latest user message, ahead of the
turn's tool traffic, so an unchanged block is part of the cached prefix.
"""

from __future__ import annotations

import copy
import json

import pytest

from virtual_context.core.provider_adapters import OpenAICodexAdapter
from virtual_context.proxy.formats import get_format


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _round(i):
    return [
        {"type": "function_call", "call_id": f"c{i}", "name": "exec", "arguments": f'{{"n": {i}}}'},
        {"type": "function_call_output", "call_id": f"c{i}", "output": f"output {i}"},
    ]


def _loop_request(rounds):
    items = [_msg("developer", "You are Vast."), _msg("user", "earlier"), _msg("assistant", "ok"),
             _msg("user", "grade this workout")]
    for i in range(rounds):
        items += _round(i)
    return {"model": "m", "instructions": "Be concise.", "input": items}


def _proxy(body, text):
    return get_format("openai_responses").inject_context(body, text)


def _adapter(body, text):
    body = copy.deepcopy(body)
    OpenAICodexAdapter(api_key="k").inject_context(body, text)
    return body


@pytest.mark.regression("BUG-100")
@pytest.mark.parametrize("inject", [_proxy, _adapter])
def test_each_round_of_a_loop_extends_the_previous_request(inject):
    calls = [inject(_loop_request(n), "facts: deadlift TM 365") for n in range(4)]
    for earlier, later in zip(calls, calls[1:]):
        a = json.dumps(earlier["input"])[:-1]
        assert json.dumps(later["input"]).startswith(a)


@pytest.mark.regression("BUG-100")
@pytest.mark.parametrize("inject", [_proxy, _adapter])
def test_the_block_follows_the_latest_user_message_once(inject):
    body = inject(inject(_loop_request(2), "old"), "new")
    roles = [(i.get("role"), i.get("type")) for i in body["input"]]
    at = roles.index(("user", "message"), 3)
    block = body["input"][at + 1]
    assert block["role"] == "developer" and "new" in block["content"][0]["text"]
    assert sum("<system-reminder>" in json.dumps(i) for i in body["input"]) == 1
    assert [i.get("type") for i in body["input"][at + 2:]] == ["function_call", "function_call_output"] * 2
