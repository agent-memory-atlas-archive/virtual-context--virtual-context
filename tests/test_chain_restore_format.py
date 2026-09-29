"""Restored turns are rewritten into the request's own format.

A collapsed turn is stored exactly as its host sent it, which may be another
wire format than the request that later restores it, and may carry host-only
fields. Restoring a chat-format chain with OpenClaw's own fields into a
Responses request produced items the provider rejects (``Unknown parameter:
'input[3].idempotencyKey'``). The chain is now converted to the request's
format and reduced to the fields that format accepts; a chain that cannot be
converted is returned to the model as text instead of spliced in.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from virtual_context.proxy.chain_restore import conform_chain
from virtual_context.proxy.handlers import _ProxyToolRuntime

REF = "chain_3508_31064b619735"
HOST = {"__openclaw": {"seq": 7}, "idempotencyKey": "k-1"}

CHAT_CHAIN = [
    {**HOST, "role": "user", "content": "check the record"},
    {**HOST, "role": "assistant", "api": "openai-responses", "model": "gpt-5.6-sol", "provider": "openai",
     "content": "Reading it now.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "exec", "arguments": "{\"cmd\": \"cat r.json\"}"}}]},
    {**HOST, "role": "tool", "tool_call_id": "call_1", "content": "{\"tm\": 365}"},
    {**HOST, "role": "assistant", "api": "openai-responses", "model": "gpt-5.6-sol", "provider": "openai",
     "content": "Training max is 365."},
]

RESPONSES_ITEM_KEYS = {
    "message": {"type", "role", "content"},
    "function_call": {"type", "call_id", "name", "arguments"},
    "function_call_output": {"type", "call_id", "output"},
}


def _stub_pair():
    return [
        {"role": "user", "content": "[Compacted turn 3508]"},
        {"role": "assistant", "content": [{"type": "output_text", "text": f"[Compacted turn 3508 | vc_restore_tool ref {REF}]"}]},
    ]


def _runtime(body, api_format, chain):
    store = SimpleNamespace(
        get_chain_snapshot=lambda conv, ref: {"chain_json": json.dumps(chain), "tool_output_refs": ""} if ref == REF else None,
    )
    return _ProxyToolRuntime(SimpleNamespace(_store=store), api_format, "conv", lambda: body)


@pytest.mark.regression("BUG-106")
def test_a_chat_chain_restores_as_valid_responses_items():
    body = {"model": "m", "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
                                    *_stub_pair(),
                                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now"}]}]}
    result = _runtime(body, "openai_responses", CHAT_CHAIN).restore_tool_output(REF)
    assert isinstance(result, str) and result.startswith("Restored.")
    restored = body["input"][1:-1]
    serialized = json.dumps(restored)
    for field in ("idempotencyKey", "__openclaw", "\"api\"", "\"provider\"", "tool_calls", "tool_call_id"):
        assert field not in serialized
    for item in restored:
        assert set(item) <= RESPONSES_ITEM_KEYS[item["type"]]
    types = [item["type"] for item in restored]
    assert types == ["message", "message", "function_call", "function_call_output", "message"]
    call, output = restored[2], restored[3]
    assert call["call_id"] == output["call_id"] == "call_1"
    assert call["name"] == "exec" and json.loads(call["arguments"]) == {"cmd": "cat r.json"}
    assert "Training max is 365." in serialized


@pytest.mark.regression("BUG-106")
def test_responses_items_keep_only_accepted_fields_and_drop_reasoning():
    chain = [
        {"type": "message", "role": "user", "id": "msg_1", "idempotencyKey": "x", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
        {"type": "function_call", "id": "fc_1", "call_id": "c9", "name": "exec", "arguments": "{}", "status": "completed"},
        {"type": "function_call_output", "call_id": "c9", "output": "ok", "__openclaw": {}},
        {"type": "function_call", "call_id": "orphan", "name": "exec", "arguments": "{}"},
    ]
    out = conform_chain(chain, "openai_responses")
    assert [item["type"] for item in out] == ["message", "function_call", "function_call_output"]
    for item in out:
        assert set(item) <= RESPONSES_ITEM_KEYS[item["type"]]


@pytest.mark.regression("BUG-106")
def test_an_anthropic_chain_converts_to_responses_items():
    chain = [
        {"role": "user", "content": [{"type": "text", "text": "check"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "..", "signature": "s"},
                                          {"type": "text", "text": "Reading."},
                                          {"type": "tool_use", "id": "tu_1", "name": "exec", "input": {"cmd": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "a.txt"}]}]},
    ]
    out = conform_chain(chain, "openai_responses")
    assert [item["type"] for item in out] == ["message", "message", "function_call", "function_call_output"]
    assert out[3]["output"] == "a.txt" and out[2]["call_id"] == out[3]["call_id"] == "tu_1"


@pytest.mark.regression("BUG-106")
def test_host_fields_are_removed_for_a_matching_chat_request():
    out = conform_chain(CHAT_CHAIN, "openai")
    assert out is not None
    assert "idempotencyKey" not in json.dumps(out) and "__openclaw" not in json.dumps(out)
    assert out[2] == {"role": "tool", "tool_call_id": "call_1", "content": "{\"tm\": 365}"}


@pytest.mark.regression("BUG-106")
def test_an_unconvertible_chain_is_returned_as_text_and_the_payload_is_untouched():
    responses_chain = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old question"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "old answer"}]},
    ]
    body = {"model": "claude", "system": "s", "messages": [
        {"role": "user", "content": "[Compacted turn 3508]"},
        {"role": "assistant", "content": [{"type": "text", "text": f"[Compacted turn 3508 | {REF}]"}]},
        {"role": "user", "content": "now"},
    ]}
    before = json.dumps(body)
    result = _runtime(body, "anthropic", responses_chain).restore_tool_output(REF)
    assert json.dumps(body) == before
    text = result if isinstance(result, str) else json.dumps(result)
    assert "old question" in text and "old answer" in text
