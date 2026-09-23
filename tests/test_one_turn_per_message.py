"""A message that takes several tool rounds is stored as one turn.

A fresh-thread host (one real user message per request; each tool round is a
new request) used to have every round's response stored as a finished turn:
the user message was written once per round, paired with whatever interim note
that round produced. Now the rounds that end in tool calls store nothing, the
interim notes are not ingested as the reply, and the final response stores the
user message once with the notes folded into the answer. Requests that resend
the whole conversation keep their existing behavior.
"""
from __future__ import annotations

import pytest

from test_proxy_review_invariants import _state
from virtual_context.proxy.formats import extract_ingestible_messages, get_format
from virtual_context.proxy.handlers import _continuation_session
from virtual_context.proxy.request_context import RequestContext
from virtual_context.proxy.turn_progress import response_ends_with_tool_calls, turn_progress
from virtual_context.types import SpeakerRetrievalContext

FMT = "openai_responses"


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _codex_round(n_rounds):
    """The request a fresh-thread host sends after ``n_rounds`` tool rounds."""
    items = [
        {"type": "additional_tools", "role": "developer", "tools": [{"name": "exec"}]},
        _msg("developer", "You are Codex."),
        _msg("user", "<environment_context>\n  <current_date>2026-09-22</current_date>\n</environment_context>"),
        _msg("user", "Can you access the bridge to my raspberry pi?"),
    ]
    for i in range(n_rounds):
        items.append(_msg("assistant", f"Checking step {i}."))
        items.append({"type": "custom_tool_call", "call_id": f"c{i}", "name": "exec", "input": "{}"})
        items.append({"type": "custom_tool_call_output", "call_id": f"c{i}", "output": "ok"})
    return {"model": "m", "input": items}


def _response(text, *, tool):
    output = [_msg("assistant", text)]
    if tool:
        output.append({"type": "custom_tool_call", "call_id": "next", "name": "exec", "input": "{}"})
    return {"id": "r", "output": output, "status": "completed"}


def _session(body):
    state = _state()
    ctx = RequestContext.create(
        body=body, state=state, provider="http://upstream", api_format=FMT, tenant_id="t",
        conversation_id="conversation-A", audience_route="a", upstream_limit=100_000, output_allowance=200,
        speaker_context=SpeakerRetrievalContext(tenant_id="t", owner_conversation_id="conversation-A",
                                                audience_conversation_id="a"),
        metrics=state.metrics,
    )
    return state, _continuation_session(ctx, state)


@pytest.mark.regression("PROXY-033")
def test_request_classification():
    fmt = get_format(FMT)
    first = turn_progress(_codex_round(0), fmt)
    assert first.fresh_thread and not first.in_progress
    mid = turn_progress(_codex_round(2), fmt)
    assert mid.in_progress and mid.interim_texts == ["Checking step 0.", "Checking step 1."]
    full_history = _codex_round(1)
    full_history["input"].insert(3, _msg("user", "an earlier question"))
    full_history["input"].insert(4, _msg("assistant", "an earlier answer"))
    assert not turn_progress(full_history, fmt).fresh_thread


@pytest.mark.regression("PROXY-033")
def test_interim_notes_are_not_ingested_as_the_reply():
    messages, stats = extract_ingestible_messages(_codex_round(2), get_format(FMT))
    assert [(m.role, m.content) for m in messages] == [("user", "Can you access the bridge to my raspberry pi?")]
    assert stats["skipped_in_progress_assistant_count"] == 2


@pytest.mark.regression("PROXY-033")
@pytest.mark.parametrize("rounds", [0, 2])
def test_a_round_that_ends_in_tool_calls_stores_nothing(rounds):
    state, session = _session(_codex_round(rounds))
    response = _response(f"Checking step {rounds}.", tool=True)
    session.persist_completed(f"Checking step {rounds}.", None,
                              ends_with_tool_calls=response_ends_with_tool_calls(response, FMT))
    assert not state.fire_turn_complete.called


@pytest.mark.regression("PROXY-033")
def test_the_final_round_stores_one_turn_with_the_notes_folded_in():
    state, session = _session(_codex_round(2))
    response = _response("The bridge is reachable.", tool=False)
    session.persist_completed("The bridge is reachable.", None,
                              ends_with_tool_calls=response_ends_with_tool_calls(response, FMT))
    saved = state.fire_turn_complete.call_args.args[0]
    assert saved[-1].role == "assistant"
    assert saved[-1].content == "Checking step 0.\n\nChecking step 1.\n\nThe bridge is reachable."


@pytest.mark.regression("PROXY-033")
def test_requests_that_resend_history_keep_per_response_completion():
    body = _codex_round(1)
    body["input"].insert(3, _msg("user", "an earlier question"))
    body["input"].insert(4, _msg("assistant", "an earlier answer"))
    state, session = _session(body)
    session.persist_completed("interim", None, ends_with_tool_calls=True)
    assert state.fire_turn_complete.called


@pytest.mark.regression("PROXY-033")
@pytest.mark.parametrize("fmt,response,expected", [
    ("anthropic", {"stop_reason": "tool_use", "content": []}, True),
    ("anthropic", {"stop_reason": "end_turn", "content": [{"type": "text", "text": "x"}]}, False),
    ("openai", {"choices": [{"finish_reason": "tool_calls", "message": {}}]}, True),
    ("openai", {"choices": [{"finish_reason": "stop", "message": {"content": "x"}}]}, False),
    ("gemini", {"candidates": [{"content": {"parts": [{"functionCall": {"name": "f"}}]}}]}, True),
    ("gemini", {"candidates": [{"content": {"parts": [{"text": "x"}]}}]}, False),
    ("openai_responses", {"output": [{"type": "function_call"}]}, True),
    ("openai_responses", {"output": [_msg("assistant", "x")]}, False),
])
def test_tool_call_endings_in_every_format(fmt, response, expected):
    assert response_ends_with_tool_calls(response, fmt) is expected
