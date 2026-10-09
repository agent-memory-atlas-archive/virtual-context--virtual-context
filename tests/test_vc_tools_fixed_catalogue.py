"""The VC tool catalogue is the same on every request, whatever the turn's history."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from virtual_context.proxy.helpers import offer_vc_tools, tool_names


def _engine(*, paging=True, tool_output=True, mode="supervised", compacted=0):
    return SimpleNamespace(
        config=SimpleNamespace(
            paging=SimpleNamespace(enabled=paging),
            tool_output=SimpleNamespace(enabled=tool_output),
        ),
        _retrieval=SimpleNamespace(_resolve_paging_mode=lambda model: mode),
        _engine_state=SimpleNamespace(compacted_prefix_messages=compacted),
    )


def _body(history_text):
    return {"model": "gpt-6-astra", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "earlier"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": history_text}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now?"}]},
    ]}


@pytest.mark.regression("PROXY-042")
def test_history_never_changes_the_offered_tools():
    plain, offered = offer_vc_tools(_body("plain answer"), _engine())
    stubbed, _ = offer_vc_tools(_body("[Compacted turn 5 | vc_restore_tool(ref=chain_1_abc)]"), _engine(compacted=40))
    assert offered is True
    assert json.dumps(plain["tools"], sort_keys=True) == json.dumps(stubbed["tools"], sort_keys=True)
    assert {"vc_find_quote", "vc_restore_tool"} <= set(tool_names(plain))


@pytest.mark.regression("PROXY-042")
def test_paging_mode_never_changes_the_offered_tools():
    supervised, _ = offer_vc_tools(_body("a"), _engine(mode="supervised"))
    autonomous, _ = offer_vc_tools(_body("a"), _engine(mode="autonomous", compacted=40))
    assert tool_names(supervised) == tool_names(autonomous)


def test_tools_configured_off_leave_the_request_alone():
    body = _body("a")
    out, offered = offer_vc_tools(body, _engine(paging=False, tool_output=False))
    assert offered is False and out is body
