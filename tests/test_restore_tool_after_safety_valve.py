"""Stubs made after tool injection still come with the restore tool.

The proxy chose whether to offer ``vc_restore_tool`` when it injected the
VC tools, but the safety valve stubs tool outputs later in the same request.
A request whose only stubs came from the valve therefore reached the model
with stubbed outputs and no tool to restore them.
"""

from __future__ import annotations

import pytest

from virtual_context.proxy.helpers import _add_restore_tool


def _names(body: dict) -> list[str]:
    names = []
    for tool in body.get("tools") or []:
        if "functionDeclarations" in tool:
            names.extend(d["name"] for d in tool["functionDeclarations"])
        else:
            names.append(tool.get("name") or (tool.get("function") or {}).get("name"))
    return names


@pytest.mark.regression("BUG-097")
@pytest.mark.parametrize("body", [
    {"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}],
     "tools": [{"type": "function", "name": "vc_find_quote", "parameters": {}}]},
    {"model": "claude-x", "system": "s", "messages": [{"role": "user", "content": "hi"}],
     "tools": [{"name": "vc_find_quote", "input_schema": {}}]},
    {"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
     "tools": [{"functionDeclarations": [{"name": "vc_find_quote"}]}]},
])
def test_the_restore_tool_is_added_once(body):
    once = _add_restore_tool(body)
    twice = _add_restore_tool(once)
    assert _names(once).count("vc_restore_tool") == 1
    assert _names(twice).count("vc_restore_tool") == 1
    assert "vc_find_quote" in _names(twice)
    assert "vc_restore_tool" not in _names(body)
