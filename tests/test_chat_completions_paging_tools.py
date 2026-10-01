"""Chat Completions requests get the paging tools like every other format.

The Chat Completions format reported no tool interception and its tool
injection returned the body unchanged, so those clients received retrieved
context but could not page deeper with vc_expand_topic, vc_find_quote or
vc_restore_tool. The format now injects function tools and the proxy
intercepts their calls.
"""

from __future__ import annotations

import pytest

from virtual_context.proxy.formats import get_format
from virtual_context.proxy.helpers import _add_restore_tool, tool_names

DEFS = [
    {"name": "vc_expand_topic", "description": "Open a topic.", "input_schema": {"type": "object", "properties": {"tag": {"type": "string"}}}},
    {"name": "vc_find_quote", "description": "Search the record.", "input_schema": {"type": "object"}},
]


def _body(**extra):
    return {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.mark.regression("BUG-109")
def test_chat_completions_supports_tool_interception():
    assert get_format("openai").supports_tool_interception is True


@pytest.mark.regression("BUG-109")
def test_tools_are_injected_as_chat_completions_functions():
    own = {"type": "function", "function": {"name": "lookup", "parameters": {}}}
    body = get_format("openai").inject_tools(_body(tools=[own]), DEFS)
    assert body["tools"][0] == own
    assert body["tools"][1] == {"type": "function", "function": {
        "name": "vc_expand_topic", "description": "Open a topic.",
        "parameters": {"type": "object", "properties": {"tag": {"type": "string"}}}}}
    assert tool_names(body) == ["lookup", "vc_expand_topic", "vc_find_quote"]
    again = get_format("openai").inject_tools(body, DEFS)
    assert tool_names(again) == ["lookup", "vc_expand_topic", "vc_find_quote"]


@pytest.mark.regression("BUG-109")
@pytest.mark.parametrize("choice", ["none", {"type": "none"}])
def test_tool_choice_none_is_respected(choice):
    body = get_format("openai").inject_tools(_body(tool_choice=choice), DEFS)
    assert "tools" not in body


@pytest.mark.regression("BUG-109")
def test_required_tool_use_sets_tool_choice_only_when_unset():
    fmt = get_format("openai")
    assert fmt.inject_tools(_body(), DEFS, require_tool_use=True)["tool_choice"] == "required"
    assert fmt.inject_tools(_body(tool_choice="auto"), DEFS, require_tool_use=True)["tool_choice"] == "auto"


@pytest.mark.regression("BUG-109")
def test_the_restore_tool_can_be_added_to_a_chat_completions_request():
    body = _add_restore_tool(_body())
    assert "vc_restore_tool" in tool_names(body)
    assert tool_names(_add_restore_tool(body)).count("vc_restore_tool") == 1
