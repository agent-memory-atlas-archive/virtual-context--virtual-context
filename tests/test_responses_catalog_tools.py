"""A Responses client that declares VC tools in its own catalog gets no second copy."""
from __future__ import annotations

import pytest

from virtual_context.core.tool_loop import vc_tool_definitions
from virtual_context.proxy.formats import get_format
from virtual_context.proxy.helpers import tool_names


def _catalog_body(*names: str) -> dict:
    return {
        "model": "gpt-5.6-sol",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{"type": "custom", "name": "exec", "description": "run JavaScript"}] + [
                        {"type": "function", "name": n, "parameters": {"type": "object"}} for n in names
                    ],
                }],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
    }


@pytest.mark.regression("BUG-116")
def test_catalog_declared_vc_tools_are_not_declared_again():
    body = _catalog_body("vc_find_quote", "vc_expand_topic")
    out = get_format("openai_responses").inject_tools(body, vc_tool_definitions())
    top = [t["name"] for t in out["tools"]]
    assert "vc_find_quote" not in top
    assert "vc_expand_topic" not in top
    assert "vc_search_summaries" in top


@pytest.mark.regression("BUG-116")
def test_tool_names_include_the_client_catalog():
    body = _catalog_body("vc_find_quote")
    assert "vc_find_quote" in tool_names(body)
    assert "exec" in tool_names(body)


def test_without_a_catalog_every_vc_tool_is_declared():
    body = {"model": "gpt-5.6-sol", "input": [{"type": "message", "role": "user", "content": "hi"}]}
    out = get_format("openai_responses").inject_tools(body, vc_tool_definitions())
    assert {t["name"] for t in out["tools"]} == {d["name"] for d in vc_tool_definitions()}
