"""Recorded VC tool calls carry what the tool itself reported finding."""
from __future__ import annotations

import json

from virtual_context.proxy.continuation import tool_found


def test_a_result_that_found_excerpts_is_found():
    assert tool_found(json.dumps({"found": True, "results": [{"text": "rhr 82"}]})) is True


def test_a_result_that_found_nothing_is_not_found():
    assert tool_found(json.dumps({"found": False, "results": []})) is False


def test_an_already_provided_result_counts_as_found():
    assert tool_found(json.dumps({"found": "already_provided"})) is True


def test_a_result_that_does_not_say_is_unknown():
    assert tool_found(json.dumps({"tokens_added": 120})) is None
    assert tool_found("plain text") is None
