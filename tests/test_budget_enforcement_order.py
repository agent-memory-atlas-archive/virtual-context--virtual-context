"""Budget enforcement is the last resort, and it trims oldest first.

It runs only when every other stage still leaves the payload over the window,
and may then reach into protected turns. What it must never do is truncate a
newer turn while an older one is still intact: the turns it touches are
always a contiguous run starting from the oldest.
"""
from __future__ import annotations

import copy
import logging

import pytest

from virtual_context.proxy.formats import get_format
from virtual_context.proxy.message_filter import enforce_payload_budget

LONG = "fact " * 800
SIZES = [1, 3, 1, 5, 2, 1, 4, 1, 2, 1]  # reply size multipliers: biggest-first would jump around


def _build(fmt):
    turns = [(f"question {i}", f"reply {i}: " + LONG * m) for i, m in enumerate(SIZES)]
    if fmt in ("anthropic", "openai"):
        msgs = [{"role": "system", "content": "sys"}] if fmt == "openai" else []
        for u, a in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": "current question"})
        body = {"model": "m", "messages": msgs}
        if fmt == "anthropic":
            body["system"] = "sys"
        return body, "messages", 1 if fmt == "openai" else 0
    if fmt == "openai_responses":
        items = []
        for u, a in turns:
            items += [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": u}]},
                      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": a}]}]
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "current question"}]})
        return {"model": "m", "input": items}, "input", 0
    contents = []
    for u, a in turns:
        contents += [{"role": "user", "parts": [{"text": u}]}, {"role": "model", "parts": [{"text": a}]}]
    contents.append({"role": "user", "parts": [{"text": "current question"}]})
    return {"contents": contents}, "contents", 0


@pytest.mark.regression("PROXY-029")
@pytest.mark.parametrize("window", [30000, 15000, 6000, 2000])
@pytest.mark.parametrize("fmt", ["anthropic", "openai", "openai_responses", "gemini"])
def test_truncation_runs_from_the_oldest_turn_forward(fmt, window):
    logging.disable(logging.CRITICAL)
    try:
        body, key, offset = _build(fmt)
        out, reductions, _ = enforce_payload_budget(copy.deepcopy(body), get_format(fmt), window)
    finally:
        logging.disable(logging.NOTSET)
    touched = sorted({(i - offset) // 2 for i, (a, b) in enumerate(zip(body[key], out[key])) if a != b})
    assert touched == list(range(len(touched))), f"{fmt} window={window}: touched turns {touched}"
    if reductions:
        assert touched, "reductions reported but nothing changed"
