"""Images are sized by what the model is billed, not by their base64 length.

The stubbing passes sized the protected zone as serialized bytes / 4, so one
attached screenshot (hundreds of KB of base64) read as ~180K tokens and a
tool loop on an image turn crossed every stubbing threshold while the model
was billed a fraction of that. The zone is now measured with the format's
media-aware estimate, which also covers images returned inside tool outputs.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from tests.test_responses_custom_tool_outputs import _Store
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import get_format
from virtual_context.proxy.message_filter import stub_tool_outputs_by_position

_B64 = base64.b64encode(os.urandom(600_000)).decode()
_IMAGE = {"type": "input_image", "image_url": "data:image/jpeg;base64," + _B64}


def _image_turn_body():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "earlier"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "grade this"}, _IMAGE]},
    ]
    for i in range(3):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "exec", "arguments": f'{{"cmd": "cat file{i}.json"}}'})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": f"line {i} " * 500})
    items.append({"type": "function_call", "call_id": "v", "name": "view_image", "arguments": '{"path": "a.jpg"}'})
    items.append({"type": "function_call_output", "call_id": "v", "output": [
        {"type": "input_text", "text": "Loaded 1 image."}, _IMAGE]})
    return {"model": "m", "input": items}


@pytest.mark.regression("BUG-102")
def test_an_image_in_a_tool_output_is_sized_as_an_image():
    fmt = get_format("openai_responses")
    item = _image_turn_body()["input"][-1]
    assert fmt.estimate_message_tokens(item) < len(json.dumps(item)) // 40


@pytest.mark.regression("BUG-102")
def test_an_image_turn_under_budget_is_not_stubbed():
    body = _image_turn_body()
    fmt = get_format("openai_responses")
    _body, count, refs = stub_tool_outputs_by_position(
        body, fmt, protected_recent_turns=6, turn_tag_index=TurnTagIndex(), store=_Store(),
        conversation_id="conv", protected_intrusion_threshold=0.6, context_budget=20000,
        deep_intrusion_trigger=1.0,
    )
    assert count == 0 and refs == []
