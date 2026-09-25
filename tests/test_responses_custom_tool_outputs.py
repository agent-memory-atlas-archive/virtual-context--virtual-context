"""Codex tool loops arrive as one Responses turn; VC must still see and shrink them."""
from __future__ import annotations

import pytest

from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import detect_format
from virtual_context.proxy.message_filter import stub_tool_outputs_by_position


class _Store:
    def __init__(self):
        self.saved: dict[str, str] = {}

    def store_tool_output(self, *, ref, conversation_id, tool_name, command, turn, content, original_bytes):
        self.saved[ref] = content

    def link_turn_tool_output(self, conversation_id, turn, ref):
        pass


def _codex_body(n_outputs: int, size: int = 4000) -> dict:
    items: list[dict] = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Vast."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "check the cameras"}]},
    ]
    for i in range(n_outputs):
        items.append({"type": "reasoning", "id": f"rs_{i}", "summary": []})
        items.append({"type": "custom_tool_call", "call_id": f"call_{i}", "name": "camera_eyes_events", "input": f"{{\"n\": {i}}}"})
        items.append({"type": "custom_tool_call_output", "call_id": f"call_{i}", "output": f"event-{i} " * (size // 8)})
    items.append({"type": "function_call", "call_id": "fn_last", "name": "mqtt_publish", "arguments": "{}"})
    items.append({"type": "function_call_output", "call_id": "fn_last", "output": "published " * 40})
    return {"model": "gpt-5.6-sol", "stream": True, "input": items}


def test_custom_tool_items_are_tool_calls_and_outputs():
    body = _codex_body(3)
    fmt = detect_format(body)
    calls = list(fmt.iter_tool_calls(body))
    outputs = list(fmt.iter_tool_outputs(body))
    assert [c.name for c in calls] == ["camera_eyes_events"] * 3 + ["mqtt_publish"]
    assert [o.call_id for o in outputs] == ["call_0", "call_1", "call_2", "fn_last"]
    assert outputs[0].content.startswith("event-0")
    fmt.replace_tool_output_content(body, outputs[0], "stub")
    assert body["input"][4]["output"] == "stub"


def test_custom_tool_output_text_parts_are_joined():
    body = {"model": "gpt-5.6-sol", "input": [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "custom_tool_call", "call_id": "c1", "name": "t", "input": ""},
        {"type": "custom_tool_call_output", "call_id": "c1", "output": [
            {"type": "input_text", "text": "part one"}, {"type": "input_text", "text": "part two"},
        ]},
    ]}
    fmt = detect_format(body)
    (output,) = fmt.iter_tool_outputs(body)
    assert output.content == "part one\npart two"


def test_codex_loop_is_one_turn_with_tool_activity():
    body = _codex_body(3)
    fmt = detect_format(body)
    turns = fmt.group_into_turns(body)
    # developer prefix, then the whole tool loop as the user's single turn
    assert len(turns) == 2
    assert turns[0].has_tool_activity is False
    assert turns[1].has_tool_activity is True
    assert turns[1].indices == list(range(1, len(body["input"])))


def test_intrusion_stubs_consumed_outputs_inside_the_current_turn():
    body = _codex_body(6, size=8000)
    fmt = detect_format(body)
    store = _Store()
    body, count, refs = stub_tool_outputs_by_position(
        body, fmt,
        protected_recent_turns=6,
        turn_tag_index=TurnTagIndex(),
        store=store,
        conversation_id="conv",
        protected_intrusion_threshold=0.6,
        context_budget=6000,  # zone (~12k tokens) far over 60% of budget
    )
    outputs = list(fmt.iter_tool_outputs(body))
    stubbed = [o for o in outputs if o.content.startswith("[tool output ref=")]
    verbatim = [o for o in outputs if not o.content.startswith("[tool output ref=")]
    assert count == len(stubbed) == len(refs) > 0
    # the newest two outputs always travel verbatim
    assert [o.call_id for o in verbatim][-2:] == ["call_5", "fn_last"]
    assert all(o.call_id in ("call_5", "fn_last") for o in verbatim)
    # oldest consumed outputs go first
    assert outputs[0].content.startswith("[tool output ref=")
    assert 'vc_restore_tool(ref="' in outputs[0].content
    assert set(refs) == set(store.saved)


def test_no_intrusion_when_zone_is_within_budget():
    body = _codex_body(4, size=2000)
    fmt = detect_format(body)
    body, count, refs = stub_tool_outputs_by_position(
        body, fmt,
        protected_recent_turns=6,
        turn_tag_index=TurnTagIndex(),
        store=_Store(),
        conversation_id="conv",
        protected_intrusion_threshold=0.6,
        context_budget=200_000,
    )
    assert (count, refs) == (0, [])
    assert all(not o.content.startswith("[tool output ref=") for o in fmt.iter_tool_outputs(body))


def test_intrusion_stops_once_zone_fits():
    body = _codex_body(10, size=4000)
    fmt = detect_format(body)
    store = _Store()
    body, count, _ = stub_tool_outputs_by_position(
        body, fmt,
        protected_recent_turns=6,
        turn_tag_index=TurnTagIndex(),
        store=store,
        conversation_id="conv",
        protected_intrusion_threshold=0.6,
        context_budget=12_000,
    )
    outputs = list(fmt.iter_tool_outputs(body))
    assert 0 < count < len(outputs) - 2
    # stubbing proceeds oldest-first and leaves a contiguous verbatim tail
    flags = [o.content.startswith("[tool output ref=") for o in outputs]
    assert flags == sorted(flags, reverse=True)


def _workout_loop_body(size: int = 4000) -> dict:
    record = "workout-records/1449190067187879997.json"
    calls = [
        ("c0", "sed -n '1,240p' /root/.openclaw/agents/vast/agent/workshop-skills/workout-log-grading/SKILL.md"),
        ("c1", f"jq 'keys' /root/.openclaw/workspace-vast/{record}"),
        ("c2", f"sed -n '1,300p' {record}"),
        ("c3", "cat notes/unrelated.md"),
        ("c4", f"jq '.schedule' {record}"),
    ]
    items: list[dict] = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "grade this workout"}]},
    ]
    for call_id, command in calls:
        items.append({"type": "function_call", "call_id": call_id, "name": "exec",
                      "arguments": '{"command": "' + command.replace('"', '\\"') + '"}'})
        items.append({"type": "function_call_output", "call_id": call_id,
                      "output": f"{call_id} line one\n" + (f"{call_id} " * (size // 4))})
    return {"model": "gpt-5.6-sol", "stream": True, "input": items}


def _stub_order(body: dict, context_budget: int) -> tuple[list[str], dict]:
    import hashlib

    fmt = detect_format(body)
    contents = {o.call_id: o.content for o in fmt.iter_tool_outputs(body)}
    by_ref = {f"tool_{hashlib.sha256(c.encode()).hexdigest()[:12]}": cid for cid, c in contents.items()}
    body, _count, refs = stub_tool_outputs_by_position(
        body, fmt, protected_recent_turns=6, turn_tag_index=TurnTagIndex(), store=_Store(),
        conversation_id="conv", protected_intrusion_threshold=0.6, context_budget=context_budget,
    )
    return [by_ref[r] for r in refs], {o.call_id: o.content for o in fmt.iter_tool_outputs(body)}


@pytest.mark.regression("BUG-098")
def test_in_loop_stubbing_takes_superseded_then_unrelated_outputs_first():
    order, _outputs = _stub_order(_workout_loop_body(), context_budget=100)
    assert order == ["c1", "c0", "c2"]


@pytest.mark.regression("BUG-098")
def test_an_in_loop_stub_keeps_a_preview_of_the_output():
    _order, outputs = _stub_order(_workout_loop_body(), context_budget=100)
    assert 'vc_restore_tool(ref="' in outputs["c1"]
    assert "c1 line one" in outputs["c1"]
    assert len(outputs["c1"]) < 600
