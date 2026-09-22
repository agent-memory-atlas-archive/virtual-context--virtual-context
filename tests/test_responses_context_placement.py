"""VC's context block goes last, so everything the host sent stays a cacheable prefix."""
import pytest

from virtual_context.core.provider_adapters import OpenAICodexAdapter
from virtual_context.proxy.formats import get_format


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _codex_body():
    return {"model": "m", "input": [
        {"type": "additional_tools", "role": "developer", "tools": [{"name": "exec"}]},
        _msg("developer", "You are Codex."),
        _msg("developer", "You are a personal agent running inside OpenClaw."),
        _msg("user", "<environment_context>\n  <current_date>2026-09-22</current_date>\n</environment_context>"),
        _msg("developer", "<openclaw_temporal_context>now</openclaw_temporal_context>"),
        _msg("user", "earlier question"), _msg("assistant", "earlier answer"),
        _msg("user", "current question"),
    ]}


def _vc_positions(body):
    return [i for i, it in enumerate(body["input"])
            if it.get("role") == "developer" and "<system-reminder>" in str(it.get("content"))]


@pytest.mark.regression("PROXY-027")
def test_context_goes_last_and_leaves_everything_the_host_sent_untouched():
    body = _codex_body()
    out = get_format("openai_responses").inject_context(body, "topics A")
    assert _vc_positions(out) == [len(body["input"])]
    assert out["input"][:-1] == body["input"]
    assert "instructions" not in out


@pytest.mark.regression("PROXY-027")
def test_reinjection_replaces_the_block_and_keeps_the_prefix_identical():
    fmt = get_format("openai_responses")
    first = fmt.inject_context(_codex_body(), "topics A, B")
    second = fmt.inject_context(first, "topics B, A")
    assert _vc_positions(second) == [8]
    assert "topics B, A" in str(second["input"][8]) and "topics A, B" not in str(second)
    assert second["input"][:8] == first["input"][:8]


@pytest.mark.regression("PROXY-027")
def test_tool_loop_adapter_uses_the_same_placement():
    body = _codex_body()
    OpenAICodexAdapter(api_key="").inject_context(body, "ctx 1")
    OpenAICodexAdapter(api_key="").inject_context(body, "ctx 2")
    assert _vc_positions(body) == [8] and "ctx 2" in str(body["input"][8])


@pytest.mark.regression("PROXY-027")
def test_block_left_in_instructions_by_an_older_request_is_moved_out():
    body = {"instructions": "<system-reminder>\nold\n</system-reminder>\n\nBe helpful.", "input": [_msg("user", "hi")]}
    out = get_format("openai_responses").inject_context(body, "new")
    assert out["instructions"] == "Be helpful."
    assert _vc_positions(out) == [1] and out["input"][0]["role"] == "user"


@pytest.mark.regression("PROXY-027")
def test_next_tool_round_keeps_the_previous_request_as_its_prefix():
    """A tool round appends items; only VC's block moves, so the earlier request stays a prefix."""
    fmt = get_format("openai_responses")
    round1 = fmt.inject_context(_codex_body(), "ctx round 1")
    host_round2 = _codex_body()
    host_round2["input"] += [{"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": "{}"},
                             {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"}]
    round2 = fmt.inject_context(host_round2, "ctx round 2 (reordered)")
    assert round2["input"][:8] == round1["input"][:8]
    assert _vc_positions(round2) == [10]
