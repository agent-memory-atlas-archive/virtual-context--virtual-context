"""The proxy's own outbound budget trims oldest turn groups regardless of the model window."""
from types import SimpleNamespace

from virtual_context.proxy.formats import get_format
from virtual_context.proxy.handlers import _handler_upstream_limit
from virtual_context.proxy.message_filter import trim_to_upstream_limit


def _state(budget=0, upstream=0):
    return SimpleNamespace(engine=SimpleNamespace(config=SimpleNamespace(proxy=SimpleNamespace(
        upstream_context_limit=upstream, outbound_context_budget=budget))))


def test_budget_caps_the_model_window_only_when_set():
    body = {"model": "gpt-5.6-sol", "input": []}
    assert _handler_upstream_limit(body, _state(), 0) == 1_000_000
    assert _handler_upstream_limit(body, _state(budget=100_000), 0) == 100_000
    assert _handler_upstream_limit(body, _state(budget=5_000_000), 0) == 1_000_000
    assert _handler_upstream_limit(body, _state(budget=100_000), 40_000) == 40_000  # an explicit request limit wins


def _item(role, text, kind="input_text"):
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def test_developer_instructions_are_prefix_and_survive_trimming():
    fmt = get_format("openai_responses")
    items = [_item("developer", "You are Codex. " * 200), _item("developer", "House rules. " * 200)]
    for i in range(30):
        items.append(_item("user", f"question {i} " * 120))
        items.append(_item("assistant", f"answer {i} " * 120, "output_text"))
    items.append(_item("user", "the current question"))
    body = {"model": "gpt-5.6-sol", "input": items}
    total = fmt.estimate_payload_tokens(body)
    trimmed, removed = trim_to_upstream_limit(body, total // 3, fmt)
    assert removed > 0
    kept = trimmed["input"]
    assert kept[0]["role"] == "developer" and kept[1]["role"] == "developer"  # prefix kept
    assert kept[-1]["content"][0]["text"] == "the current question"  # newest kept
    assert kept[-3]["content"][0]["text"].startswith("question 29")  # previous turn kept
    assert fmt.estimate_payload_tokens(trimmed) <= total // 3
