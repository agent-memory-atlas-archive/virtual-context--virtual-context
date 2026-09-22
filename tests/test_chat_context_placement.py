"""Chat Completions: VC's block rides the latest user message, so the rest stays a cacheable prefix."""
import pytest

from virtual_context.core.provider_adapters import OpenAIAdapter
from virtual_context.proxy.formats import get_format


def _body():
    return {"model": "m", "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]}


@pytest.mark.regression("PROXY-027")
def test_system_prompt_and_history_stay_byte_identical():
    body = _body()
    out = get_format("openai").inject_context(body, "topics A")
    assert out["messages"][:3] == body["messages"][:3]
    assert out["messages"][3]["content"] == "second\n\n<system-reminder>\ntopics A\n</system-reminder>"


@pytest.mark.regression("PROXY-027")
def test_reinjection_replaces_the_block_without_stacking():
    fmt = get_format("openai")
    once = fmt.inject_context(_body(), "topics A")
    twice = fmt.inject_context(once, "topics B")
    assert str(twice).count("<system-reminder>") == 1
    assert twice["messages"][3]["content"].endswith("topics B\n</system-reminder>")


@pytest.mark.regression("PROXY-027")
def test_list_content_and_the_tool_loop_adapter_use_the_same_rule():
    body = {"messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    OpenAIAdapter("k").inject_context(body, "ctx 1")
    OpenAIAdapter("k").inject_context(body, "ctx 2")
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "hi"} and "ctx 2" in parts[-1]["text"] and len(parts) == 2
