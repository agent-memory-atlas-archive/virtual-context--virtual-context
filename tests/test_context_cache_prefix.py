"""VC's injected context must never break the provider's prompt cache.

Providers reuse a cached prompt only up to the first byte that differs from a
previous request. VC's context block changes on almost every call, so wherever
it is placed, everything after it is re-billed at full price. The invariant
these tests hold, for every payload format and for both places VC injects
context (the proxy's first injection and the tool loop's re-injection):

    Everything before the latest user message is byte-identical to what the
    client sent, contains no VC block, and is carried unchanged into the next
    request.

The tests deliberately do not say *where* the block goes. A change that moves
the block into the system prompt, the instructions, a leading message, or
anywhere else ahead of the conversation fails here, however the per-format
tests are rewritten. Do not relax this file to fit a new placement: a
placement that fails it costs every user a full-price prompt on every call.
"""

from __future__ import annotations

import copy
import json

import pytest

from virtual_context.core.provider_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    OpenAICodexAdapter,
)
from virtual_context.proxy.formats import get_format

MARK = "<system-reminder>"
SYSTEM = "You are a careful assistant. " * 40
TOOLS_ANTHROPIC = [{"name": "search", "description": "find things", "input_schema": {"type": "object"}}]
TOOLS_OPENAI = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
TOOLS_RESPONSES = [{"type": "function", "name": "search", "parameters": {"type": "object"}}]
TOOLS_GEMINI = [{"function_declarations": [{"name": "search"}]}]


# -- Per-format request builders ------------------------------------------------
# Each returns a request with a system prompt, tools, and a history ending in
# the user's current message. ``turns`` is a list of (user, assistant) pairs
# followed by the current user text.

def _anthropic(turns, current):
    messages = []
    for user, assistant in turns:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": current})
    return {"model": "m", "system": SYSTEM, "tools": TOOLS_ANTHROPIC, "messages": messages}


def _openai(turns, current):
    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in turns:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": current})
    return {"model": "m", "tools": TOOLS_OPENAI, "messages": messages}


def _responses(turns, current):
    def msg(role, text):
        kind = "output_text" if role == "assistant" else "input_text"
        return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}
    items = [msg("developer", SYSTEM)]
    for user, assistant in turns:
        items += [msg("user", user), msg("assistant", assistant)]
    items.append(msg("user", current))
    return {"model": "m", "instructions": "Be concise.", "tools": TOOLS_RESPONSES, "input": items}


def _gemini(turns, current):
    contents = []
    for user, assistant in turns:
        contents += [{"role": "user", "parts": [{"text": user}]}, {"role": "model", "parts": [{"text": assistant}]}]
    contents.append({"role": "user", "parts": [{"text": current}]})
    return {"system_instruction": {"parts": [{"text": SYSTEM}]}, "tools": TOOLS_GEMINI, "contents": contents}


# -- Reading order ----------------------------------------------------------------
# The sequence a provider reads, which is what its cache keys on.

def _chunks(fmt, body):
    if fmt == "anthropic":
        head = [("system", body.get("system")), ("tools", body.get("tools"))]
        return head + [("message", m) for m in body["messages"]]
    if fmt == "openai":
        return [("tools", body.get("tools"))] + [("message", m) for m in body["messages"]]
    if fmt == "openai_responses":
        head = [("instructions", body.get("instructions")), ("tools", body.get("tools"))]
        return head + [("message", m) for m in body["input"]]
    head = [("system_instruction", body.get("system_instruction")), ("tools", body.get("tools"))]
    return head + [("message", m) for m in body["contents"]]


def _is_user(item):
    return isinstance(item, dict) and item.get("role") == "user"


def _stable_prefix(fmt, body):
    """Serialized reading order up to (not including) the latest user message."""
    chunks = _chunks(fmt, body)
    last_user = max(i for i, (kind, item) in enumerate(chunks) if kind == "message" and _is_user(item))
    return "".join(json.dumps(item, sort_keys=True) for _, item in chunks[:last_user])


def _serialized(fmt, body):
    return "".join(json.dumps(item, sort_keys=True) for _, item in _chunks(fmt, body))


BUILDERS = {"anthropic": _anthropic, "openai": _openai, "openai_responses": _responses, "gemini": _gemini}
ADAPTERS = {"anthropic": AnthropicAdapter, "openai": OpenAIAdapter,
            "openai_responses": OpenAICodexAdapter, "gemini": GeminiAdapter}
HISTORY = [("What is a prime?", "A number with two divisors."), ("Is 9 prime?", "No, 9 = 3 x 3.")]


def _proxy_inject(fmt, body, text):
    return get_format(fmt).inject_context(body, text)


def _loop_inject(fmt, body, text):
    body = copy.deepcopy(body)
    ADAPTERS[fmt](api_key="k").inject_context(body, text)
    return body


INJECTORS = {"proxy": _proxy_inject, "tool_loop": _loop_inject}


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("injector", sorted(INJECTORS))
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_nothing_before_the_latest_user_message_changes(fmt, injector):
    sent = BUILDERS[fmt](HISTORY, "Is 11 prime?")
    before = _stable_prefix(fmt, sent)
    out = INJECTORS[injector](fmt, copy.deepcopy(sent), "topics: primes, divisors")
    assert MARK in _serialized(fmt, out), "context was not injected at all"
    assert MARK not in _stable_prefix(fmt, out), (
        f"{fmt}/{injector}: VC's block sits ahead of the latest user message, "
        "so every call re-bills the system prompt, tools and history"
    )
    assert _stable_prefix(fmt, out) == before, (
        f"{fmt}/{injector}: content ahead of the latest user message was changed"
    )


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("injector", sorted(INJECTORS))
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_the_next_request_reuses_this_request_as_its_prefix(fmt, injector):
    """Two consecutive calls with different VC context share everything up to the older turn."""
    call_1 = INJECTORS[injector](fmt, BUILDERS[fmt](HISTORY, "Is 11 prime?"), "topics: primes")
    call_2 = INJECTORS[injector](
        fmt,
        BUILDERS[fmt](HISTORY + [("Is 11 prime?", "Yes.")], "And 15?"),
        "topics: divisors, primes (reordered)",
    )
    shared = _stable_prefix(fmt, call_1)
    assert _serialized(fmt, call_2).startswith(shared), (
        f"{fmt}/{injector}: the second call does not start with the first call's history, "
        "so the provider cannot reuse its cache"
    )
    assert len(shared) > len(SYSTEM), "the shared prefix must at least cover the system prompt"


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("injector", sorted(INJECTORS))
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_reinjection_replaces_instead_of_stacking(fmt, injector):
    body = BUILDERS[fmt](HISTORY, "Is 11 prime?")
    once = INJECTORS[injector](fmt, body, "first context")
    twice = INJECTORS[injector](fmt, once, "second context")
    text = _serialized(fmt, twice)
    assert text.count(MARK) == 1
    assert "second context" in text and "first context" not in text
    assert MARK not in _stable_prefix(fmt, twice)


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_proxy_injection_does_not_mutate_the_client_request(fmt):
    sent = BUILDERS[fmt](HISTORY, "Is 11 prime?")
    snapshot = copy.deepcopy(sent)
    _proxy_inject(fmt, sent, "ctx")
    assert sent == snapshot


def _breakpoints(body):
    count = 0
    for key in ("tools", "system"):
        if isinstance(body.get(key), list):
            count += sum(1 for b in body[key] if isinstance(b, dict) and "cache_control" in b)
    for msg in body.get("messages", []):
        if isinstance(msg.get("content"), list):
            count += sum(1 for b in msg["content"] if isinstance(b, dict) and "cache_control" in b)
    return count


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("injector", sorted(INJECTORS))
def test_anthropic_never_exceeds_four_cache_breakpoints(injector):
    """A tool loop re-injects every round; each round must stay a valid request."""
    body = _anthropic(HISTORY, "Is 11 prime?")
    body["system"] = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    for n in range(8):
        body = INJECTORS[injector]("anthropic", body, f"context round {n}")
        body["messages"] += [
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{n}", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{n}", "content": "ok"}]},
        ]
        assert _breakpoints(body) <= 4
    body = INJECTORS[injector]("anthropic", body, "final context")
    assert _breakpoints(body) <= 4
    last_user = body["messages"][-1]["content"]
    assert last_user[0]["type"] == "tool_result" and "cache_control" in last_user[0]
    assert MARK in last_user[-1]["text"]


@pytest.mark.regression("PROXY-027")
def test_proxy_keeps_the_clients_own_anthropic_cache_breakpoints():
    """A client that marks its system prompt for caching keeps that marker through the proxy."""
    from virtual_context.proxy.helpers import _inject_context

    sent = _anthropic(HISTORY, "Is 11 prime?")
    sent["system"] = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    out = _inject_context(copy.deepcopy(sent), "ctx", "anthropic")
    assert out["system"] == sent["system"]


# -- Intents carried over from the per-format placement tests --------------------

@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("injector", sorted(INJECTORS))
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_five_reassemblies_leave_one_current_block(fmt, injector):
    body = BUILDERS[fmt](HISTORY, "Is 11 prime?")
    for n in range(5):
        body = INJECTORS[injector](fmt, body, f"reassembled round {n}")
    text = _serialized(fmt, body)
    assert text.count(MARK) == 1 and "reassembled round 4" in text
    assert not any(f"reassembled round {n}" in text for n in range(4))


def _with_legacy_system_block(fmt, body, tag):
    stale = f"<{tag}>\nstale context\n</{tag}>"
    if fmt == "anthropic":
        body["system"] = f"{SYSTEM}\n\n{stale}"
    elif fmt == "openai":
        body["messages"][0]["content"] = f"{SYSTEM}\n\n{stale}"
    elif fmt == "openai_responses":
        body["instructions"] = f"Be concise.\n\n{stale}"
    else:
        body["system_instruction"]["parts"].append({"text": stale})
    return body


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("tag", ["system-reminder", "virtual-context"])
@pytest.mark.parametrize("injector", sorted(INJECTORS))
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_a_stale_block_in_the_system_prompt_is_removed_and_the_user_text_kept(fmt, injector, tag):
    body = _with_legacy_system_block(fmt, BUILDERS[fmt](HISTORY, "Is 11 prime?"), tag)
    out = INJECTORS[injector](fmt, body, "fresh context")
    text = _serialized(fmt, out)
    assert "stale context" not in text and "fresh context" in text
    assert MARK not in _stable_prefix(fmt, out)
    assert _stable_prefix(fmt, out) == _stable_prefix(fmt, BUILDERS[fmt](HISTORY, "Is 11 prime?"))


@pytest.mark.regression("PROXY-027")
@pytest.mark.parametrize("fmt", sorted(BUILDERS))
def test_tool_loop_request_built_with_a_wrapped_system_prompt(fmt):
    """VC's own tool loop builds its request with the context wrapped in the system prompt."""
    adapter = ADAPTERS[fmt](api_key="k")
    body = adapter.build_request_body(
        model="m", messages=[{"role": "user", "content": "hi"}],
        system=f"{SYSTEM}\n\n<system-reminder>\ninitial context\n</system-reminder>",
        max_tokens=64, temperature=0.0, tools=None,
    )
    adapter.inject_context(body, "reassembled context")
    text = json.dumps(body)
    assert text.count(MARK) == 1 and "reassembled context" in text and "initial context" not in text
    assert "You are a careful assistant." in text
