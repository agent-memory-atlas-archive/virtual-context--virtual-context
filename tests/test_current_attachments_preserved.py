"""Attachments on the message the user just sent reach the model unchanged.

Images, files and audio on the latest user message are the request itself.
Every stage that rewrites a payload (context injection in each format and in
the tool loop, the host-replay split, the compacted-turn drop, the turn filter,
media stubbing and budget enforcement) must leave them in place. Older
attachments may still be stubbed or dropped by design.
"""
from __future__ import annotations

import copy
import datetime

import pytest

from virtual_context.core.provider_adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAICodexAdapter
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.formats import get_format
from virtual_context.proxy.host_replay import expand_host_replay
from virtual_context.proxy.media import stub_media_by_position
from virtual_context.proxy.message_filter import drop_compacted_turns, filter_body_messages
from virtual_context.types import TurnTagEntry

B64 = "A" * 60000  # large enough to be a budget-reduction candidate

MEDIA = {
    "anthropic": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": B64}},
                  {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}}],
    "openai": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{B64}"}},
               {"type": "input_audio", "input_audio": {"data": "UklGR", "format": "wav"}}],
    "openai_responses": [{"type": "input_image", "image_url": f"data:image/jpeg;base64,{B64}"},
                         {"type": "input_file", "filename": "labs.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}],
    "gemini": [{"inline_data": {"mime_type": "image/jpeg", "data": B64}},
               {"file_data": {"mime_type": "video/mp4", "file_uri": "gs://bucket/clip.mp4"}}],
}
ADAPTERS = {"anthropic": AnthropicAdapter, "openai": OpenAIAdapter,
            "openai_responses": OpenAICodexAdapter, "gemini": GeminiAdapter}


def _history(fmt, n=8):
    turns = [(f"old question {i}", f"old answer {i} " + "x" * 400) for i in range(n)]
    media = copy.deepcopy(MEDIA[fmt])
    if fmt == "anthropic":
        msgs = []
        for u, a in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": [{"type": "text", "text": "What is in this?"}] + media})
        return {"model": "m", "system": "sys", "messages": msgs}
    if fmt == "openai":
        msgs = [{"role": "system", "content": "sys"}]
        for u, a in turns:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": [{"type": "text", "text": "What is in this?"}] + media})
        return {"model": "m", "messages": msgs}
    if fmt == "openai_responses":
        items = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]}]
        for u, a in turns:
            items += [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": u}]},
                      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": a}]}]
        items.append({"type": "message", "role": "user",
                      "content": [{"type": "input_text", "text": "What is in this?"}] + media})
        return {"model": "m", "input": items}
    contents = []
    for u, a in turns:
        contents += [{"role": "user", "parts": [{"text": u}]}, {"role": "model", "parts": [{"text": a}]}]
    contents.append({"role": "user", "parts": [{"text": "What is in this?"}] + media})
    return {"contents": contents}


def _latest_user_parts(fmt, body):
    key = {"openai_responses": "input", "gemini": "contents"}.get(fmt, "messages")
    field = "parts" if fmt == "gemini" else "content"
    last = [m for m in body[key] if isinstance(m, dict) and m.get("role") == "user"][-1]
    return last[field]


def _media_kept(fmt, body):
    # A cache marker on the attachment block (Anthropic) is allowed; the
    # attachment itself must be unchanged.
    parts = [{k: v for k, v in p.items() if k != "cache_control"} if isinstance(p, dict) else p
             for p in _latest_user_parts(fmt, body)]
    return all(any(p == m for p in parts) for m in MEDIA[fmt])


@pytest.mark.regression("PROXY-028")
@pytest.mark.parametrize("fmt", sorted(MEDIA))
def test_context_injection_keeps_current_attachments(fmt):
    out = get_format(fmt).inject_context(_history(fmt), "ctx")
    assert _media_kept(fmt, out)
    body = _history(fmt)
    ADAPTERS[fmt](api_key="k").inject_context(body, "ctx 1")
    ADAPTERS[fmt](api_key="k").inject_context(body, "ctx 2")
    assert _media_kept(fmt, body)


@pytest.mark.regression("PROXY-028")
@pytest.mark.parametrize("fmt", sorted(MEDIA))
def test_history_trimming_keeps_current_attachments(fmt):
    f = get_format(fmt)
    body, _ = drop_compacted_turns(_history(fmt), TurnTagIndex(), 1000, fmt=f, protected_recent_turns=1)
    assert _media_kept(fmt, body)
    index = TurnTagIndex()
    for t in range(20):
        index.append(TurnTagEntry(turn_number=t, message_hash=f"h{t}", tags=["x"], primary_tag="x",
                                  timestamp=datetime.datetime.now()))
    body, _ = filter_body_messages(_history(fmt), index, ["y"], recent_turns=1, compacted_turn=1000, fmt=f)
    assert _media_kept(fmt, body)
    body, _ = stub_media_by_position(_history(fmt), f, protected_recent_turns=1)
    assert _media_kept(fmt, body)


@pytest.mark.regression("PROXY-028")
def test_host_replay_split_keeps_current_attachments():
    media = copy.deepcopy(MEDIA["openai_responses"])
    prompt = ("OpenClaw assembled context for this turn:\n<conversation_context>\n[user]\nhi\n\n[assistant]\nhello\n"
              "</conversation_context>\n\nCurrent user request:\nWhat is in this?")
    body = {"model": "m", "input": [{"type": "message", "role": "user",
                                     "content": [{"type": "input_text", "text": prompt}] + media}]}
    out, n = expand_host_replay(body)
    assert n == 2 and _media_kept("openai_responses", out)
