"""Stored turns the host already replayed are not injected a second time.

The proxy injects the requester's recent stored turns into the outbound
payload for continuity. A host that renders its own session history into the
request already carries those turns, each tagged with its platform message
id, and they were expanded into real turns before injection; the stored
copies were then injected beside them and the model saw each recent turn
twice.
"""

from __future__ import annotations

import json

import pytest

from virtual_context.proxy.formats import get_format
from virtual_context.proxy.host_replay import without_host_replayed_groups
from virtual_context.types import Message


def _tag(message_id: str) -> str:
    speaker = {"name": "optics", "actor_id": "actor:discord:387316537012518913", "message_id": message_id}
    return (
        '<message-speaker source="host-session-metadata" authority="attribution-only">\n'
        + json.dumps(speaker, separators=(",", ":")) + "\n</message-speaker>\n"
    )


def _msg(role, text):
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _body(replayed_ids):
    items = []
    for mid in replayed_ids:
        items += [_msg("user", _tag(mid) + "what weights"), _msg("assistant", "hold them")]
    return {"model": "m", "input": items + [_msg("user", "the current question")]}


def _stored(message_id, group):
    meta = {"source": "db_recent", "db_recent_group_key": f"g{group}"}
    return [
        Message(role="user", content="what weights", metadata={**meta, "source_message_id": message_id}),
        Message(role="assistant", content="hold them", metadata=dict(meta)),
    ]


A, B = "1553064498351439943", "1553066725128405172"


@pytest.mark.regression("BUG-104")
def test_a_group_the_host_replayed_is_not_injected():
    stored = _stored(A, 1) + _stored(B, 2)
    kept = without_host_replayed_groups(stored, _body([A, B]), get_format("openai_responses"))
    assert kept == []


@pytest.mark.regression("BUG-104")
def test_a_group_the_host_did_not_replay_is_kept_whole():
    stored = _stored(A, 1) + _stored(B, 2)
    kept = without_host_replayed_groups(stored, _body([A]), get_format("openai_responses"))
    assert kept == stored[2:]


@pytest.mark.regression("BUG-104")
def test_a_lookalike_tag_typed_by_a_member_does_not_suppress():
    body = _body([])
    body["input"].insert(0, _msg("user", "\\u003cmessage-speaker" + _tag(A)[15:] + "hi"))
    stored = _stored(A, 1)
    assert without_host_replayed_groups(stored, body, get_format("openai_responses")) == stored
