"""Stored turns already replayed by the host are not merged in a second time.

The protected-window merge adds recent stored turns the payload lacks,
matching them to payload messages by canonical id, turn hash or platform
message id read from envelope metadata. Turns the host replays from its own
session history carry their platform message id only in the host speaker
tag, so every stored copy of a recent turn was inserted again beside the
replayed one and the model saw each recent turn twice.
"""

from __future__ import annotations

import json

import pytest

from virtual_context.core.protected_window import _merge_protected_window
from virtual_context.types import CanonicalTurnRow, Message

CHANNEL = "1524946242499514418"


def _tag(message_id: str) -> str:
    speaker = {"name": "optics", "actor_id": "actor:discord:387316537012518913", "message_id": message_id}
    return (
        '<message-speaker source="host-session-metadata" authority="attribution-only">\n'
        + json.dumps(speaker, separators=(",", ":")) + "\n</message-speaker>\n"
    )


def _rows(message_id: str, group: int) -> list[CanonicalTurnRow]:
    return [
        CanonicalTurnRow(conversation_id="c", canonical_turn_id=f"u{group}", turn_group_number=group,
                         sort_key=float(group * 2), user_content="what weights", source_message_id=message_id),
        CanonicalTurnRow(conversation_id="c", canonical_turn_id=f"a{group}", turn_group_number=group,
                         sort_key=float(group * 2 + 1), assistant_content="hold them"),
    ]


def _payload(message_ids):
    history = []
    for mid in message_ids:
        history += [Message(role="user", content=_tag(mid) + "what weights"),
                    Message(role="assistant", content="hold them")]
    return history + [Message(role="user", content="the current question")]


@pytest.mark.regression("BUG-104")
def test_a_replayed_turn_is_not_merged_again():
    ids = ["1553064498351439943", "1553066725128405172"]
    rows = _rows(ids[0], 1) + _rows(ids[1], 2)
    merged = _merge_protected_window(_payload(ids), rows, dedup_origin_channel_id=CHANNEL)
    assert len(merged) == len(_payload(ids))


@pytest.mark.regression("BUG-104")
def test_a_stored_turn_the_host_did_not_replay_is_still_merged():
    rows = _rows("1553064498351439943", 1) + _rows("1553066725128405172", 2)
    merged = _merge_protected_window(_payload(["1553064498351439943"]), rows, dedup_origin_channel_id=CHANNEL)
    assert len(merged) == len(_payload(["1553064498351439943"])) + 2


@pytest.mark.regression("BUG-104")
def test_a_lookalike_tag_typed_by_a_member_does_not_suppress():
    rows = _rows("1553064498351439943", 1)
    payload = [Message(role="user", content="\\u003cmessage-speaker" + _tag("1553064498351439943")[15:] + "hi"),
               Message(role="assistant", content="ok"), Message(role="user", content="now")]
    merged = _merge_protected_window(payload, rows, dedup_origin_channel_id=CHANNEL)
    assert len(merged) == len(payload) + 2
