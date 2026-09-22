"""A completed pair from a payload that omits earlier turns is the conversation's newest turn."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from tests.test_handle_prepare_payload import _make_proxy_state
from virtual_context.types import Message, TurnTagEntry


def _pair(user: str, assistant: str) -> list[Message]:
    return [Message(role="user", content=user), Message(role="assistant", content=assistant)]


def _hash(user: str, assistant: str) -> str:
    return hashlib.sha256(f"{user} {assistant}".encode()).hexdigest()[:16]


def _indexed(state, *hashes):
    for n, h in enumerate(hashes):
        state.engine._turn_tag_index.append(TurnTagEntry(turn_number=n, message_hash=h, tags=["t"], primary_tag="t"))


def _capture_submits(state):
    submitted = []
    state._pool = MagicMock()
    state._pool.submit = lambda fn, *args: submitted.append(args) or MagicMock()
    return submitted


def test_fresh_thread_turn_is_reserved_after_the_index(tmp_path):
    state = _make_proxy_state(tmp_path)
    _indexed(state, "h0", "h1")
    submitted = _capture_submits(state)
    state.fire_turn_complete(_pair("third question", "third answer"))
    assert len(submitted) == 1
    history, _tokens, _turn_id, reserved_turn, message_hash = submitted[0]
    assert reserved_turn == 2 and message_hash == _hash("third question", "third answer")


def test_refiring_the_newest_indexed_turn_is_deduped(tmp_path):
    state = _make_proxy_state(tmp_path)
    _indexed(state, "h0", _hash("second question", "second answer"))
    submitted = _capture_submits(state)
    state.fire_turn_complete(_pair("second question", "second answer"))
    assert submitted == []


def test_a_client_that_resends_its_history_keeps_its_position(tmp_path):
    state = _make_proxy_state(tmp_path)
    _indexed(state, _hash("q1", "a1"), _hash("q2", "a2"))
    submitted = _capture_submits(state)
    history = _pair("q1", "a1") + _pair("q2", "a2") + _pair("q3", "a3")
    state.fire_turn_complete(history)
    assert len(submitted) == 1 and submitted[0][3] == 2


def test_a_divergent_pair_at_an_indexed_position_is_still_skipped(tmp_path):
    state = _make_proxy_state(tmp_path)
    _indexed(state, _hash("q1", "a1"), _hash("q2", "a2"), "h2")
    submitted = _capture_submits(state)
    history = _pair("q1", "a1") + _pair("q2", "a2") + _pair("other q3", "other a3")
    state.fire_turn_complete(history)
    assert submitted == []


def test_a_misaligned_history_is_treated_as_a_fresh_thread(tmp_path):
    state = _make_proxy_state(tmp_path)
    _indexed(state, "h0", "h1", "h2")
    submitted = _capture_submits(state)
    history = _pair("x1", "y1") + _pair("x2", "y2")
    state.fire_turn_complete(history)
    assert len(submitted) == 1 and submitted[0][3] == 3
