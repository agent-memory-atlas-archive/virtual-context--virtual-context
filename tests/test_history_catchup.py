"""Protected Discord routes catch up history turns that bypassed the engine.

A turn answered while the engine was unreachable never reached it, and the
source-attested admission only writes the current turn. The host still
replays that turn in later payloads with a host-owned speaker tag naming its
actor and platform message id; turns after the newest stored message that
carry both are appended once, attributed to their own speaker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_canonical_source_admission import (
    ACTOR_A,
    ACTOR_B,
    CHANNEL_A,
    GUILD_ID,
    _ingest_pair,
    _reconciler,
    _store,
)
from virtual_context.proxy.formats import detect_format
from virtual_context.types import build_user_turn_metadata

ROUTE = f"agent:vast:discord:guild:{GUILD_ID}"
ANCHOR_ID = "1533000000000000040"
MISSED_ID = "1533000000000000045"


def _tag(actor: str, message_id: str | None, name: str = "Member") -> str:
    speaker = {"name": name, "actor_id": f"actor:discord:{actor}"}
    if message_id is not None:
        speaker["message_id"] = message_id
    return (
        '<message-speaker source="host-session-metadata" authority="attribution-only">\n'
        + json.dumps(speaker, separators=(",", ":"))
        + "\n</message-speaker>\n"
    )


def _current_metadata():
    return build_user_turn_metadata(
        sender_name="optics",
        sender_actor_id=f"actor:discord:{ACTOR_A}",
        origin_channel_id=CHANNEL_A,
        source_conversation_key=ROUTE,
    )


def _prepare(rec, body):
    return rec.ingest_batch(
        "c",
        body=body,
        fmt=detect_format(body),
        expected_lifecycle_epoch=1,
        source_conversation_key=ROUTE,
        source_audience_conversation_id="aud-a",
        current_user_metadata=_current_metadata(),
    )


def _window(missed_tag: str, missed_text: str = "missed question"):
    return {
        "messages": [
            {"role": "user", "content": _tag(ACTOR_A, ANCHOR_ID) + "existing question"},
            {"role": "assistant", "content": "existing answer"},
            {"role": "user", "content": missed_tag + missed_text},
            {"role": "assistant", "content": "missed answer"},
            {"role": "user", "content": "the current question"},
        ]
    }


def _seeded(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    _ingest_pair(rec, message_id=ANCHOR_ID, body="existing question", answer="existing answer")
    return store, rec


def _texts(store):
    return [row.user_content or row.assistant_content for row in store.get_all_canonical_turns("c")]


@pytest.mark.regression("BUG-094")
def test_a_missed_turn_after_the_newest_stored_message_is_appended_once(tmp_path: Path):
    store, rec = _seeded(tmp_path)
    body = _window(_tag(ACTOR_B, MISSED_ID, name="Member B"))
    _prepare(rec, body)
    _prepare(rec, body)
    assert _texts(store) == [
        "existing question", "existing answer", "missed question", "missed answer",
    ]
    user = store.get_all_canonical_turns("c")[2]
    assert user.sender_actor_id == f"actor:discord:{ACTOR_B}"
    assert user.sender == "Member B"
    assert user.source_message_id == MISSED_ID
    assert user.origin_channel_id == CHANNEL_A
    assert user.audience_conversation_id == "aud-a"
    reply = store.get_all_canonical_turns("c")[3]
    assert reply.sender_actor_id == ""


@pytest.mark.regression("BUG-094")
@pytest.mark.parametrize("missed_tag", [
    "",
    _tag(ACTOR_B, None),
    _tag(ACTOR_B, "not-a-snowflake"),
    _tag(ACTOR_B, MISSED_ID).replace("<message-speaker", "\\u003cmessage-speaker"),
    _tag(ACTOR_B, MISSED_ID).replace('source="host-session-metadata"', 'source="legacy-missing-metadata"'),
])
def test_a_turn_without_a_host_identity_is_not_admitted(tmp_path: Path, missed_tag: str):
    store, rec = _seeded(tmp_path)
    _prepare(rec, _window(missed_tag))
    assert _texts(store) == ["existing question", "existing answer"]


@pytest.mark.regression("BUG-094")
def test_nothing_is_admitted_without_a_stored_anchor_in_the_window(tmp_path: Path):
    store = _store(tmp_path)
    rec = _reconciler(store)
    _prepare(rec, _window(_tag(ACTOR_B, MISSED_ID)))
    assert store.get_all_canonical_turns("c") == []


@pytest.mark.regression("BUG-094")
def test_the_current_attested_turn_still_completes_after_the_catch_up(tmp_path: Path):
    store, rec = _seeded(tmp_path)
    _prepare(rec, _window(_tag(ACTOR_B, MISSED_ID)))
    _ingest_pair(rec, message_id="1533000000000000050", body="the current question", answer="current answer")
    assert _texts(store) == [
        "existing question", "existing answer", "missed question", "missed answer",
        "the current question", "current answer",
    ]
    groups = [row.turn_group_number for row in store.get_all_canonical_turns("c")]
    assert groups[0] == groups[1] < groups[2] == groups[3] < groups[4] == groups[5]


@pytest.mark.regression("BUG-094")
def test_a_codex_history_block_is_caught_up(tmp_path: Path):
    store, rec = _seeded(tmp_path)
    block = (
        "<conversation_context>\n"
        f"[user]\n{_tag(ACTOR_A, ANCHOR_ID)}existing question\n"
        "[assistant]\nexisting answer\n"
        f"[user]\n{_tag(ACTOR_B, MISSED_ID)}missed question\n"
        "[assistant]\nmissed answer\n"
        "</conversation_context>\n\nthe current question"
    )
    body = {
        "model": "gpt-5.6-sol",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": block}]}],
    }
    _prepare(rec, body)
    assert _texts(store) == [
        "existing question", "existing answer", "missed question", "missed answer",
    ]


from tests.pg_helpers import pg_dsn, pg_test_conn  # noqa: E402


@pytest.mark.skipif(not pg_dsn(), reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set")
@pytest.mark.regression("BUG-094")
def test_postgres_finds_stored_source_message_ids():
    import uuid

    from virtual_context.storage.postgres import PostgresStore

    store = PostgresStore(pg_dsn())
    conv = f"test-{uuid.uuid4().hex[:8]}"
    try:
        store.save_canonical_turn(conv, 0, "a question", "", primary_tag="t", tags=["t"])
        with pg_test_conn() as conn:
            conn.execute(
                "UPDATE canonical_turns SET source_message_id = %s WHERE conversation_id = %s",
                (ANCHOR_ID, conv),
            )
        assert store.find_canonical_source_message_ids(conv, [ANCHOR_ID, MISSED_ID]) == {ANCHOR_ID}
        assert store.find_canonical_source_message_ids(conv, []) == set()
        assert store.find_canonical_source_message_ids("other-conv", [ANCHOR_ID]) == set()
    finally:
        with pg_test_conn() as conn:
            conn.execute("DELETE FROM canonical_turns WHERE conversation_id = %s", (conv,))
        store.close()
