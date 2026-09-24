"""Speaker roster and label scans read only the fields they use.

Both scans walked the newest 400 logical turns through
``get_recent_canonical_turns``, which returns every content column (raw and
normalized text, tags, fact signals). Only the speaker, channel, audience and
ordering fields are read. ``get_recent_speaker_rows`` returns the same rows
in the same order with content reduced to presence markers.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from tests.pg_helpers import pg_dsn, pg_test_conn
from virtual_context.core.speaker_labels import resolve_speaker_labels
from virtual_context.core.speaker_roster import build_speaker_roster
from virtual_context.core.store import ContextStore
from virtual_context.types import CanonicalTurnRow, SpeakerRetrievalContext

OWNER = "owner-conv"
ACTOR = "actor:discord:1"


def _context():
    return SpeakerRetrievalContext(
        tenant_id="t",
        owner_conversation_id=OWNER,
        audience_conversation_id=OWNER,
        audience_channel_id="",
        audience_channel_scope="conversation",
        request_origin_channel_id="chan",
        requester_actor_id="",
        original_active_user_text="",
    )


class _LeanOnlyStore:
    """Full-row scans fail; only the speaker-row scan answers."""

    def __init__(self):
        self.speaker_calls = []

    def get_recent_canonical_turns(self, conversation_id, *, limit):
        raise AssertionError("full-row scan used for a speaker read")

    def get_recent_speaker_rows(self, conversation_id, *, limit):
        self.speaker_calls.append((conversation_id, limit))
        return [CanonicalTurnRow(
            conversation_id=OWNER,
            canonical_turn_id="ct-1",
            sort_key=1.0,
            user_content="u",
            sender="Ada",
            sender_actor_id=ACTOR,
            audience_conversation_id=OWNER,
            audience_attribution_version=1,
        )]

    def get_lifecycle_epoch(self, conversation_id):
        return 0

    def supports_speaker_handles(self):
        return True

    def get_speaker_handles(self, tenant_id, audience_conversation_id, actors):
        from virtual_context.types import SpeakerHandleAssignment
        return [
            SpeakerHandleAssignment(
                tenant_id=tenant_id,
                audience_conversation_id=audience_conversation_id,
                actor_id=ACTOR,
                handle="ada",
            ),
        ]


@pytest.mark.regression("BUG-091")
def test_roster_reads_speaker_rows():
    store = _LeanOnlyStore()
    build = build_speaker_roster(
        store, speaker_context=_context(),
        token_counter=lambda text: len(text) // 4, max_tokens=300,
    )
    assert store.speaker_calls
    assert build.snapshot is not None
    assert [entry.actor_id for entry in build.snapshot.entries] == [ACTOR]


@pytest.mark.regression("BUG-091")
def test_labels_read_speaker_rows():
    store = _LeanOnlyStore()
    labels = resolve_speaker_labels(
        store, [ACTOR], speaker_context=_context(),
    )
    assert store.speaker_calls
    assert labels == {ACTOR: "Ada"}


@pytest.mark.regression("BUG-091")
def test_base_store_speaker_rows_fall_back_to_full_rows():
    marker = [object()]
    store = SimpleNamespace(
        get_recent_canonical_turns=lambda conversation_id, *, limit: marker,
    )
    assert ContextStore.get_recent_speaker_rows(store, OWNER, limit=5) is marker


# ---------------------------------------------------------------------------
# Postgres parity with the full-row scan
# ---------------------------------------------------------------------------

PG_URL = pg_dsn()
_pg = pytest.mark.skipif(
    not PG_URL, reason="VC_TEST_POSTGRES_URL / DATABASE_URL not set",
)


def _speaker_view(row):
    return (
        row.canonical_turn_id,
        row.turn_group_number,
        row.sort_key,
        row.sender,
        row.sender_actor_id,
        row.origin_channel_id,
        row.audience_conversation_id,
        row.audience_attribution_version,
        bool(row.user_content),
        bool((row.user_content or "").strip()),
        bool(row.assistant_content),
    )


@_pg
@pytest.mark.regression("BUG-091")
@pytest.mark.parametrize("limit", [1, 2, 3, 50])
def test_postgres_speaker_rows_match_full_rows(limit):
    from virtual_context.storage.postgres import PostgresStore

    store = PostgresStore(PG_URL)
    conv = f"test-{uuid.uuid4().hex[:8]}"
    try:
        specs = [
            # (user, assistant, group)
            ("hello from ada", "", 0),
            ("", "reply one", 0),
            ("   ", "", 1),
            ("", "reply two", 1),
            ("both halves", "inline reply", -1),
            ("late user", "", 2),
        ]
        for number, (user, assistant, group) in enumerate(specs):
            store.save_canonical_turn(
                conv, number, user, assistant,
                primary_tag="topic", tags=["topic"],
                turn_group_number=group,
            )
        with pg_test_conn() as conn:
            conn.execute(
                """UPDATE canonical_turns
                      SET sender = 'Ada', sender_actor_id = %s,
                          origin_channel_id = 'chan',
                          audience_conversation_id = %s,
                          audience_attribution_version = 1
                    WHERE conversation_id = %s AND user_content <> ''""",
                (ACTOR, conv, conv),
            )
        full = store.get_recent_canonical_turns(conv, limit=limit)
        lean = store.get_recent_speaker_rows(conv, limit=limit)
        assert [_speaker_view(r) for r in lean] == [
            _speaker_view(r) for r in full
        ]
    finally:
        with pg_test_conn() as conn:
            conn.execute(
                "DELETE FROM canonical_turns WHERE conversation_id = %s",
                (conv,),
            )
        store.close()
