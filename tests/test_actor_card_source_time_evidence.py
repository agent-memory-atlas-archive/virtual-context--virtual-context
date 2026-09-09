"""BUG-076: card evidence distinguishes attested occurrence from ingestion."""

from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from virtual_context.core.community.actor_card_evidence import ActorCardEvidenceService


def _source():
    return SimpleNamespace(
        tenant_id="tenant", owner_conversation_id="room",
        audience_conversation_id="room", audience_channel_id="channel",
        turn=SimpleNamespace(canonical_turn_id="source", conversation_id="room",
                             user_content="Our agreement starts now, for fourteen days.",
                             created_at="2026-08-30T12:00:00+00:00", first_seen_at="",
                             turn_group_number=0),
    )


@pytest.mark.regression("BUG-076")
def test_card_prompt_and_fingerprint_use_attested_time_independently_of_ingest_time():
    event_time = "2026-04-03T12:04:05.678Z"
    calls = []

    def getter(keys, *, tenant_id):
        calls.append((keys, tenant_id))
        return {("room", "source"): event_time}

    service = ActorCardEvidenceService(
        store=SimpleNamespace(get_canonical_source_event_times=getter),
        paired_agent_replies=lambda _: {},
    )
    turn = service.prompt_turns([_source()])[0]
    assert turn["occurred_at"] == event_time
    assert turn["timestamp"] == "2026-08-30T12:00:00+00:00"
    assert turn["timestamp_basis"] == "ingestion"
    assert {"kind": "source_event_time", "owner": "room", "id": "source",
            "occurred_at": event_time} in list(service.fingerprint_records([], [_source()]))
    assert calls == [([("room", "source")], "tenant")] * 2


@pytest.mark.regression("BUG-076")
def test_missing_occurrence_proof_is_never_replaced_with_ingestion_time():
    service = ActorCardEvidenceService(store=SimpleNamespace(), paired_agent_replies=lambda _: {})
    turn = service.prompt_turns([_source()])[0]
    assert "occurred_at" not in turn
    assert turn["timestamp_basis"] == "ingestion"
    assert list(service.fingerprint_records([], [_source()])) == []


@pytest.mark.regression("BUG-076")
def test_source_without_tenant_authority_cannot_read_occurrence_metadata():
    def forbidden(*args, **kwargs):
        raise AssertionError("unscoped source occurrence read")

    source = _source()
    del source.tenant_id
    service = ActorCardEvidenceService(
        store=SimpleNamespace(get_canonical_source_event_times=forbidden),
        paired_agent_replies=lambda _: {},
    )
    assert "occurred_at" not in service.prompt_turns([source])[0]


@pytest.mark.regression("BUG-076")
def test_large_configured_source_set_uses_bounded_tenant_reads():
    sizes = []

    def getter(keys, *, tenant_id):
        assert tenant_id == "tenant"
        assert len(keys) <= 2000
        sizes.append(len(keys))
        return {}

    sources = []
    for index in range(2001):
        source = _source()
        source.turn.canonical_turn_id = f"source-{index}"
        sources.append(source)
    service = ActorCardEvidenceService(
        store=SimpleNamespace(get_canonical_source_event_times=getter),
        paired_agent_replies=lambda _: {},
    )
    assert list(service.fingerprint_records([], sources)) == []
    assert sizes == [2000, 1]


@pytest.mark.regression("BUG-076")
@pytest.mark.parametrize("use_first_seen", [False, True])
def test_segment_message_and_fingerprint_label_ingestion_timestamp(use_first_seen):
    row = _source().turn
    row.sender_actor_id = "actor:discord:member-a"
    row.audience_conversation_id = "room"
    row.audience_attribution_version = 1
    row.origin_channel_id = "channel"
    row.turn_number = 3
    ingestion_time = row.created_at
    if use_first_seen:
        row.first_seen_at, row.created_at = row.created_at, ""
    segment = SimpleNamespace(
        metadata=SimpleNamespace(canonical_turn_ids=["source"]),
        end_timestamp=datetime(2026, 8, 30, tzinfo=timezone.utc),
    )

    def rows(keys, *, internal_validation):
        assert internal_validation and keys == [("room", "source")]
        return {("room", "source"): row}

    store = SimpleNamespace(
        get_segment=lambda ref, *, conversation_id: segment,
        get_canonical_turn_rows_by_id=rows,
    )
    fact = SimpleNamespace(
        owner_conversation_id="room", audience_conversation_id="room",
        fact=SimpleNamespace(id="fact", segment_ref="segment"),
    )
    service = ActorCardEvidenceService(store=store, paired_agent_replies=lambda _: {})
    segments, refs = service.evidence_segments(row.sender_actor_id, "room", [fact], {"fact"})
    message = segments[0]["messages"][0]
    assert refs == {("room", "segment")}
    assert message["timestamp"] == ingestion_time
    assert message["timestamp_basis"] == "ingestion"
    assert "occurred_at" not in message
    fingerprint = next(record for record in service.fingerprint_records([fact], [])
                       if record["kind"] == "fact_source")
    assert fingerprint["timestamp"] == message["timestamp"]
    assert fingerprint["timestamp_basis"] == message["timestamp_basis"]
    assert "occurred_at" not in fingerprint
