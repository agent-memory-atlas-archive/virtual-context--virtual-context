"""Stored summaries reach the model as written, bounded only by audience."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from virtual_context.core.summary_identity import (
    SUMMARY_ATTRIBUTION_QUARANTINE,
    contains_ambiguous_human_referent,
    render_summaries_for_model,
    sanitize_summary_payload_for_model,
    summaries_admitted_for_audience,
)
from virtual_context.types import SegmentMetadata, SpeakerRetrievalContext, StoredSummary

OWNER = "sk:agent:demo:discord:guild:1"
DM = "sk:agent:demo:discord:direct:9"


def _row(canonical_id: str, audience: str, version: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        conversation_id=OWNER,
        canonical_turn_id=canonical_id,
        audience_conversation_id=audience,
        audience_attribution_version=version,
    )


class _Store:
    def __init__(self, rows, *, fail: bool = False):
        self.rows = {(OWNER, row.canonical_turn_id): row for row in rows}
        self.fail = fail
        self.calls: list[dict] = []

    def get_canonical_turn_rows_by_id(self, keys, *, speaker_context=None, internal_validation=False):
        self.calls.append({"speaker_context": speaker_context, "internal_validation": internal_validation})
        if self.fail:
            raise RuntimeError("database unavailable")
        return {key: self.rows[key] for key in keys if key in self.rows}


def _summary(text: str, ids: list[str]) -> StoredSummary:
    return StoredSummary(
        ref=f"seg-{text[:8]}",
        primary_tag="health",
        tags=["health"],
        summary=text,
        summary_tokens=len(text) // 4,
        metadata=SegmentMetadata(canonical_turn_ids=ids),
    )


def _context(audience: str = OWNER) -> SpeakerRetrievalContext:
    return SpeakerRetrievalContext(
        tenant_id="t1", owner_conversation_id=OWNER, audience_conversation_id=audience,
    )


def _render(items, store, context):
    return render_summaries_for_model(items, store=store, conversation_id=OWNER, speaker_context=context)


def test_summary_from_the_request_audience_is_shown_as_written():
    text = "optics asked how long a 10x5000 IU kit lasts; Vast said about 23 months."
    store = _Store([_row("ct-1", OWNER), _row("ct-2", OWNER)])
    assert _render([_summary(text, ["ct-1", "ct-2"])], store, _context()) == [text]


def test_summary_drawing_on_another_audience_is_withheld():
    store = _Store([_row("ct-1", OWNER), _row("ct-2", DM)])
    rendered = _render([_summary("mixed guild and DM content", ["ct-1", "ct-2"])], store, _context())
    assert rendered == [SUMMARY_ATTRIBUTION_QUARANTINE]


def test_a_dm_request_does_not_see_guild_turns():
    store = _Store([_row("ct-1", OWNER)])
    rendered = _render([_summary("guild-only content", ["ct-1"])], store, _context(DM))
    assert rendered == [SUMMARY_ATTRIBUTION_QUARANTINE]


def test_rows_without_an_audience_are_the_owners_history():
    store = _Store([_row("ct-1", "", version=0)])
    assert _render([_summary("legacy content", ["ct-1"])], store, _context()) == ["legacy content"]


def test_a_request_without_a_proved_route_reads_as_the_owner():
    store = _Store([_row("ct-1", OWNER), _row("ct-2", DM)])
    items = [_summary("owner content", ["ct-1"]), _summary("dm content", ["ct-2"])]
    rendered = _render(items, store, SpeakerRetrievalContext(owner_conversation_id=OWNER))
    assert rendered == ["owner content", SUMMARY_ATTRIBUTION_QUARANTINE]


def test_source_rows_are_read_without_request_scoping():
    store = _Store([_row("ct-1", DM)])
    summaries_admitted_for_audience(
        [_summary("x", ["ct-1"])], store=store, conversation_id=OWNER, speaker_context=_context(),
    )
    assert store.calls == [{"speaker_context": None, "internal_validation": True}]


def test_a_failed_source_lookup_withholds_sourced_summaries():
    store = _Store([], fail=True)
    items = [_summary("sourced", ["ct-1"]), _summary("unsourced", [])]
    assert _render(items, store, _context()) == [SUMMARY_ATTRIBUTION_QUARANTINE, "unsourced"]


def test_full_depth_uses_the_stored_full_text():
    item = _summary("short", ["ct-1"])
    item.full_text = "the whole exchange"
    rendered = render_summaries_for_model(
        [item], store=_Store([_row("ct-1", OWNER)]), conversation_id=OWNER,
        speaker_context=_context(), depth="full",
    )
    assert rendered == ["the whole exchange"]


def test_tool_payloads_keep_stored_summary_text():
    payload = {"results": [{"summary": "The user stopped tesamorelin.", "excerpt": "They said so."}]}
    assert sanitize_summary_payload_for_model(payload) == payload


@pytest.mark.parametrize(
    "text",
    [
        "The user stopped tesamorelin.",
        "The user's experience ended.",
        "The user’s experience ended.",
        "A member reported tachycardia.",
        "This person planned a cycle.",
        "User has stopped the medication.",
        "User's plan changed.",
        "user shared a dosing update.",
        "Member discussed side effects.",
        "User: stopped the medication.",
        "User (10:00): stopped the medication.",
        "The user experience with tesamorelin included edema.",
        "The user profile shows current tesamorelin use.",
        "The user data indicates elevated IGF-1.",
        "The patient stopped tesamorelin.",
        "A participant reported tachycardia.",
        "They stopped tesamorelin.",
        "Someone stopped tesamorelin.",
        "An individual stopped tesamorelin.",
        "Their tesamorelin cycle ended.",
        "He took 2 mg.",
        "After the visit, she discontinued tesamorelin.",
        "- They started the protocol.",
        "Later, they reported edema.",
        "I stopped tesamorelin.",
        "My protocol ended.",
        "You stopped tesamorelin.",
        "Your protocol ended.",
        "I'm taking tesamorelin.",
        "A Discord member reported tachycardia.",
        "The customer stopped tesamorelin.",
        "User specified an initial requirement.",
    ],
)
def test_detects_unresolved_singular_human_referents(text: str) -> None:
    assert contains_ambiguous_human_referent(text)


@pytest.mark.parametrize(
    "text",
    [
        "The user interface now uses a users table.",
        "User input is validated before storage.",
        "A member function returns the account ID.",
        "The user-facing API was removed.",
        "User has a required email field in the schema.",
        "Their API was removed after the migration.",
        "BigTex stopped tesamorelin.",
        "Participants compared two protocols.",
    ],
)
def test_detector_does_not_match_technical_compounds_or_named_speakers(
    text: str,
) -> None:
    assert not contains_ambiguous_human_referent(text)
