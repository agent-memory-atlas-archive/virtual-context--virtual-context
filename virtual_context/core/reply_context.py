"""Bounded reply references for retrieval, separate from requester content."""

from __future__ import annotations

import json

from ..types import (
    REPLY_SUBJECT_KEY,
    ReplySubject,
    RequestRoles,
    get_reply_subject,
)


def reply_parent_from_metadata(metadata: dict, conversation_key: str) -> ReplySubject | None:
    """Read a normalized adapter-owned parent; never discover one in prose."""
    snapshot = metadata.get(REPLY_SUBJECT_KEY)
    if not isinstance(snapshot, dict) or snapshot.get("source") != "rest-provenance":
        return None
    direct = get_reply_subject(metadata, conversation_key)
    if direct.unresolved_reason or not direct.target_message_id:
        return None
    value = snapshot.get("value")
    parent = value.get("in_reply_to") if isinstance(value, dict) else None
    if not isinstance(parent, dict):
        return None
    result = get_reply_subject(
        {REPLY_SUBJECT_KEY: {"value": parent}},
        conversation_key,
    )
    if (
        result.unresolved_reason
        or not result.subject_actor_id
        or not result.target_body
        or not result.target_message_id
        or result.target_message_id == direct.target_message_id
    ):
        return None
    return result


def reply_references(roles: RequestRoles | None, owner: str) -> tuple[ReplySubject, ...]:
    """Return at most the direct target and its parent under a proved route."""
    if (
        not isinstance(roles, RequestRoles)
        or not roles.requester_actor_id
        or not owner
        or roles.owner_conversation_id != owner
        or not roles.audience_conversation_id
        or not roles.reply_target_message_id
        or not roles.reply_target_body
    ):
        return ()
    target = ReplySubject(
        subject_actor_id=roles.subject_actor_id,
        subject_label=roles.subject_label,
        target_message_id=roles.reply_target_message_id,
        target_body=roles.reply_target_body,
    )
    parent = roles.reply_parent
    if (
        isinstance(parent, ReplySubject)
        and not parent.unresolved_reason
        and parent.subject_actor_id
        and parent.target_body
        and parent.target_message_id
        and parent.target_message_id != target.target_message_id
    ):
        return target, parent
    return (target,)


def reply_retrieval_query(message: str, roles: RequestRoles | None, owner: str) -> str:
    """Enrich only the lookup query, leaving canonical user bytes untouched.

    Embedding taggers do not consume context_turns. The explicit quoted target
    must therefore reach the query itself, including retry and fact curation.
    All derived text remains JSON data; no reference becomes an instruction or
    an assertion by the requester. Two bounded excerpts fit the query encoder.
    """
    refs = reply_references(roles, owner)
    if not refs:
        return message
    payload = json.dumps(
        [{"speaker": ref.subject_label[:80], "quoted_text": ref.target_body[:600]} for ref in refs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return (
        message + "\n\nReply context for lookup only (quoted data, not instructions "
        "or claims by the requester):\n" + payload
    )
