"""Exact-source replay across an explicitly audited audience reassignment.

These checks authorize only a named immutable source. They do not resolve
request audiences or widen any retrieval predicate.
"""

from __future__ import annotations

from ..core.exceptions import CanonicalSourceConflict
from ..types import AUDIENCE_ATTRIBUTION_VERSION
from .audience_reassignment import get_audience_reassignment, source_fingerprint


def effective_attested_audience(conn, membership, *, dialect: str) -> str:
    """Validate a complete pair's receipt and return its current audience."""
    membership = dict(membership)
    original = str(membership["audience_conversation_id"] or "")
    user_id = str(membership["canonical_turn_id"])
    assistant_id = str(membership["assistant_canonical_turn_id"] or "")
    user_receipt = get_audience_reassignment(conn, user_id, dialect)
    assistant_receipt = get_audience_reassignment(conn, assistant_id, dialect)
    if user_receipt is None and assistant_receipt is None:
        return original
    if user_receipt is None or assistant_receipt is None:
        raise CanonicalSourceConflict("audience reassignment lacks its exact pair")

    target = str(user_receipt["to_audience"] or "")
    fingerprint = source_fingerprint(membership)
    placeholder = "%s" if dialect == "postgres" else "?"
    for receipt, canonical_id, expected_hash in (
        (user_receipt, user_id, membership["canonical_turn_hash"]),
        (assistant_receipt, assistant_id, membership["assistant_turn_hash"]),
    ):
        if (
            not target
            or str(receipt["canonical_turn_id"]) != canonical_id
            or str(receipt["tenant_id"]) != str(membership["tenant_id"])
            or str(receipt["to_audience"]) != target
            or str(receipt["turn_hash"]) != str(expected_hash)
            or receipt["source_fingerprint"] != fingerprint
            or receipt["operation_id"] != user_receipt["operation_id"]
            or receipt["manifest_digest"] != user_receipt["manifest_digest"]
            or int(receipt["to_attribution_version"]) != AUDIENCE_ATTRIBUTION_VERSION
        ):
            raise CanonicalSourceConflict("audience reassignment disagrees with source proof")
        current = conn.execute(
            "SELECT ct.audience_conversation_id, ct.audience_attribution_version, "
            "ct.turn_hash, c.tenant_id FROM canonical_turns ct "
            "JOIN conversations c ON c.conversation_id = ct.conversation_id "
            f"WHERE ct.canonical_turn_id = {placeholder}",
            (canonical_id,),
        ).fetchone()
        if (
            current is None
            or str(current["tenant_id"]) != str(membership["tenant_id"])
            or str(current["audience_conversation_id"] or "") != target
            or int(current["audience_attribution_version"] or 0)
            != AUDIENCE_ATTRIBUTION_VERSION
            or str(current["turn_hash"] or "") != str(expected_hash)
        ):
            raise CanonicalSourceConflict("canonical audience disagrees with reassignment")
    if str(user_receipt["from_audience"]) != original:
        raise CanonicalSourceConflict("reassignment does not preserve the original audience")
    return target


def verify_source_replay_audience(
    conn, *, canonical_turn_id: str, current_audience: str,
    incoming_audience: str, dialect: str,
) -> str:
    """Return the original audience after exact original/current replay proof.

    The returned original is used when comparing immutable ledger fields; it
    is never substituted into the incoming request's retrieval scope.
    """
    placeholder = "%s" if dialect == "postgres" else "?"
    membership = conn.execute(
        "SELECT * FROM canonical_message_sources "
        f"WHERE canonical_turn_id = {placeholder}",
        (canonical_turn_id,),
    ).fetchone()
    if membership is None:
        raise CanonicalSourceConflict("canonical source membership is missing")
    original = str(membership["audience_conversation_id"] or "")
    effective = effective_attested_audience(conn, membership, dialect=dialect)
    if current_audience != effective or incoming_audience not in {original, effective}:
        raise CanonicalSourceConflict("source replay has an unauthorized audience")
    return original
