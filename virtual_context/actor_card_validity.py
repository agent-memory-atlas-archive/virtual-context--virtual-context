"""Explicit actor-card validity windows; source prose never supplies dates here."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)


def _bound(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise ValueError("Actor-card validity must be an aware ISO timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Invalid actor-card validity timestamp") from exc


def normalize_actor_card_validity(
    valid_from: str | None, expires_at: str | None,
) -> tuple[str | None, str | None]:
    """Validate explicit bounds and normalize them to UTC without extending them."""
    start, end = _bound(valid_from), _bound(expires_at)
    if start is not None and end is not None and start >= end:
        raise ValueError("Actor-card validity must end after it starts")
    return (
        start.isoformat(timespec="microseconds") if start is not None else None,
        end.isoformat(timespec="microseconds") if end is not None else None,
    )


def _window(valid_from, expires_at, now):
    start, end = normalize_actor_card_validity(valid_from, expires_at)
    current = datetime.now(timezone.utc) if now is None else now
    if not isinstance(current, datetime) or current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Actor-card validity requires an aware clock")
    return _bound(start), _bound(end), current.astimezone(timezone.utc)


def actor_card_is_active(
    valid_from: str | None, expires_at: str | None, *, now: datetime | None = None,
) -> bool:
    """Start inclusive, end exclusive; malformed stored windows are hidden."""
    try:
        start, end, current = _window(valid_from, expires_at, now)
        return (start is None or start <= current) and (end is None or current < end)
    except (ValueError, TypeError, OverflowError):
        return False


def actor_card_is_unexpired(
    valid_from: str | None, expires_at: str | None, *, now: datetime | None = None,
) -> bool:
    """Keep valid future entries available for carryover without serving them."""
    try:
        _, end, current = _window(valid_from, expires_at, now)
        return end is None or current < end
    except (ValueError, TypeError, OverflowError):
        return False
