"""Select replayed history turns that carry a host-owned speaker identity.

A host replays its session history on every request. On a source-attested
group route only the current turn is admitted, so a turn the host answered
without reaching this engine (for example while it was unreachable) would
never be stored. Each replayed user message can carry a host speaker tag,
rendered from the host's own message metadata::

    <message-speaker source="host-session-metadata" authority="attribution-only">
    {"name":"...","actor_id":"actor:discord:<id>","message_id":"<snowflake>"}
    </message-speaker>

Member-typed lookalikes are escaped before the host renders history, so a tag
that parses here was emitted by the host. A turn is selected only when its tag
names both a platform actor and a platform message id, the message id is not
already stored, and it comes after the newest message the window shares with
storage. Its replies travel with it; untagged turns are never selected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

_TAG_RE = re.compile(
    r'\A\s*<message-speaker source="host-session-metadata" '
    r'authority="attribution-only">\n(\{[^\n]*\})\n</message-speaker>\n?'
)
_ACTOR_RE = re.compile(r"^actor:discord:\d{1,32}$")
_MESSAGE_ID_RE = re.compile(r"^\d{15,24}$")


@dataclass(frozen=True)
class HostSpeaker:
    actor_id: str
    message_id: str
    name: str
    body: str


@dataclass
class MissedTurn:
    speaker: HostSpeaker
    replies: list[str] = field(default_factory=list)


def read_host_speaker(text: str) -> HostSpeaker | None:
    """The host identity at the start of a replayed user message, or None."""
    if not isinstance(text, str):
        return None
    match = _TAG_RE.match(text)
    if not match:
        return None
    try:
        speaker = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(speaker, dict):
        return None
    actor_id = speaker.get("actor_id")
    message_id = speaker.get("message_id")
    name = speaker.get("name")
    if not isinstance(actor_id, str) or not _ACTOR_RE.match(actor_id):
        return None
    if not isinstance(message_id, str) or not _MESSAGE_ID_RE.match(message_id):
        return None
    body = text[match.end():].strip()
    if not body:
        return None
    return HostSpeaker(
        actor_id=actor_id,
        message_id=message_id,
        name=name.strip() if isinstance(name, str) else "",
        body=body,
    )


_REPLAY_ROLES = {"user": "user", "assistant": "assistant"}
_REPLY_NOISE_RE = re.compile(r"^\s*(?:\[thinking content omitted\]|tool call:).*$", re.M)


def replay_history(block: str) -> list[tuple[str, str]]:
    """``(role, text)`` sections of a host history block, user and assistant only.

    Tool calls, tool results and system sections are the host's working
    traffic, not conversation, and are dropped.
    """
    from ..proxy.host_replay import _SECTION_RE

    heads = list(_SECTION_RE.finditer(block or ""))
    out: list[tuple[str, str]] = []
    for index, head in enumerate(heads):
        role = _REPLAY_ROLES.get(head.group(1))
        if role is None:
            continue
        end = heads[index + 1].start() if index + 1 < len(heads) else len(block)
        out.append((role, block[head.end():end].strip()))
    return out


def final_reply(texts: list[str]) -> str:
    """The last assistant text that says something once working lines are removed."""
    for text in reversed(texts):
        cleaned = _REPLY_NOISE_RE.sub("", text or "").strip()
        if cleaned:
            return cleaned
    return ""


def select_missed_turns(
    history: Iterable[tuple[str, str]],
    stored_message_ids: Callable[[list[str]], set[str]],
) -> list[MissedTurn]:
    """Tagged turns after the newest stored one, in history order.

    ``history`` is ``(role, text)`` for the replayed window, excluding the
    current turn. ``stored_message_ids`` returns which of the given platform
    message ids already exist in storage.
    """
    turns: list[tuple[HostSpeaker | None, list[str]]] = []
    for role, text in history:
        if role == "user":
            turns.append((read_host_speaker(text), []))
        elif role == "assistant" and turns and (text or "").strip():
            turns[-1][1].append(text.strip())
    ids = [speaker.message_id for speaker, _ in turns if speaker is not None]
    if not ids:
        return []
    stored = stored_message_ids(ids)
    anchor = max(
        (index for index, (speaker, _) in enumerate(turns)
         if speaker is not None and speaker.message_id in stored),
        default=None,
    )
    if anchor is None:
        return []
    missed: list[MissedTurn] = []
    seen: set[str] = set()
    for speaker, replies in turns[anchor + 1:]:
        if speaker is None or speaker.message_id in stored or speaker.message_id in seen:
            continue
        seen.add(speaker.message_id)
        reply = final_reply(replies)
        missed.append(MissedTurn(speaker=speaker, replies=[reply] if reply else []))
    return missed
