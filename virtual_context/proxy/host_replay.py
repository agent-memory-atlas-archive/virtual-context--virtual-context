"""Turn a host's embedded history replay back into real conversation turns.

Some hosts (the OpenClaw Codex harness among them) cannot hand prior turns to
the model as separate messages, so they paste recent history into the current
user message as a ``<conversation_context>`` block. Left in place, that block
is part of the protected current turn and nothing in it can be trimmed. This
module splits it into ordinary user/assistant items placed before the current
message, so the history is managed like any other history: protected recent
turns stay verbatim and older compacted turns can be dropped in favour of
summaries.

Only the outbound payload is rewritten. Ingestion reads the client's own shape
before this runs, so expanded turns are never stored a second time.
"""

from __future__ import annotations

import copy
import re

from ._envelope import _HOST_ASSEMBLED_LABEL

_BLOCK_RE = re.compile(r"<conversation_context>([\s\S]*?)</conversation_context>")
_SECTION_RE = re.compile(
    r"(?m)^[ \t]*\[(user|assistant|toolResult|toolCall|tool|system|compactionSummary)\][ \t]*$"
)
_PREAMBLE_RE = re.compile(
    r"(?m)^[ \t]*" + re.escape(_HOST_ASSEMBLED_LABEL) + r"[ \t]*\n"
    r"(?:[ \t]*Treat the conversation context below[^\n]*\n)?\s*$"
)
_USER_ROLES = {"user", "compactionSummary", "system"}


def parse_replay_block(block: str) -> list[tuple[str, str]]:
    """Ordered ``(role, text)`` entries; non-user sections between users fold into one assistant entry."""
    heads = list(_SECTION_RE.finditer(block))
    entries: list[tuple[str, str]] = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(block)
        text = block[head.end():end].strip()
        label = head.group(1)
        if not text:
            continue
        if label in _USER_ROLES:
            if label != "user":
                text = f"[{label}]\n{text}"
            entries.append(("user", text))
        elif entries and entries[-1][0] == "assistant":
            entries[-1] = ("assistant", entries[-1][1] + "\n\n" + text)
        else:
            entries.append(("assistant", text))
    return entries


def _item_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _message(role: str, text: str) -> dict:
    kind = "input_text" if role == "user" else "output_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def find_replay_block(body: dict) -> str | None:
    """The newest user message's ``<conversation_context>`` block text, if any."""
    items = body.get("input") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return None
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user":
            match = _BLOCK_RE.search(_item_text(item))
            return match.group(1) if match else None
    return None


def expand_host_replay(body: dict) -> tuple[dict, int]:
    """Split the newest user message's replay block into turns before it.

    Returns ``(body, inserted_item_count)``. The input body is never mutated;
    when there is nothing to expand the same object is returned with 0.
    """
    items = body.get("input") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return body, 0
    target = None
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("type", "message") == "message" and item.get("role") == "user":
            if "<conversation_context>" in _item_text(item):
                target = index
            break
    if target is None:
        return body, 0
    # Only the text part holding the block is rewritten; every other part of
    # the current message (images, files, audio, further text) is kept as-is.
    content = items[target].get("content")
    parts = content if isinstance(content, list) else [{"type": "input_text", "text": content}]
    at = next((i for i, part in enumerate(parts)
               if isinstance(part, dict) and "<conversation_context>" in str(part.get("text", ""))), None)
    if at is None:
        return body, 0
    text = parts[at]["text"]
    match = _BLOCK_RE.search(text)
    if not match:
        return body, 0
    entries = parse_replay_block(match.group(1))
    if not entries:
        return body, 0
    before = _PREAMBLE_RE.sub("", text[:match.start()]).rstrip()
    after = text[match.end():].lstrip()
    remaining = (before + "\n\n" + after).strip() if before else after.strip()
    current = copy.deepcopy(items[target])
    new_parts = copy.deepcopy(parts)
    new_parts[at] = {**new_parts[at], "text": remaining}
    current["content"] = new_parts
    expanded = [_message(role, entry) for role, entry in entries]
    out = dict(body)
    out["input"] = list(items[:target]) + expanded + [current] + list(items[target + 1:])
    return out, len(expanded)


def without_host_replayed_groups(replay_messages: list, body: dict, fmt) -> list:
    """Stored turn groups minus those the host already replayed into ``body``.

    A replayed user message names its platform message id in the host speaker
    tag. A stored group whose user message has one of those ids is already in
    the payload, so it is left out whole; other groups are kept whole.
    """
    if not replay_messages:
        return replay_messages
    from ..core.history_catchup import read_host_speaker

    replayed = set()
    for item in fmt.get_messages(body) or []:
        if isinstance(item, dict) and item.get("role") == "user":
            speaker = read_host_speaker(_item_text(item))
            if speaker is not None:
                replayed.add(speaker.message_id)
    if not replayed:
        return replay_messages

    def _meta(message) -> dict:
        metadata = getattr(message, "metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    dropped = {
        _meta(m).get("db_recent_group_key")
        for m in replay_messages
        if m.role == "user" and _meta(m).get("source_message_id") in replayed
    }
    dropped.discard(None)
    return [m for m in replay_messages if _meta(m).get("db_recent_group_key") not in dropped]
