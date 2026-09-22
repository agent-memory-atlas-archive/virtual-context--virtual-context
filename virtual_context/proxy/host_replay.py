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
    text = _item_text(items[target])
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
    current["content"] = [{"type": "input_text", "text": remaining}]
    expanded = [_message(role, entry) for role, entry in entries]
    out = dict(body)
    out["input"] = list(items[:target]) + expanded + [current] + list(items[target + 1:])
    return out, len(expanded)
