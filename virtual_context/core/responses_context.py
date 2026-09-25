"""Where VC's context block goes in a Responses-API request.

Providers reuse a cached prompt only while its beginning is byte-identical.
VC's context block changes from turn to turn, so it goes after the history:
a developer item right after the latest user message. Everything the host
sent before that message stays an unchanged prefix. The block is the same
for every round of a turn's tool loop, so the loop's calls and outputs go
after it and each round extends the cached prefix of the previous one,
block included. Any earlier VC block, in the items or in ``instructions``,
is removed first so blocks never stack.
"""

from __future__ import annotations

import re

VC_BLOCK_RE = re.compile(
    r"<(?:virtual-context|system-reminder)>\n.*?\n</(?:virtual-context|system-reminder)>",
    re.DOTALL,
)
_VC_ITEM_MARK = "vc_context"


def strip_vc_text(text: str) -> str:
    """Remove VC blocks and the blank-line separator VC put beside them.

    The rest of the text is returned byte-for-byte, including the client's
    own leading or trailing whitespace, so a cleaned prompt matches what the
    client originally sent.
    """
    if not VC_BLOCK_RE.search(text):
        return text
    pattern = VC_BLOCK_RE.pattern
    text = re.sub(r"\n{1,2}" + pattern, "", text, flags=re.DOTALL)
    text = re.sub(pattern + r"\n{1,2}", "", text, flags=re.DOTALL)
    return VC_BLOCK_RE.sub("", text)


def _item_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _is_vc_item(item: object) -> bool:
    if not isinstance(item, dict) or item.get("role") != "developer":
        return False
    if item.get("id") == _VC_ITEM_MARK:
        return True
    text = _item_text(item).strip()
    match = VC_BLOCK_RE.fullmatch(text)
    return bool(match) and text.startswith("<system-reminder>")


def place_context_block(body: dict, prepend_text: str) -> None:
    """Put VC's context block into ``body`` in place, replacing any earlier one."""
    block = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
    instructions = body.get("instructions")
    items = body.get("input")
    if not isinstance(items, list):
        # No item list to place it in: the instructions are the only slot.
        if isinstance(instructions, str) and VC_BLOCK_RE.search(instructions):
            body["instructions"] = VC_BLOCK_RE.sub(lambda _m: block, instructions, count=1)
        elif isinstance(instructions, str) and instructions:
            body["instructions"] = f"{block}\n\n{instructions}"
        else:
            body["instructions"] = block
        return
    if isinstance(instructions, str) and VC_BLOCK_RE.search(instructions):
        cleaned = strip_vc_text(instructions)
        if cleaned.strip():
            body["instructions"] = cleaned
        else:
            body.pop("instructions", None)
    kept = [item for item in items if not _is_vc_item(item)]
    at = next(
        (i + 1 for i in range(len(kept) - 1, -1, -1)
         if isinstance(kept[i], dict) and kept[i].get("role") == "user"),
        len(kept),
    )
    kept.insert(at, {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": block}],
    })
    items[:] = kept
