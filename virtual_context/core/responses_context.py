"""Where VC's context block goes in a Responses-API request.

Providers reuse a cached prompt only while its beginning is byte-identical,
and a Responses request is read as instructions, tools, then input items. VC's
context block changes from call to call, so putting it in ``instructions``
(the very start) made every call a cache miss, including the host's large,
unchanging developer prompts and tool catalog. The block is placed instead as
a developer input item after the request's leading instruction items and
before the conversation, so everything ahead of it stays cacheable.
"""

from __future__ import annotations

import re

VC_BLOCK_RE = re.compile(
    r"<(?:virtual-context|system-reminder)>\n.*?\n</(?:virtual-context|system-reminder)>",
    re.DOTALL,
)
_VC_ITEM_MARK = "vc_context"


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


def _is_host_scaffolding(item: dict) -> bool:
    from ..proxy.formats import get_format

    check = getattr(get_format("openai_responses"), "_is_host_context_item", None)
    return bool(callable(check) and check(item))


def _is_leading_instruction(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("role") in ("system", "developer"):
        return True
    kind = str(item.get("type", "message"))
    if kind != "message":
        return not kind.endswith(("_call", "_output"))
    return _is_host_scaffolding(item)


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
        cleaned = VC_BLOCK_RE.sub("", instructions, count=1).strip()
        if cleaned:
            body["instructions"] = cleaned
        else:
            body.pop("instructions", None)
    kept = [item for item in items if not _is_vc_item(item)]
    at = 0
    while at < len(kept) and _is_leading_instruction(kept[at]):
        at += 1
    vc_item = {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": block}],
    }
    kept.insert(at, vc_item)
    items[:] = kept
