"""Where VC's context block goes in an OpenAI Chat Completions request.

Providers reuse a cached prompt only while its beginning is byte-identical,
and VC's context block changes from call to call. It is therefore appended
to the latest user message, so the system prompt and every earlier message
stay an unchanged prefix. Any earlier VC block, in any message, is removed
first so blocks never stack.
"""

from __future__ import annotations

from .responses_context import VC_BLOCK_RE


def _without_block(content):
    if isinstance(content, str):
        return VC_BLOCK_RE.sub("", content).strip() if VC_BLOCK_RE.search(content) else content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and VC_BLOCK_RE.search(part["text"]):
                text = VC_BLOCK_RE.sub("", part["text"]).strip()
                if text:
                    out.append({**part, "text": text})
                continue
            out.append(part)
        return out
    return content


def place_context_block(messages: list, prepend_text: str) -> None:
    """Put VC's context block on the latest user message, in place."""
    block = f"<system-reminder>\n{prepend_text}\n</system-reminder>"
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and "content" in msg:
            cleaned = _without_block(msg["content"])
            if cleaned is not msg["content"]:
                messages[i] = {**msg, "content": cleaned}
    messages[:] = [m for m in messages if not (isinstance(m, dict) and m.get("content") in ("", []) and m.get("role") in ("system", "user") and not m.get("tool_calls"))]
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            messages[i] = {**msg, "content": list(content) + [{"type": "text", "text": block}]}
        else:
            text = content if isinstance(content, str) else ""
            messages[i] = {**msg, "content": f"{text}\n\n{block}" if text else block}
        return
    messages.append({"role": "user", "content": block})
