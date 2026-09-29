"""Rewrite a stored turn chain into the wire format of the request restoring it.

A collapsed turn is stored exactly as its host sent it. That may be a different
wire format than the request that later restores it (a chat-completions chain
restored into a Responses request), and hosts attach their own bookkeeping
fields to messages. Providers reject both. ``conform_chain`` returns the chain
as items the target format accepts, keeping only the fields that format
defines, or ``None`` when it cannot be expressed there faithfully.
"""

from __future__ import annotations

import json

_CHAT_MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}
_ANTHROPIC_MESSAGE_KEYS = {"role", "content"}
_GEMINI_MESSAGE_KEYS = {"role", "parts"}
_RESPONSES_KEYS = {
    "message": ("type", "role", "content"),
    "function_call": ("type", "call_id", "name", "arguments"),
    "function_call_output": ("type", "call_id", "output"),
    "custom_tool_call": ("type", "call_id", "name", "input"),
    "custom_tool_call_output": ("type", "call_id", "output"),
}
_RESPONSES_TEXT_PARTS = {"input_text", "output_text"}


def _shape(message: dict) -> str:
    if "type" in message and "role" not in message or message.get("type") in _RESPONSES_KEYS or message.get("type") == "reasoning":
        return "responses"
    if "parts" in message:
        return "gemini"
    content = message.get("content")
    if message.get("role") == "tool" or "tool_calls" in message:
        return "chat"
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result", "thinking", "redacted_thinking")
        for block in content
    ):
        return "anthropic"
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in _RESPONSES_TEXT_PARTS for block in content
    ):
        return "responses"
    return "plain"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                texts.append(_text_of(block.get("content")))
        return "\n".join(t for t in texts if t)
    return ""


def _message(role: str, text: str) -> dict:
    part = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": part, "text": text}]}


def _to_responses(chain: list[dict]) -> list[dict] | None:
    items: list[dict] = []
    for message in chain:
        shape = _shape(message)
        role = message.get("role")
        if shape == "responses":
            kind = message.get("type", "message")
            if kind == "reasoning":
                continue  # opaque and bound to the original context
            keys = _RESPONSES_KEYS.get(kind)
            if keys is None:
                return None
            item = {key: message[key] for key in keys if key in message}
            item["type"] = kind
            if kind == "message":
                text = _text_of(message.get("content"))
                if not text or item.get("role") not in ("user", "assistant", "developer", "system"):
                    continue
                item = _message("assistant" if item["role"] == "assistant" else item["role"], text)
            items.append(item)
        elif shape in ("chat", "plain"):
            if role in ("user", "system", "developer"):
                text = _text_of(message.get("content"))
                if text:
                    items.append(_message("developer" if role == "system" else role, text))
            elif role == "assistant":
                text = _text_of(message.get("content"))
                if text:
                    items.append(_message("assistant", text))
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if not call.get("id") or not function.get("name"):
                        return None
                    arguments = function.get("arguments", "")
                    items.append({
                        "type": "function_call", "call_id": call["id"], "name": function["name"],
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    })
            elif role == "tool":
                if not message.get("tool_call_id"):
                    return None
                items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                              "output": _text_of(message.get("content"))})
            else:
                return None
        elif shape == "anthropic":
            texts: list[str] = []
            for block in message.get("content") or []:
                kind = block.get("type") if isinstance(block, dict) else None
                if kind == "text" and block.get("text"):
                    texts.append(block["text"])
                elif kind == "tool_use":
                    if texts:
                        items.append(_message("assistant" if role == "assistant" else "user", "\n".join(texts)))
                        texts = []
                    items.append({"type": "function_call", "call_id": block.get("id", ""), "name": block.get("name", ""),
                                  "arguments": json.dumps(block.get("input", {}))})
                elif kind == "tool_result":
                    items.append({"type": "function_call_output", "call_id": block.get("tool_use_id", ""),
                                  "output": _text_of(block.get("content"))})
            if texts:
                items.append(_message("assistant" if role == "assistant" else "user", "\n".join(texts)))
        else:
            return None
    return _paired_calls_only(items)


def _paired_calls_only(items: list[dict]) -> list[dict] | None:
    calls = {item["call_id"] for item in items if item["type"] in ("function_call", "custom_tool_call")}
    outputs = {item["call_id"] for item in items if item["type"] in ("function_call_output", "custom_tool_call_output")}
    if any(not item.get("call_id") for item in items if item["type"] != "message"):
        return None
    kept = [
        item for item in items
        if item["type"] == "message"
        or (item["type"] in ("function_call", "custom_tool_call") and item["call_id"] in outputs)
        or (item["type"] in ("function_call_output", "custom_tool_call_output") and item["call_id"] in calls)
    ]
    return kept or None


def _same_shape(chain: list[dict], shapes: set[str], keys: set[str]) -> list[dict] | None:
    if not all(_shape(message) in shapes for message in chain):
        return None
    return [{key: value for key, value in message.items() if key in keys} for message in chain]


def conform_chain(chain: list[dict], api_format: str) -> list[dict] | None:
    """``chain`` as ``api_format`` items with only that format's fields, or None."""
    if api_format == "openai_responses":
        return _to_responses(chain)
    if api_format == "openai":
        return _same_shape(chain, {"chat", "plain"}, _CHAT_MESSAGE_KEYS)
    if api_format == "anthropic":
        return _same_shape(chain, {"anthropic", "plain"}, _ANTHROPIC_MESSAGE_KEYS)
    if api_format == "gemini":
        return _same_shape(chain, {"gemini"}, _GEMINI_MESSAGE_KEYS)
    return None


def chain_as_text(chain: list[dict]) -> str:
    """A readable transcript of ``chain`` for returning it as a tool result."""
    lines: list[str] = []
    for message in chain:
        role = message.get("role") or message.get("type") or "item"
        if message.get("type") == "reasoning":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(f"[tool call] {function.get('name', '')} {function.get('arguments', '')}")
        if message.get("type") in ("function_call", "custom_tool_call"):
            lines.append(f"[tool call] {message.get('name', '')} {message.get('arguments', message.get('input', ''))}")
            continue
        if message.get("type") in ("function_call_output", "custom_tool_call_output"):
            lines.append(f"[tool result] {_text_of(message.get('output'))}")
            continue
        content = message.get("content") if "content" in message else message.get("parts")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    lines.append(f"[tool call] {block.get('name', '')} {json.dumps(block.get('input', {}))}")
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    lines.append(f"[tool result] {_text_of(block.get('content'))}")
        text = _text_of(content)
        if text:
            lines.append(f"[{'tool result' if role == 'tool' else role}] {text}")
    return "\n".join(lines)
