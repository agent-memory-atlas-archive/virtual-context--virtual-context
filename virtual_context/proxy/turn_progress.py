"""One user message, one stored turn, however many tool rounds it takes.

A host that starts a fresh model thread for every user message sends only the
current turn as items: one real user message, then (once tools run) the
model's interim notes and the tool calls and outputs of each round. Every
round is a separate request. Storing each round's response as a finished turn
wrote the user message several times, once per round, paired with whatever
interim note that round produced.

For such a *fresh-thread* request this module decides:

* whether the request is mid-turn (tool activity follows the user message),
  so interim assistant notes are not stored as that message's reply;
* whether a response ends in tool calls, so it does not finish the turn;
* the interim notes to fold into the final answer when the turn does finish.

Requests that carry more than one real user message (clients that resend the
whole conversation) are outside this rule and keep their existing behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnProgress:
    fresh_thread: bool
    tool_activity: bool
    user_index: int = -1
    interim_texts: list[str] = field(default_factory=list)

    @property
    def in_progress(self) -> bool:
        return self.fresh_thread and self.tool_activity


def _is_tool_item(msg: Any, fmt_name: str) -> bool:
    if not isinstance(msg, dict):
        return False
    if fmt_name == "openai_responses":
        kind = str(msg.get("type", "") or "")
        return kind.endswith("_call") or kind.endswith("_output")
    if fmt_name == "openai":
        return msg.get("role") == "tool" or bool(msg.get("tool_calls"))
    if fmt_name == "gemini":
        parts = msg.get("parts") or []
        return any(isinstance(p, dict) and ("functionCall" in p or "function_call" in p
                                            or "functionResponse" in p or "function_response" in p)
                   for p in parts)
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result", "server_tool_use")
        for b in content
    )


def _is_real_user(msg: Any, fmt: Any) -> bool:
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    if fmt.name == "openai_responses" and msg.get("type", "message") != "message":
        return False
    is_host_context = getattr(fmt, "_is_host_context_item", None)
    if callable(is_host_context) and is_host_context(msg):
        return False
    content = msg.get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return False
    try:
        return bool(fmt.extract_message_text(msg).strip())
    except Exception:
        return False


def turn_progress(body: dict, fmt: Any) -> TurnProgress:
    """Classify the request's current turn."""
    try:
        messages = fmt.get_messages(body)
    except Exception:
        messages = []
    if not isinstance(messages, list):
        return TurnProgress(fresh_thread=False, tool_activity=False)
    users = [i for i, m in enumerate(messages) if _is_real_user(m, fmt)]
    if len(users) != 1:
        return TurnProgress(fresh_thread=False, tool_activity=False)
    at = users[0]
    after = messages[at + 1:]
    tool_activity = any(_is_tool_item(m, fmt.name) for m in after)
    interim: list[str] = []
    if tool_activity:
        for m in after:
            if isinstance(m, dict) and m.get("role") in ("assistant", "model"):
                try:
                    text = fmt.extract_message_text(m).strip()
                except Exception:
                    text = ""
                if text:
                    interim.append(text)
    return TurnProgress(fresh_thread=True, tool_activity=tool_activity, user_index=at, interim_texts=interim)


def response_ends_with_tool_calls(response: Any, api_format: str) -> bool:
    """True when the model handed control back for tool execution."""
    if not isinstance(response, dict):
        return False
    if api_format == "openai_responses":
        return any(isinstance(item, dict) and str(item.get("type", "")).endswith("_call")
                   for item in response.get("output") or [])
    if api_format == "anthropic":
        if response.get("stop_reason") == "tool_use":
            return True
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in response.get("content") or [])
    if api_format == "openai":
        choice = (response.get("choices") or [{}])[0] or {}
        return choice.get("finish_reason") == "tool_calls" or bool((choice.get("message") or {}).get("tool_calls"))
    if api_format == "gemini":
        parts = (((response.get("candidates") or [{}])[0] or {}).get("content") or {}).get("parts") or []
        return any(isinstance(p, dict) and ("functionCall" in p or "function_call" in p) for p in parts)
    return False


def stream_ends_with_tool_calls(raw_events: list, api_format: str) -> bool:
    """Same decision for a streamed response, from its raw events."""
    markers = {
        "openai_responses": ('"function_call"', '"custom_tool_call"', '"local_shell_call"', '"tool_search_call"'),
        "anthropic": ('"tool_use"',),
        "openai": ('"tool_calls"',),
        "gemini": ('"functionCall"', '"function_call"'),
    }.get(api_format, ())
    for event in raw_events or []:
        text = event if isinstance(event, str) else json.dumps(event, default=str)
        if any(marker in text for marker in markers):
            return True
    return False
