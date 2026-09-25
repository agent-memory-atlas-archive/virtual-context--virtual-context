"""The flush gate's decision for one user turn, shared by the calls of its tool loop.

When the prompt cache is cold the gate lets the proxy reshape the payload
up to a flushed boundary; while it is warm it holds all reshaping. A turn's
first call makes the cache warm, so its continuation calls would be sent
without the reshaping the first call applied and would miss the cache the
first call wrote. The boundary the first call used is kept per turn so its
continuations reshape exactly the same way.
"""

from __future__ import annotations

import hashlib


def turn_gate_key(user_text: str, turn_count: int) -> str:
    """Key for one user turn: its text and its position in the conversation."""
    if not user_text:
        return ""
    digest = hashlib.sha256(f"{turn_count}\n{user_text}".encode("utf-8")).hexdigest()[:32]
    return f"flush_gate:{digest}"


def remember_turn_gate(provider, conversation_id: str, key: str, *, flushed_prefix: int) -> None:
    if provider is None or not key or not callable(getattr(type(provider), "save_retrieval_memo", None)):
        return
    provider.save_retrieval_memo(conversation_id, key, {"flushed_prefix": int(flushed_prefix)})


def recall_turn_gate(provider, conversation_id: str, key: str) -> int | None:
    """The flushed boundary this turn's first call reshaped with, or None."""
    if provider is None or not key or not callable(getattr(type(provider), "load_retrieval_memo", None)):
        return None
    memo = provider.load_retrieval_memo(conversation_id, key)
    value = memo.get("flushed_prefix") if isinstance(memo, dict) else None
    return int(value) if isinstance(value, int) else None
