"""Engines in one process share a token counter per mode."""
from __future__ import annotations

import pytest

from virtual_context.token_counter import create_token_counter


@pytest.mark.regression("BUG-123")
def test_the_same_mode_returns_the_same_counter():
    assert create_token_counter("estimate") is create_token_counter("estimate")
    pytest.importorskip("tiktoken")
    first = create_token_counter("tiktoken")
    assert first is create_token_counter("tiktoken")
    assert first("hello world") > 0
