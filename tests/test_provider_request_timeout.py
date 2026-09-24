"""Each LLM provider request carries its provider's own timeout.

Providers and the judgment client share one process-wide HTTP client, which
kept the timeout of whichever caller created it. When a 3 s judgment call
came first, every later provider call (summaries, tags, actor cards) was
also cut off at 3 s.
"""

from __future__ import annotations

import httpx
import pytest

from virtual_context.providers import base
from virtual_context.providers.generic_openai import GenericOpenAIProvider


@pytest.fixture
def fresh_client():
    base._cleanup_client()
    yield
    base._cleanup_client()


@pytest.mark.regression("BUG-093")
def test_a_provider_request_uses_its_own_timeout(fresh_client, monkeypatch):
    shared = base._get_client(3.0)  # created first by a short-timeout caller
    seen = {}

    def fake_post(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(shared, "post", fake_post)
    provider = GenericOpenAIProvider(base_url="http://example.invalid/v1", model="m")
    text, _usage = provider.complete("sys", "user", 10)

    assert text == "ok"
    timeout = seen["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == provider._timeout == 20.0
