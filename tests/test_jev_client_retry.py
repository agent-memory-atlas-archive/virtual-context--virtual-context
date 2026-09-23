"""A judgment call resends once when a pooled connection was already closed."""

from __future__ import annotations

import pytest

from virtual_context.types import JudgmentConfig


@pytest.mark.regression("BUG-083")
def test_a_dropped_pooled_connection_is_retried_once():
    import httpx
    from virtual_context.core.judgment import JevClient

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json={"model": "jev", "answers": {"q": {"type": "noul", "noul": 0.8}},
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = JevClient(JudgmentConfig(), environ={"TYPESAFE_API_KEY": "k"},
                       http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    resp = client.ask(seam="fact_curation", state={}, questions={"q": {}})
    assert resp is not None and len(calls) == 2
