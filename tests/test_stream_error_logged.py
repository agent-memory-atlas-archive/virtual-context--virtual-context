"""A provider error inside a stream is logged with the provider's own detail."""
import asyncio
import json
import logging

import httpx
import pytest

from virtual_context.proxy.continuation import ContinuationError
from virtual_context.proxy.response_codec import collect_response


@pytest.mark.regression("PROXY-041")
def test_mid_stream_error_detail_is_logged(caplog):
    events = [
        {"type": "response.created", "response": {"id": "r1", "error": None}},
        {"type": "response.failed", "response": {"id": "r1", "error": {"code": "server_error", "message": "upstream overloaded"}}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    with caplog.at_level(logging.WARNING, logger="virtual_context.proxy.response_codec"):
        with pytest.raises(ContinuationError):
            asyncio.run(collect_response(response, "openai_responses"))

    assert "upstream overloaded" in caplog.text
    assert "server_error" in caplog.text
