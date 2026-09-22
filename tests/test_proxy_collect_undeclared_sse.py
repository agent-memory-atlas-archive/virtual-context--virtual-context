"""Upstream response collection recognizes SSE that arrives without a content type."""
import asyncio
import json

import httpx
import pytest

from virtual_context.proxy.continuation import ContinuationError
from virtual_context.proxy.response_codec import collect_response


class _Resp:
    def __init__(self, chunks, headers=None, status=200):
        self._chunks = chunks
        self.headers = httpx.Headers(headers or {})
        self.status_code = status
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


RESPONSE = {"id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}]}


def _events(terminal="response.completed"):
    return [{"type": "response.created", "response": {**RESPONSE, "status": "in_progress", "output": []}},
            {"type": "response.output_text.delta", "delta": "OK"},
            {"type": terminal, "response": RESPONSE}]


@pytest.mark.parametrize("headers", [{}, {"content-type": "text/event-stream; charset=utf-8"}])
def test_sse_is_collected_with_or_without_a_declared_content_type(headers):
    resp = _Resp([_sse(_events())], headers)
    assert asyncio.run(collect_response(resp, "openai_responses")) == RESPONSE
    assert resp.closed


def test_sniffing_survives_tiny_first_chunks_and_accepts_response_done():
    raw = _sse(_events("response.done"))
    chunks = [raw[:2], raw[2:5], raw[5:9]] + [raw[i:i + 1000] for i in range(9, len(raw), 1000)]
    assert asyncio.run(collect_response(_Resp(chunks), "openai_responses")) == RESPONSE


def test_json_without_a_content_type_still_parses_as_json():
    resp = _Resp([json.dumps(RESPONSE).encode()])
    assert asyncio.run(collect_response(resp, "openai_responses")) == RESPONSE


def test_undeclared_non_sse_garbage_is_still_rejected():
    with pytest.raises(ContinuationError):
        asyncio.run(collect_response(_Resp([b"<html>nope</html>"]), "openai_responses"))


def test_output_is_rebuilt_from_item_events_when_the_terminal_response_has_none():
    message = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "OK2", "annotations": []}]}
    call = {"id": "fc_1", "type": "function_call", "call_id": "c1", "name": "vc_recall_all", "arguments": "{\"q\":1}"}
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "abc"}
    events = [
        {"type": "response.created", "response": {"id": "r", "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": {**reasoning, "encrypted_content": ""}},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {"type": "response.output_item.added", "output_index": 1, "item": {**message, "content": []}},
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "OK"},
        {"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "2"},
        {"type": "response.output_item.done", "output_index": 1, "item": message},
        {"type": "response.output_item.added", "output_index": 2, "item": {**call, "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": "{\"q\":1}"},
        {"type": "response.done", "response": {"id": "r", "status": "completed", "output": [], "usage": {"output_tokens": 6}}},
    ]
    result = asyncio.run(collect_response(_Resp([_sse(events)]), "openai_responses"))
    assert result["output"] == [reasoning, message, call]
    assert result["usage"] == {"output_tokens": 6}


def test_a_populated_terminal_output_is_kept_as_is():
    events = [
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "partial"},
        {"type": "response.completed", "response": RESPONSE},
    ]
    assert asyncio.run(collect_response(_Resp([_sse(events)]), "openai_responses")) == RESPONSE


def test_leading_blank_lines_do_not_hide_an_undeclared_sse_body():
    raw = _sse(_events())
    chunks = [b"\n" * 8, b"\n\n", raw[:3], raw[3:]]
    assert asyncio.run(collect_response(_Resp(chunks), "openai_responses")) == RESPONSE
