"""Responses streams are relayed live; VC tool rounds stay inside the same stream."""
from __future__ import annotations

import json
from unittest.mock import ANY, MagicMock, patch

import httpx
import pytest

from virtual_context.config import load_config
from virtual_context.core.turn_tag_index import TurnTagIndex
from virtual_context.proxy.server import create_app
from virtual_context.storage.sqlite import SQLiteStore
from virtual_context.types import AssembledContext, EngineState, PagingConfig


@pytest.fixture
def responses_client(tmp_path):
    from starlette.testclient import TestClient
    db_path = str(tmp_path / "store.db")
    with patch("virtual_context.proxy.server.VirtualContextEngine") as MockEngine:
        config = load_config(config_dict={
            "context_window": 10000,
            "storage_root": str(tmp_path),
            "storage": {"backend": "sqlite", "sqlite": {"path": db_path}},
            "tag_generator": {"type": "keyword"},
        })
        config.paging = PagingConfig(enabled=True, autonomous_models=["gpt-5"])
        engine = MagicMock()
        engine.config = config
        engine._store = SQLiteStore(db_path)
        engine._store.upsert_conversation(tenant_id="", conversation_id=config.conversation_id)
        engine._engine_state = EngineState(lifecycle_epoch=1)
        engine.on_message_inbound.return_value = AssembledContext()
        engine.on_turn_complete.return_value = None
        engine.tag_turn.return_value = None
        engine._turn_tag_index = TurnTagIndex()
        engine._retrieval._resolve_paging_mode.return_value = "autonomous"
        engine._engine_state.compacted_prefix_messages = 0
        engine.expand_topic.return_value = {"tag": "database", "depth": "full", "tokens_added": 500, "tokens_evicted": 0, "evicted_tags": []}
        MockEngine.return_value = engine
        app = create_app(upstream="http://fake:9999", config_path=None)
    with TestClient(app) as client:
        yield client, engine
    engine._store.close()


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _message_round(text, rid="resp_2", seq0=0):
    item = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    started = {**item, "content": []}
    return [
        {"type": "response.created", "sequence_number": seq0, "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "sequence_number": seq0 + 1, "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "sequence_number": seq0 + 2, "output_index": 0, "item": started},
        {"type": "response.content_part.added", "sequence_number": seq0 + 3, "output_index": 0, "item_id": "msg_1", "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
        {"type": "response.output_text.delta", "sequence_number": seq0 + 4, "output_index": 0, "item_id": "msg_1", "content_index": 0, "delta": text},
        {"type": "response.output_text.done", "sequence_number": seq0 + 5, "output_index": 0, "item_id": "msg_1", "content_index": 0, "text": text},
        {"type": "response.content_part.done", "sequence_number": seq0 + 6, "output_index": 0, "item_id": "msg_1", "content_index": 0, "part": item["content"][0]},
        {"type": "response.output_item.done", "sequence_number": seq0 + 7, "output_index": 0, "item": item},
        {"type": "response.completed", "sequence_number": seq0 + 8, "response": {"id": rid, "object": "response", "status": "completed", "output": [item], "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}}},
    ]


def _vc_call_round(rid="resp_1"):
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "vc_expand_topic",
            "arguments": '{"tag": "database", "depth": "full"}', "status": "completed"}
    return [
        {"type": "response.created", "sequence_number": 0, "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "sequence_number": 1, "response": {"id": rid, "object": "response", "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": {**call, "arguments": "", "status": "in_progress"}},
        {"type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 0, "item_id": "fc_1", "delta": call["arguments"]},
        {"type": "response.function_call_arguments.done", "sequence_number": 4, "output_index": 0, "item_id": "fc_1", "arguments": call["arguments"]},
        {"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": call},
        {"type": "response.completed", "sequence_number": 6, "response": {"id": rid, "object": "response", "status": "completed", "output": [call], "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}}},
    ]


class _Rounds:
    """Serve one SSE body per upstream request, in order."""

    def __init__(self, bodies, statuses=None):
        self.bodies = list(bodies)
        self.statuses = list(statuses or [200] * len(bodies))
        self.requests = []

    def __enter__(self):
        async def respond(request):
            self.requests.append(json.loads(request.content))
            i = len(self.requests) - 1
            body = self.bodies[i]
            if isinstance(body, dict):
                return httpx.Response(self.statuses[i], json=body)
            return httpx.Response(self.statuses[i], headers={"content-type": "text/event-stream"}, content=body)
        self._patch = patch("httpx.AsyncClient._transport_for_url", return_value=httpx.MockTransport(respond))
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def _post(client):
    return client.post("/backend-api/codex/responses", json={
        "model": "gpt-5.6-sol", "stream": True,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Tell me about the database"}]}],
    })


def _events(raw: bytes):
    out = []
    for block in raw.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def test_vc_tool_round_is_hidden_and_the_continuation_streams_in_the_same_response(responses_client):
    client, engine = responses_client
    with _Rounds([_sse(_vc_call_round()), _sse(_message_round("Here is the database detail."))]) as rounds:
        resp = _post(client)
    assert resp.status_code == 200
    events = _events(resp.content)
    kinds = [e["type"] for e in events]
    assert kinds.count("response.created") == 1 and kinds.count("response.in_progress") == 1
    assert kinds.count("response.completed") == 1 and kinds[-1] == "response.completed"
    assert "vc_expand_topic" not in resp.content.decode()
    assert not any(k.startswith("response.function_call_arguments") for k in kinds)
    assert any(e.get("delta") == "Here is the database detail." for e in events)
    seqs = [e["sequence_number"] for e in events]
    assert seqs == list(range(len(events)))
    message_items = [e for e in events if e["type"] == "response.output_item.added"]
    assert [e["output_index"] for e in message_items] == [0]
    final = events[-1]["response"]
    assert final["id"] == "resp_1"
    assert [i["type"] for i in final["output"]] == ["message"]
    assert final["usage"]["input_tokens"] == 50 and final["usage"]["output_tokens"] == 13
    engine.expand_topic.assert_called_once_with(tag="database", depth="full", speaker_context=ANY)
    assert len(rounds.requests) == 2
    assert rounds.requests[1]["input"][-1]["type"] == "function_call_output"


def test_text_only_stream_is_relayed_with_one_terminal_event(responses_client):
    client, _engine = responses_client
    with _Rounds([_sse(_message_round("Hello there", rid="resp_9"))]):
        resp = _post(client)
    assert resp.status_code == 200
    events = _events(resp.content)
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds.count("response.completed") == 1 and kinds[-1] == "response.completed"
    assert any(e.get("delta") == "Hello there" for e in events)
    assert events[-1]["response"]["id"] == "resp_9"


def test_upstream_failure_before_any_event_keeps_its_real_status(responses_client):
    client, _engine = responses_client
    with _Rounds([{"error": {"message": "upstream exploded"}}], statuses=[502]):
        resp = _post(client)
    assert resp.status_code >= 400
    assert b"response.created" not in resp.content
