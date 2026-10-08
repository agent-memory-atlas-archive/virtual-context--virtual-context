"""The request log records payloads without a per-response memory-state dump."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from starlette.responses import JSONResponse

from virtual_context.proxy.server import create_app

PAYLOAD = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}).encode()


@pytest.mark.regression("PROXY-039")
def test_logged_request_skips_the_session_dump(tmp_path):
    seen = {}

    async def prepare(body, state, fmt, request_metrics, **kwargs):
        return SimpleNamespace(vc_command=False, is_passthrough=False, is_streaming=False, paging_enabled=False,
                               tool_output_find_quote=False, restore_tool_injected=False, enriched_body=body,
                               api_format="openai", turn=1, request_turn=1, turn_id="t", overhead_ms=0,
                               conversation_id="c", speaker_context=None, upstream_limit=200_000,
                               speaker_roster_snapshot=None)

    async def handler(*args, **kwargs):
        seen.update(kwargs)
        return JSONResponse({"ok": True})

    async def run():
        with patch("virtual_context.proxy.server.VirtualContextEngine", side_effect=RuntimeError("no storage")), \
             patch("virtual_context.proxy.server.prepare_payload", side_effect=prepare), \
             patch("virtual_context.proxy.server._handle_non_streaming", side_effect=handler):
            app = create_app("http://upstream.invalid", shared_metrics=SimpleNamespace(owner="default"))
            app.state.request_log_dir = str(tmp_path)
            state = SimpleNamespace(metrics=SimpleNamespace(owner="c"), engine=SimpleNamespace(config=SimpleNamespace(conversation_id="c")))
            app.state.state_resolver = lambda request, body, cid: (state, False)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                return await client.post("/v1/chat/completions", content=PAYLOAD)

    resp = asyncio.run(run())

    assert resp.status_code == 200
    assert list(tmp_path.rglob("*.1-inbound.json"))
    assert "session_log_path" not in seen
    assert not list(tmp_path.rglob("*.session.json"))
