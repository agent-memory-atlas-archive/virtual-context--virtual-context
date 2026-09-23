"""A conversation named out of band is the request's route for audience proof.

A client that names its conversation outside the payload (a signed route)
sends no in-band conversation marker. The resolver that verified the route
records the id it named; that id must reach payload preparation as the
inbound route, or the request can never prove its audience: every summary is
withheld and ingested rows are stored without an audience.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from starlette.responses import JSONResponse

from virtual_context.proxy.server import create_app

ROUTE = "sk:agent:demo:discord:guild:1"
OWNER = "owner-conversation"
BODY = {"model": "gpt-4.1", "input": [{"role": "user", "content": "What did we decide?"}], "max_output_tokens": 64}


def _run(resolver_sets_route: bool) -> dict:
    seen: dict = {}

    async def run():
        state = SimpleNamespace(metrics=None, engine=SimpleNamespace(config=SimpleNamespace(conversation_id=OWNER)))

        def resolver(request, body, inbound):
            seen["resolver_inbound"] = inbound
            request.state.conversation_out_of_band = True
            if resolver_sets_route:
                request.state.conversation_route_id = ROUTE
            return state, False

        async def prepare(body, state, fmt, request_metrics, **kwargs):
            seen["prepare_inbound"] = kwargs.get("inbound_conversation_id")
            return SimpleNamespace(
                vc_command=False, is_passthrough=False, is_streaming=False, paging_enabled=False,
                tool_output_find_quote=False, restore_tool_injected=False, enriched_body=body,
                api_format="openai_responses", turn=1, request_turn=1, turn_id="t", overhead_ms=0,
                conversation_id=OWNER, speaker_context=None, upstream_limit=200_000,
                speaker_roster_snapshot=None,
            )

        async def handler(*args, **kwargs):
            seen["audience_route"] = kwargs["request_context"].audience_route
            return JSONResponse({})

        with patch("virtual_context.proxy.server.VirtualContextEngine", side_effect=RuntimeError("no storage")), \
                patch("virtual_context.proxy.server.prepare_payload", side_effect=prepare), \
                patch("virtual_context.proxy.server._handle_non_streaming", side_effect=handler):
            app = create_app("http://upstream.invalid")
            app.state.state_resolver = resolver
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                await client.post("/v1/responses", json=BODY)

    asyncio.run(run())
    return seen


@pytest.mark.regression("PROXY-034")
def test_out_of_band_route_reaches_preparation_as_the_inbound_route():
    seen = _run(resolver_sets_route=True)
    assert seen["resolver_inbound"] is None
    # The raw route the client named, not the resolved owner: an alias route
    # must stay distinguishable from its owner for audience validation.
    assert seen["prepare_inbound"] == ROUTE
    assert seen["audience_route"] == ROUTE


@pytest.mark.regression("PROXY-034")
def test_without_a_named_route_nothing_is_invented():
    seen = _run(resolver_sets_route=False)
    assert seen["prepare_inbound"] is None
    assert seen["audience_route"] == OWNER
