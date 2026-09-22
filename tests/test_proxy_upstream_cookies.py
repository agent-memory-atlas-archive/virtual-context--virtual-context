"""The shared upstream client never carries one caller's cookies into another request."""
import asyncio

import httpx

from virtual_context.proxy.helpers import make_upstream_client


def test_upstream_cookies_are_not_persisted_between_requests():
    seen = []

    def handler(request):
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, headers={"set-cookie": "_cfuvid=abc; Path=/; HttpOnly"}, json={})

    async def run():
        async with make_upstream_client(transport=httpx.MockTransport(handler)) as client:
            await client.get("https://upstream.example/a")
            await client.get("https://upstream.example/b")
            await client.get("https://upstream.example/c", headers={"cookie": "mine=1"})
        return seen
    assert asyncio.run(run()) == [None, None, "mine=1"]
