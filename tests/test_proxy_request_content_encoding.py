"""Compressed request bodies are decoded before the proxy parses them."""
import asyncio
import gzip
import json
import zlib
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from starlette.responses import JSONResponse

from virtual_context.proxy.helpers import DecodedBodyTooLarge, UnsupportedContentEncoding, decode_request_body
from virtual_context.proxy.server import create_app

try:
    import zstandard
except ImportError:  # pragma: no cover - optional codec
    zstandard = None

PAYLOAD = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "h\u00e9llo"}]}).encode()


def test_identity_and_missing_encoding_pass_bytes_through():
    assert decode_request_body(PAYLOAD, None) == PAYLOAD
    assert decode_request_body(PAYLOAD, "identity") == PAYLOAD
    assert decode_request_body(PAYLOAD, "") == PAYLOAD


def test_gzip_and_deflate_are_decoded():
    assert decode_request_body(gzip.compress(PAYLOAD), "gzip") == PAYLOAD
    assert decode_request_body(gzip.compress(PAYLOAD), "x-gzip") == PAYLOAD
    assert decode_request_body(zlib.compress(PAYLOAD), "deflate") == PAYLOAD
    raw = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    assert decode_request_body(raw.compress(PAYLOAD) + raw.flush(), "deflate") == PAYLOAD


@pytest.mark.skipif(zstandard is None, reason="zstandard not installed")
def test_zstd_is_decoded_including_frames_without_content_size():
    assert decode_request_body(zstandard.ZstdCompressor().compress(PAYLOAD), "zstd") == PAYLOAD
    streaming = zstandard.ZstdCompressor().compressobj()
    frame = streaming.compress(PAYLOAD) + streaming.flush()
    assert decode_request_body(frame, "zstd") == PAYLOAD


def test_unknown_encoding_and_corrupt_bodies_raise():
    with pytest.raises(UnsupportedContentEncoding) as exc:
        decode_request_body(PAYLOAD, "lzma")
    assert exc.value.encoding == "lzma"
    with pytest.raises(Exception):
        decode_request_body(b"not gzip", "gzip")


def _run_app_post(headers, body):
    async def run():
        seen = {}

        async def prepare(body, state, fmt, request_metrics, **kwargs):
            seen["body"] = body
            return SimpleNamespace(vc_command=False, is_passthrough=False, is_streaming=False, paging_enabled=False,
                                   tool_output_find_quote=False, restore_tool_injected=False, enriched_body=body,
                                   api_format="openai", turn=1, request_turn=1, turn_id="t", overhead_ms=0,
                                   conversation_id="c", speaker_context=None, upstream_limit=200_000,
                                   speaker_roster_snapshot=None)

        async def handler(*args, **kwargs):
            return JSONResponse({"content": args[3]["messages"][-1]["content"]})

        with patch("virtual_context.proxy.server.VirtualContextEngine", side_effect=RuntimeError("no storage")), \
             patch("virtual_context.proxy.server.prepare_payload", side_effect=prepare), \
             patch("virtual_context.proxy.server._handle_non_streaming", side_effect=handler):
            app = create_app("http://upstream.invalid", shared_metrics=SimpleNamespace(owner="default"))
            state = SimpleNamespace(metrics=SimpleNamespace(owner="c"), engine=SimpleNamespace(config=SimpleNamespace(conversation_id="c")))
            app.state.state_resolver = lambda request, body, cid: (state, False)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
                resp = await client.post("/v1/chat/completions", headers=headers, content=body)
        return resp, seen
    return asyncio.run(run())


def test_gzip_request_body_reaches_the_pipeline_decoded():
    resp, seen = _run_app_post({"content-type": "application/json", "content-encoding": "gzip"}, gzip.compress(PAYLOAD))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"content": "h\u00e9llo"}
    assert seen["body"]["messages"][0]["content"] == "h\u00e9llo"


@pytest.mark.skipif(zstandard is None, reason="zstandard not installed")
def test_zstd_request_body_reaches_the_pipeline_decoded():
    resp, _ = _run_app_post({"content-type": "application/json", "content-encoding": "zstd"},
                            zstandard.ZstdCompressor().compress(PAYLOAD))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"content": "h\u00e9llo"}


def test_unsupported_and_corrupt_encodings_are_client_errors_not_500s():
    resp, _ = _run_app_post({"content-type": "application/json", "content-encoding": "lzma"}, PAYLOAD)
    assert resp.status_code == 415 and resp.json()["error"]["encoding"] == "lzma"
    resp, _ = _run_app_post({"content-type": "application/json", "content-encoding": "gzip"}, b"\x28\xb5\x2f\xfd garbage")
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_content_encoding"


def test_decoded_size_is_capped_for_every_codec():
    big = b"x" * 4096
    with pytest.raises(DecodedBodyTooLarge):
        decode_request_body(gzip.compress(big), "gzip", limit=1024)
    with pytest.raises(DecodedBodyTooLarge):
        decode_request_body(zlib.compress(big), "deflate", limit=1024)
    assert decode_request_body(gzip.compress(big), "gzip", limit=4096) == big
    if zstandard is not None:
        with pytest.raises(DecodedBodyTooLarge):
            decode_request_body(zstandard.ZstdCompressor().compress(big), "zstd", limit=1024)
        assert decode_request_body(zstandard.ZstdCompressor().compress(big), "zstd", limit=4096) == big


def test_oversized_decoded_body_is_a_413():
    with patch("virtual_context.proxy.server.decode_request_body", side_effect=DecodedBodyTooLarge(1)):
        resp, _ = _run_app_post({"content-type": "application/json", "content-encoding": "gzip"}, gzip.compress(PAYLOAD))
    assert resp.status_code == 413 and resp.json()["error"]["type"] == "request_too_large"


def test_concatenated_gzip_members_decode_whole_and_still_respect_the_cap():
    body = gzip.compress(b'{"messages":') + gzip.compress(b'[]}')
    assert decode_request_body(body, "gzip") == b'{"messages":[]}'
    with pytest.raises(DecodedBodyTooLarge):
        decode_request_body(gzip.compress(b"a" * 700) + gzip.compress(b"b" * 700), "gzip", limit=1024)


def test_client_accept_encoding_never_travels_upstream():
    from virtual_context.proxy.helpers import _forward_headers
    fwd = _forward_headers({"accept-encoding": "br, gzip, deflate, zstd", "accept": "text/event-stream", "authorization": "Bearer t"})
    assert "accept-encoding" not in fwd and fwd["accept"] == "text/event-stream" and fwd["authorization"] == "Bearer t"
