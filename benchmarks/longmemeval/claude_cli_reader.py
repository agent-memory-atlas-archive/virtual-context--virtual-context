"""VC reader through ``claude -p``, with the paging tools run on the live engine.

The tool loop runs inside Claude Code. Each tool call reaches this process
through the ``vc`` MCP relay and runs on the question's engine with the same
arguments the API path's tool loop passes: the question as intent context,
the request's speaker context, and the presented segment and fact sets.

Three things differ from the API path, and the result records them:

- tool use cannot be required on the first call, because the CLI exposes no
  tool-choice setting;
- the CLI's system prompt is fixed for the session, so after
  ``vc_expand_topic`` the re-assembled context is returned in the tool result
  instead of replacing the system prompt;
- ``vc_find_session`` is offered from the start, because the relay's tool list
  is fixed for the session.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from virtual_context.core.tool_loop import (
    _vc_find_session_def,
    execute_vc_tool,
    vc_tool_definitions_for_runtime,
)

from .claude_cli import run_claude_cli

RELAY = Path(__file__).with_name("vc_tools_relay.py")
CLI_READER_NOTES = (
    "tool use not required on the first call",
    "vc_expand_topic returns the re-assembled context in its tool result",
    "vc_find_session offered from the start",
)


class _ToolHost:
    """Runs relayed tool calls on one engine, one at a time, and records them."""

    def __init__(self, engine, *, intent_context: str, speaker_context) -> None:
        self.engine = engine
        self.intent_context = intent_context
        self.speaker_context = speaker_context
        self.presented_refs: set[str] = set()
        self.presented_facts: set[str] = set()
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length) or b"{}")
                text = host.call(request.get("name", ""), request.get("input") or {})
                payload = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/call"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()

    def call(self, name: str, tool_input: dict) -> str:
        with self._lock:
            t0 = time.monotonic()
            result = execute_vc_tool(
                self.engine, name, tool_input,
                intent_context=self.intent_context,
                presented_segment_refs=self.presented_refs,
                presented_fact_ids=self.presented_facts,
                speaker_context=self.speaker_context,
            )
            if name == "vc_expand_topic":
                context = self.engine.reassemble_context(speaker_context=self.speaker_context)
                if context:
                    result = f"{result}\n\n{context}"
            self.calls.append({
                "tool": name,
                "input": tool_input,
                "result": result,
                "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            })
            return result


def run_cli_reader(
    engine,
    *,
    model: str,
    system: str,
    user_prompt: str,
    speaker_context,
) -> dict:
    """Answer *user_prompt* with *model* and the VC tools; returns the reader result."""
    definitions = vc_tool_definitions_for_runtime(None) + [_vc_find_session_def()]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(definitions, f)
        defs_path = f.name
    try:
        with _ToolHost(engine, intent_context=user_prompt, speaker_context=speaker_context) as host:
            mcp_config = {"mcpServers": {"vc": {
                "type": "stdio",
                "command": os.environ.get("LME_RELAY_PYTHON") or sys.executable,
                "args": [str(RELAY), host.url, defs_path],
            }}}
            result = run_claude_cli(user_prompt, model=model, system=system, mcp_config=mcp_config)
            result["tool_calls"] = list(host.calls)
    finally:
        os.unlink(defs_path)
    result["reader_notes"] = list(CLI_READER_NOTES)
    return result
