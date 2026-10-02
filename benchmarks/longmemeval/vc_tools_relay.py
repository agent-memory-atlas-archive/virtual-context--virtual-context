"""MCP stdio server that relays VC paging tool calls to the benchmark harness.

``claude -p`` starts this as its ``vc`` MCP server. It lists the tool
definitions written by the harness and forwards every call to the harness's
localhost endpoint, where the call runs on the question's live engine. It
imports nothing from the package, so it runs from any working directory.

Usage: vc_tools_relay.py <call_url> <tool_definitions.json>
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


def main() -> None:
    call_url, defs_path = sys.argv[1], sys.argv[2]
    with open(defs_path) as f:
        definitions = json.load(f)
    server = Server("vc")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=d["name"], description=d.get("description", ""),
                       inputSchema=d.get("input_schema") or {"type": "object"})
            for d in definitions
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        body = json.dumps({"name": name, "input": arguments or {}}).encode()

        def post() -> str:
            request = urllib.request.Request(
                call_url, data=body, headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                return response.read().decode()

        return [types.TextContent(type="text", text=await asyncio.to_thread(post))]

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
