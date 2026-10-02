"""Run a Claude model through the Claude Code CLI (``claude -p``).

The CLI authenticates with the signed-in Claude subscription. Each call:

- drops ``ANTHROPIC_API_KEY`` from the child environment, which the CLI would
  otherwise bill instead of the subscription;
- runs from an empty working directory with settings, skills, every MCP server
  but ``vc`` and the built-in tools off, so nothing of the local Claude Code
  setup is added to the prompt;
- takes the prompt on stdin and returns the parsed ``--output-format json``
  result.

The reader's VC paging tools, when given, reach the model through an MCP
server named ``vc``; every other tool stays unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CLAUDE_BIN = os.environ.get("CLAUDE_CLI_BIN", "claude")
# The API baseline and judge send no system prompt. The CLI substitutes its own
# when none is given, so those calls pass this one-line prompt instead.
NEUTRAL_SYSTEM = "You are a helpful assistant."


class ClaudeCliError(RuntimeError):
    """The CLI failed or returned an error result."""


def run_claude_cli(
    prompt: str,
    *,
    model: str,
    system: str = "",
    mcp_config: dict | None = None,
    timeout: int = 900,
) -> dict:
    """Send *prompt* to *model* and return the answer with its token usage.

    Returns ``hypothesis``, ``input_tokens`` (fresh plus cache-read and
    cache-write input), ``output_tokens``, ``cache_read_tokens``,
    ``cache_write_tokens``, ``num_turns`` and ``raw`` (the CLI's JSON).
    """
    if shutil.which(CLAUDE_BIN) is None:
        raise ClaudeCliError(f"{CLAUDE_BIN} not found on PATH")
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    workdir = tempfile.mkdtemp(prefix="lme-claude-")
    try:
        cmd = [
            CLAUDE_BIN, "-p",
            "--model", model,
            "--output-format", "json",
            "--no-session-persistence",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--tools", "",
        ]
        if system:
            cmd += ["--system-prompt", system]
        if mcp_config:
            config_path = Path(workdir) / "mcp.json"
            config_path.write_text(json.dumps(mcp_config))
            config_path.chmod(0o600)
            cmd += [
                "--mcp-config", str(config_path),
                "--allowedTools", "mcp__vc__*",
            ]
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=timeout, cwd=workdir, env=env,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    stdout = result.stdout.strip()
    start = stdout.find("{")
    if start < 0:
        raise ClaudeCliError(
            f"claude CLI returned no JSON (rc={result.returncode}): "
            f"{stdout[:300]} {result.stderr[:300]}"
        )
    data = json.loads(stdout[start:])
    if data.get("is_error") or result.returncode != 0:
        raise ClaudeCliError(f"claude CLI error (rc={result.returncode}): {str(data.get('result'))[:300]}")

    usage = data.get("usage") or {}
    fresh = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    return {
        "hypothesis": data.get("result") or "",
        "input_tokens": fresh + cache_read + cache_write,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "num_turns": int(data.get("num_turns") or 0),
        "raw": data,
    }
