"""LongMemEval models can run through the Claude Code CLI on a subscription."""
from __future__ import annotations

import json
import subprocess
import urllib.request

import pytest

from benchmarks.longmemeval import claude_cli
from benchmarks.longmemeval.claude_cli_reader import _ToolHost


def _completed(payload: dict, rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=json.dumps(payload), stderr="")


def test_the_cli_runs_without_the_api_key_or_local_setup(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["env"], seen["input"] = cmd, kwargs["env"], kwargs["input"]
        return _completed({"result": "Paris", "is_error": False, "num_turns": 1,
                           "usage": {"input_tokens": 10, "cache_read_input_tokens": 5,
                                     "cache_creation_input_tokens": 2, "output_tokens": 3}})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)

    out = claude_cli.run_claude_cli("Capital of France?", model="claude-sonnet-5", system="Be brief.")

    assert out["hypothesis"] == "Paris"
    assert (out["input_tokens"], out["output_tokens"], out["cache_read_tokens"]) == (17, 3, 5)
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert seen["input"] == "Capital of France?"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--tools") + 1] == ""
    assert seen["cmd"][seen["cmd"].index("--setting-sources") + 1] == ""


def test_an_error_result_raises(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run",
                        lambda *a, **k: _completed({"result": "usage limit", "is_error": True}))
    with pytest.raises(claude_cli.ClaudeCliError, match="usage limit"):
        claude_cli.run_claude_cli("hi", model="claude-sonnet-5")


def test_relayed_calls_run_on_the_engine_and_expand_returns_the_context(monkeypatch):
    calls = []

    def fake_execute(engine, name, tool_input, **kwargs):
        calls.append((name, tool_input, kwargs["intent_context"]))
        return json.dumps({"tag": tool_input.get("tag"), "tokens_added": 40})

    monkeypatch.setattr("benchmarks.longmemeval.claude_cli_reader.execute_vc_tool", fake_execute)
    engine = type("Engine", (), {"reassemble_context": lambda self, **_: "<virtual-context>full topic</virtual-context>"})()

    with _ToolHost(engine, intent_context="the question", speaker_context=None) as host:
        request = urllib.request.Request(
            host.url, data=json.dumps({"name": "vc_expand_topic", "input": {"tag": "bike"}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        text = urllib.request.urlopen(request, timeout=10).read().decode()

    assert calls == [("vc_expand_topic", {"tag": "bike"}, "the question")]
    assert text.endswith("<virtual-context>full topic</virtual-context>")
    assert host.calls[0]["tool"] == "vc_expand_topic"


def test_an_engine_config_supplies_memory_settings_and_the_run_keeps_its_own(tmp_path):
    import yaml

    from benchmarks.longmemeval.vc_runner import _build_vc_config, _with_engine_config

    path = tmp_path / "prod.yaml"
    path.write_text(yaml.safe_dump({
        "context_window": 200000,
        "tag_generator": {"type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
        "storage": {"backend": "postgres"},
        "retrieval": {"vector_search_enabled": True},
        "telemetry": {"models_file": "/app/models.yaml"},
        "judgment": {"seams": {"fact_curation": "jev"}},
    }))
    harness = _build_vc_config(context_window=65536, storage_dir=str(tmp_path), session_id="bench-q",
                               tagger_provider="openrouter", tagger_model="other")
    cfg = _with_engine_config(str(path), harness)

    assert cfg["tag_generator"]["model"] == "google/gemini-2.5-flash-lite"
    assert cfg["judgment"]["seams"]["fact_curation"] == "jev"
    assert (cfg["context_window"], cfg["storage"]["backend"], cfg["session_id"]) == (65536, "sqlite", "bench-q")
    assert "telemetry" not in cfg and "vector_search_enabled" not in cfg["retrieval"]
