"""admin consolidate-tags: dispatch, plan/apply/revert flow, threshold source, embed probe."""
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from virtual_context.cli import main as cli_main


def _stub_engine(monkeypatch, store):
    from virtual_context import engine as engine_module

    class StubEngine:
        def __init__(self, config=None, **kwargs):
            self.config = config
            self._store = store
            self._llm_provider = None

        def close(self):
            pass

    monkeypatch.setattr(cli_main, "load_config", lambda path: SimpleNamespace(
        storage=SimpleNamespace(backend="sqlite"), retriever=SimpleNamespace(), judgment=SimpleNamespace(noul_threshold=0.8)))
    monkeypatch.setattr(cli_main, "_apply_storage_overrides", lambda config, args: None)
    monkeypatch.setattr(engine_module, "VirtualContextEngine", StubEngine)


def _args(**kw):
    base = dict(conversation_id="conv-A", tenant_id="t1", config=None, apply=False, max_pairs=500, embed_url="",
                jev=False, threshold=None, plan="", revert="", out="", storage_backend=None, postgres_dsn=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_admin_dispatch_reaches_consolidate_tags(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "cmd_admin_consolidate_tags", lambda args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["virtual-context", "admin", "consolidate-tags", "conv-9", "--tenant-id", "t1", "--jev", "--max-pairs", "7"])
    cli_main.main()
    assert calls and calls[0].conversation_id == "conv-9" and calls[0].jev is True and calls[0].max_pairs == 7


def test_threshold_defaults_to_the_config_cut():
    rt = cli_main._consolidation_judgment_runtime(SimpleNamespace(jev=True, threshold=None),
                                                  SimpleNamespace(judgment=SimpleNamespace(noul_threshold=0.8)))
    assert rt.config.noul_threshold == 0.8
    rt2 = cli_main._consolidation_judgment_runtime(SimpleNamespace(jev=True, threshold=0.6), None)
    assert rt2.config.noul_threshold == 0.6
    assert cli_main._consolidation_judgment_runtime(SimpleNamespace(jev=False), None) is None


def test_plan_apply_revert_flow(tmp_sqlite_db, tmp_path, monkeypatch, capsys):
    from virtual_context.core.conversation_store import ConversationStoreView
    from virtual_context.storage.sqlite import SQLiteStore
    from virtual_context.types import SegmentMetadata, StoredSegment

    raw = SQLiteStore(db_path=tmp_sqlite_db)
    now = datetime.now(timezone.utc)
    for i, tag in enumerate(["dosing-advice", "dosing-advice", "dosing-accuracy"]):
        raw.store_segment(StoredSegment(ref=f"ref-{i}", conversation_id="conv-A", primary_tag=tag, tags=[tag], summary="s",
                                        summary_tokens=1, full_tokens=2, full_text="t", metadata=SegmentMetadata(),
                                        created_at=now, start_timestamp=now, end_timestamp=now))
    store = ConversationStoreView(raw, "conv-A", 0)
    _stub_engine(monkeypatch, store)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"conversation_id": "conv-A",
                                     "groups": [{"canonical": "dosing-advice", "aliases": ["dosing-accuracy"], "reason": "reviewed"}]}))

    def run(**kw):
        cli_main.cmd_admin_consolidate_tags(_args(**kw))
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    applied_file = tmp_path / "applied.json"
    applied = run(apply=True, plan=str(plan_file), out=str(applied_file))
    assert applied["mode"] == "apply_plan" and applied["aliases_written"] == 1 and applied["segment_tags_added"] == 1
    assert json.loads(applied_file.read_text())["applied"] == applied["applied"]
    assert store.get_tag_aliases(conversation_id="conv-A") == {"dosing-accuracy": "dosing-advice"}
    assert "dosing-advice" in store.get_segment("ref-2").tags

    reverted = run(revert=str(applied_file))
    assert reverted["mode"] == "revert" and reverted == {"status": "ok", "conversation_id": "conv-A", "mode": "revert",
                                                          "aliases_deleted": 1, "segment_tags_removed": 1}
    assert store.get_tag_aliases(conversation_id="conv-A") == {}
    assert store.get_segment("ref-2").tags == ["dosing-accuracy"]
    raw.close()


def test_plan_for_another_conversation_is_refused(tmp_path, monkeypatch, capsys):
    from unittest import mock
    _stub_engine(monkeypatch, mock.MagicMock())
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"conversation_id": "conv-B", "groups": []}))
    with pytest.raises(SystemExit):
        cli_main.cmd_admin_consolidate_tags(_args(apply=True, plan=str(plan_file)))
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "error" and "conv-B" in out["error"]


def test_embed_probe_failure_aborts(monkeypatch, capsys):
    from unittest import mock
    _stub_engine(monkeypatch, mock.MagicMock())
    monkeypatch.setattr(cli_main, "_sidecar_embed_fn", lambda url: (lambda texts: (_ for _ in ()).throw(ConnectionError("sidecar down"))))
    with pytest.raises(SystemExit):
        cli_main.cmd_admin_consolidate_tags(_args(embed_url="http://127.0.0.1:8091"))
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "error" and out["stage"] == "embed_probe"


def test_embed_url_is_normalized_to_the_embed_route(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"vectors": [[1.0]]}'

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert cli_main._sidecar_embed_fn("http://127.0.0.1:8091/")(["x"]) == [[1.0]]
    assert seen["url"] == "http://127.0.0.1:8091/embed"


def test_revert_record_for_another_conversation_is_refused(tmp_path, monkeypatch, capsys):
    from unittest import mock
    store = mock.MagicMock()
    _stub_engine(monkeypatch, store)
    applied_file = tmp_path / "applied.json"
    applied_file.write_text(json.dumps({"conversation_id": "conv-B", "applied": [
        {"canonical": "x", "aliases_written": ["y"], "segment_refs": ["r1"]}]}))
    with pytest.raises(SystemExit):
        cli_main.cmd_admin_consolidate_tags(_args(revert=str(applied_file)))
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "error" and "conv-B" in out["error"]
    store.delete_tag_alias.assert_not_called()
    store.remove_tag_from_segments.assert_not_called()


def test_apply_writes_provenance_per_group_so_a_later_failure_stays_revertable(tmp_sqlite_db, tmp_path, monkeypatch, capsys):
    from virtual_context.core.conversation_store import ConversationStoreView
    from virtual_context.storage.sqlite import SQLiteStore
    from virtual_context.types import SegmentMetadata, StoredSegment

    raw = SQLiteStore(db_path=tmp_sqlite_db)
    now = datetime.now(timezone.utc)
    for i, tag in enumerate(["dosing-advice", "dosing-accuracy", "squat-form", "squat-technique"]):
        raw.store_segment(StoredSegment(ref=f"ref-{i}", conversation_id="conv-A", primary_tag=tag, tags=[tag], summary="s",
                                        summary_tokens=1, full_tokens=2, full_text="t", metadata=SegmentMetadata(),
                                        created_at=now, start_timestamp=now, end_timestamp=now))
    store = ConversationStoreView(raw, "conv-A", 0)
    _stub_engine(monkeypatch, store)
    real_add = raw.add_tag_to_segments_with_tags
    calls = []

    def flaky_add(canonical, aliases, *, conversation_id=""):
        calls.append(canonical)
        if len(calls) == 2:
            raise RuntimeError("connection reset")
        return real_add(canonical, aliases, conversation_id=conversation_id)

    monkeypatch.setattr(raw, "add_tag_to_segments_with_tags", flaky_add)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"conversation_id": "conv-A", "groups": [
        {"canonical": "dosing-advice", "aliases": ["dosing-accuracy"]},
        {"canonical": "squat-form", "aliases": ["squat-technique"]}]}))
    out_file = tmp_path / "applied.json"
    with pytest.raises(SystemExit):
        cli_main.cmd_admin_consolidate_tags(_args(apply=True, plan=str(plan_file), out=str(out_file)))
    err = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert err["status"] == "error" and err["stage"] == "apply" and err["applied_so_far"] == 2
    assert [e["canonical"] for e in err["applied"]] == ["dosing-advice", "squat-form"]
    partial = json.loads(out_file.read_text())
    assert partial["status"] == "partial" and partial["conversation_id"] == "conv-A"
    assert partial["applied"] == [
        {"canonical": "dosing-advice", "aliases_written": ["dosing-accuracy"], "aliases_rewritten": [], "segment_refs": ["ref-1"]},
        {"canonical": "squat-form", "aliases_written": ["squat-technique"], "aliases_rewritten": [], "segment_refs": []},
    ]  # the failed group's committed alias is in the record too
    raw.close()


def test_max_pairs_default_covers_a_large_vocabulary(monkeypatch):
    calls = []
    monkeypatch.setattr(cli_main, "cmd_admin_consolidate_tags", lambda args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["virtual-context", "admin", "consolidate-tags", "conv-9", "--tenant-id", "t1"])
    cli_main.main()
    assert calls[0].max_pairs == 5000
