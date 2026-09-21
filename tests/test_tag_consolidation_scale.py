"""Tag consolidation on large vocabularies: blocked candidates, embedding neighbours, scoped stores, CLI runner."""
import json
import time
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest

from virtual_context.core import judgment
from virtual_context.core.judgment import build_runtime, candidate_tag_pairs
from virtual_context.core.tag_consolidator import consolidate_tags
from virtual_context.types import JudgmentConfig, TagStats


@pytest.fixture(autouse=True)
def _reset():
    judgment.reset()
    yield
    judgment.reset()


def _jev_runtime(same_pairs: set[frozenset], threshold=0.5):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        answers = {}
        for key in body["questions"]:
            pair = frozenset(body["state"]["pairs"][key.split("__", 1)[1]])
            answers[key] = {"type": "noul", "noul": 0.95 if pair in same_pairs else 0.05}
        return httpx.Response(200, json={"model": "jev-t", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    rt = build_runtime(JudgmentConfig(mode="legacy", seams={"tag_consolidation": "jev"}, noul_threshold=threshold),
                       environ={"TYPESAFE_API_KEY": "k"}, http_client=http)
    return rt, seen


def test_blocked_candidates_scale_to_thousands_of_tags():
    tags = [f"topic{i}-{suffix}" for i in range(600) for suffix in ("advice", "guidelines", "notes", "history", "plan")]
    tags += ["dosing-advice", "dosing-accuracy", "deploy-notes", "deployment", "unrelated-zebra"]
    started = time.monotonic()
    pairs = candidate_tag_pairs(tags, limit=100_000)
    elapsed = time.monotonic() - started
    assert elapsed < 8.0, f"pair generation took {elapsed:.1f}s for {len(tags)} tags"
    assert ("dosing-accuracy", "dosing-advice") in pairs
    assert ("deploy-notes", "deployment") in pairs
    assert not any("unrelated-zebra" in p for p in pairs)
    assert ("topic1-advice", "topic1-guidelines") in pairs  # same token block


def test_embedding_neighbours_add_synonym_pairs_without_lexical_overlap():
    concept = {"kpv-dosing": 0, "dosage": 0, "dosage-recommendations": 0, "car-repair": 1, "brake-replacement": 1, "gardening": 2}

    def embed(texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0, 0.0]
            v[concept[t]] = 1.0
            v[3] = 0.05 * (hash(t) % 7)  # small per-tag noise
            out.append(v)
        return out

    pairs = candidate_tag_pairs(list(concept), limit=50, embed_fn=embed)
    assert ("dosage", "kpv-dosing") in pairs
    assert ("brake-replacement", "car-repair") in pairs
    assert not any("gardening" in p for p in pairs)


def test_consolidate_tags_scopes_reads_to_the_store_conversation_and_caps_pairs():
    rt, seen = _jev_runtime({frozenset(("dosing-advice", "dosing-accuracy"))})
    store = mock.MagicMock()
    store.conversation_id = "conv-A"
    store.get_all_tags.return_value = [TagStats(tag="dosing-advice", usage_count=5), TagStats(tag="dosing-accuracy", usage_count=2),
                                       TagStats(tag="dosing-notes", usage_count=1), TagStats(tag="dosing-history", usage_count=1)]
    store.get_tag_aliases.return_value = {}
    store.add_tag_to_segments_with_tags.return_value = ["seg-9"]
    result = consolidate_tags(store, llm=None, dry_run=False, judgment_runtime=rt, max_pairs=2)
    store.get_all_tags.assert_called_once_with(conversation_id="conv-A")
    assert len(seen[0]["state"]["pairs"]) <= 2
    assert [(g.canonical, g.aliases) for g in result.groups] == [("dosing-advice", ["dosing-accuracy"])]
    store.create_tag_alias_if_absent.assert_called_once_with("dosing-accuracy", "dosing-advice", conversation_id="conv-A")
    store.add_tag_to_segments_with_tags.assert_called_once_with("dosing-advice", ["dosing-accuracy"], conversation_id="conv-A")
    assert result.segment_tags_added == 1


def test_admin_consolidate_tags_dry_run_then_apply(tmp_sqlite_db, monkeypatch, capsys):
    from datetime import datetime, timezone

    from virtual_context.cli import main as cli_main
    from virtual_context.storage.sqlite import SQLiteStore
    from virtual_context.types import SegmentMetadata, StoredSegment
    from virtual_context import engine as engine_module

    store = SQLiteStore(db_path=tmp_sqlite_db)
    now = datetime.now(timezone.utc)
    # dosing-advice carries two segments so it outranks dosing-accuracy as canonical
    for i, tag in enumerate(["dosing-advice", "dosing-advice", "dosing-accuracy", "car-repair"]):
        store.store_segment(StoredSegment(ref=f"ref-{i}", primary_tag=tag, tags=[tag], summary="s", summary_tokens=1,
                                          full_tokens=2, full_text="t", metadata=SegmentMetadata(), created_at=now,
                                          start_timestamp=now, end_timestamp=now))

    class StubEngine:
        def __init__(self, config=None, **kwargs):
            self.config = config
            self._store = store
            self._llm_provider = None

        def close(self):
            pass

    rt, _ = _jev_runtime({frozenset(("dosing-advice", "dosing-accuracy"))})
    monkeypatch.setattr(cli_main, "load_config", lambda path: SimpleNamespace(storage=SimpleNamespace(backend="sqlite"), retriever=SimpleNamespace()))
    monkeypatch.setattr(cli_main, "_apply_storage_overrides", lambda config, args: None)
    monkeypatch.setattr(engine_module, "VirtualContextEngine", StubEngine)
    monkeypatch.setattr(cli_main, "_consolidation_judgment_runtime", lambda *a: rt)

    def run(apply):
        args = SimpleNamespace(conversation_id="conv-1", tenant_id="t1", config=None, apply=apply, max_pairs=500,
                               embed_url="", jev=True, storage_backend=None, postgres_dsn=None)
        cli_main.cmd_admin_consolidate_tags(args)
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    dry = run(apply=False)
    assert dry["dry_run"] is True and dry["groups"] == [{"canonical": "dosing-advice", "aliases": ["dosing-accuracy"], "reason": dry["groups"][0]["reason"]}]
    assert dry["aliases_written"] == 0 and store.get_tag_aliases() == {}
    applied = run(apply=True)
    assert applied["dry_run"] is False and applied["aliases_written"] == 1
    assert store.get_tag_aliases() == {"dosing-accuracy": "dosing-advice"}
    store.close()
