"""End-to-end smoke test of a recommended preset through the proxy.

A local stub stands in for the model provider: it records every request the
proxy forwards and answers with a canned reply, so no provider is billed.
The tagging and summarization calls are real and go to whichever model the
preset in the working directory names. Run it from scripts/smoke_recommended.sh, which
builds a clean environment first.

Checks: the proxy starts; stored turns get model tags (not the fallback);
compaction writes segments and topic summaries; and a later question
reaches the provider with the paging tools and retrieved context about an
earlier topic.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

WORKDIR = Path(sys.argv[1]).resolve()
CLI = sys.argv[2]
MODEL = "claude-sonnet-4-5"  # an autonomous model, so the paging tools are injected

TOPICS = {
    "sourdough": [
        "I'm starting a sourdough starter with whole wheat flour, fed twice a day at 1:1:1.",
        "My sourdough loaf came out dense. I used 70% hydration and a 4 hour bulk ferment.",
        "For the next sourdough bake I settled on 78% hydration with 20% whole wheat flour.",
        "The sourdough crumb is much more open now after a cold retard overnight in the fridge.",
        "Scoring the sourdough at a 30 degree angle finally gave me a proper ear.",
    ],
    "postgres": [
        "We are migrating our Postgres database from version 13 to 16 next week.",
        "The Postgres upgrade plan uses pg_upgrade with --link to keep downtime short.",
        "One Postgres extension, pg_partman, needs a newer build before the upgrade.",
        "After the Postgres 16 upgrade, autovacuum settings need retuning for the orders table.",
        "We will run ANALYZE on every Postgres table right after pg_upgrade finishes.",
    ],
    "japan": [
        "I'm planning a two week trip to Japan in April for the cherry blossoms.",
        "For the Japan trip I booked five nights in Kyoto near Gion.",
        "In Japan I want a day trip from Kyoto to Nara to see the deer park.",
        "The Japan itinerary ends with four nights in Tokyo, staying in Shibuya.",
        "I bought a 14 day Japan Rail Pass for the trip between Kyoto and Tokyo.",
    ],
}
QUESTION = "What hydration did I settle on for my sourdough bake?"

captured: list[dict] = []
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


class _Provider(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        captured.append(body)
        reply = {
            "id": "msg_smoke", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "text", "text": "Noted, that is recorded."}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port: int, messages: list[dict]) -> dict:
    # Anthropic Messages format, as Claude Code sends it.
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps({"model": MODEL, "max_tokens": 256, "system": "You are a helpful assistant.",
                         "messages": messages, "stream": False}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "smoke", "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _store() -> sqlite3.Connection:
    return sqlite3.connect(WORKDIR / ".virtualcontext" / "store.db")


def main() -> int:
    provider_port, proxy_port = _free_port(), _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", provider_port), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    log = open(WORKDIR / "proxy.log", "w")
    proxy = subprocess.Popen(
        [CLI, "-c", str(WORKDIR / "virtual-context.yaml"), "proxy",
         "--upstream", f"http://127.0.0.1:{provider_port}", "--port", str(proxy_port)],
        cwd=WORKDIR, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        started = False
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{proxy_port}/dashboard/settings", timeout=2)
                started = True
                break
            except Exception:
                time.sleep(1)
        check("proxy starts", started, f"port {proxy_port}")
        if not started:
            return 1

        history: list[dict] = []
        turns = [t for group in zip(*TOPICS.values()) for t in group]
        for text in turns:
            history.append({"role": "user", "content": text})
            reply = _post(proxy_port, history)
            history.append({"role": "assistant", "content": reply["content"][0]["text"]})

        tagged = []
        for _ in range(90):
            with _store() as db:
                rows = db.execute("SELECT tags_json FROM canonical_turns WHERE tagged_at IS NOT NULL").fetchall()
            tagged = [json.loads(r[0] or "[]") for r in rows]
            if len(tagged) >= len(turns):
                break
            time.sleep(2)
        real = [t for t in tagged if t and not all(tag.startswith("_") for tag in t)]
        check("turns tagged by the model", len(real) >= len(turns) // 2,
              f"{len(real)} of {len(tagged)} tagged rows carry model tags")

        history.append({"role": "user", "content": "VCCOMPACT"})
        _post(proxy_port, history)
        history.pop()
        segments = summaries = 0
        for _ in range(150):
            with _store() as db:
                segments = db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
                summaries = db.execute("SELECT COUNT(*) FROM tag_summaries").fetchone()[0]
            if segments and summaries:
                break
            time.sleep(2)
        check("compaction writes segments", segments > 0, f"{segments} segments")
        check("compaction writes topic summaries", summaries > 0, f"{summaries} topic summaries")

        before = len(captured)
        history.append({"role": "user", "content": QUESTION})
        _post(proxy_port, history)
        sent = captured[before:]
        last = sent[0] if sent else {}
        tools = [t.get("function", {}).get("name") or t.get("name") for t in last.get("tools", [])]
        check("paging tools reach the provider", any(str(n).startswith("vc_") for n in tools),
              ", ".join(str(n) for n in tools[:6]))
        text = json.dumps([last.get("system", ""), last.get("messages", [])])
        injected = "virtual-context" in text.lower() or "<virtual-context" in text
        check("retrieved context reaches the provider", injected and "hydration" in text.lower(),
              "context block present" if injected else "no context block")
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proxy.kill()
        server.shutdown()
        log.close()

    failed = [name for name, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
