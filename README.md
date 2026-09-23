[![PyPI](https://img.shields.io/pypi/v/virtual-context.svg)](https://pypi.org/project/virtual-context/)
[![Python](https://img.shields.io/pypi/pyversions/virtual-context.svg)](https://pypi.org/project/virtual-context/)
[![Downloads](https://img.shields.io/pypi/dm/virtual-context.svg)](https://pypistats.org/packages/virtual-context)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://github.com/virtual-context/virtual-context/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Chat%20with%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/kGJva2D8Ej)
[![Twitter](https://img.shields.io/badge/Twitter-Follow-1DA1F2?logo=x&logoColor=white)](https://x.com/virtualctx)

<p align="center">
  <a href="assets/dashboard.png">
    <img src="assets/dashboard.png" alt="virtual-context dashboard" width="800">
  </a>
</p>
<p align="center"><sub>virtual-context cloud: a 3 million token virtual window served at 80K real tokens</sub></p>

# virtual-context

**Virtual memory for LLM context. Give your agent a context window of tens or hundreds of millions of tokens; the model only ever sees the part that matters.**

Your client sets `contextWindow: 20000000` (or 200,000,000). Your model's real window is 200K. virtual-context sits between them and makes it work, the same way an operating system lets a process address more memory than physically exists. The client keeps sending its whole conversation; virtual-context stores all of it, organizes it by topic, and forwards the model a compact window of what matters for the current turn, at a size you choose. Nothing is discarded, and anything older can be paged back in at full fidelity. The dashboard above shows a live 3 million token conversation served at 80K real tokens.

It runs as an HTTP proxy, so integration is a base-URL change; a Python SDK and an MCP server are there for direct use.

## Why

- **Memory that lasts.** What the user said at turn 12 is still recallable at turn 1,000, across sessions, clients and models.
- **Better answers from less context.** A focused window of relevant summaries and facts beats a raw window of everything, where the one fact you need is buried.
- **Lower cost.** Smaller payloads, arranged to keep providers' prompt caches warm.
- **Group chats that know who said what.** Every message keeps its sender, facts keep their author, and search can be scoped to one person.
- **One memory everywhere.** Build context in Claude Code, continue it from Telegram, query it from Cursor.

## How it works

```
client ──> virtual-context proxy ──> model provider
             │   before the call: retrieve topics and facts for this turn,
             │                    assemble a bounded window
             │   after the call:  store the turn, tag it, compact when needed
             v
           store (SQLite or PostgreSQL)
             canonical turns   every message verbatim, with sender and channel
             segments + facts  per-topic summaries and structured facts
             topic summaries   one digest per topic, the bird's-eye view
```

Recent turns stay verbatim. Older turns are grouped by topic, summarized and mined for facts. Each turn, the model gets recent turns plus the topics and facts relevant to the question, and a topic index telling it what else it can open with built-in tools (`vc_expand_topic`, `vc_find_quote`, `vc_recall_all`, and others).

## Quick start

```bash
pip install "virtual-context[proxy,embeddings]"    # Python 3.11+
```

virtual-context uses a language model of your choice to tag turns and write summaries. Pick where it runs:

```bash
# A local model server (Ollama, LM Studio, llama.cpp, vLLM): nothing leaves your machine
virtual-context init recommended-local

# Or a hosted model through OpenRouter
virtual-context init recommended
export OPENROUTER_API_KEY=...
```

Either writes `virtual-context.yaml`; set `tag_generator.model` and `summarization.model` to the model you want. Then start the proxy and point your client at `http://127.0.0.1:5757` instead of the provider:

```bash
virtual-context proxy --upstream https://api.anthropic.com
```

A dashboard runs at `http://127.0.0.1:5757/dashboard`.

The two presets are the hosted service's configuration on one machine and differ only in where the model runs: SQLite storage, paging tools on, fact curation on. Answer quality depends on the model you pick. `scripts/smoke_recommended.sh` checks a preset end to end from a clean install. The optional parts (PostgreSQL vector search, the judgment layer, group attribution) are each a section of [docs/configuration.md](docs/configuration.md); `virtual-context presets list` shows every preset, and [docs/install.md](docs/install.md) covers install options and running it as a service.

The proxy accepts Anthropic Messages, OpenAI Responses, OpenAI Chat Completions and Gemini requests. The paging tools are injected for Anthropic, OpenAI Responses and Gemini; Chat Completions clients get retrieved context but no tools.

**Hosted:** [virtual-context.com](https://virtual-context.com) runs the same engine as a service with the dashboard and cost reports; you change your base URL and nothing else.

## Integrations

**Claude Code**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:5757
```

Tool output (file reads, searches, command results) is collapsed into restorable stubs, and when Claude Code trims its own history, the missing turns are restored from storage.

**Any client with a base URL** (Cursor, Continue, your own app)

```python
client = anthropic.Anthropic(base_url="http://127.0.0.1:5757")
client = OpenAI(base_url="http://127.0.0.1:5757/v1")
```

**OpenClaw:** use the [OpenClaw plugin](https://github.com/virtual-context/openclaw-plugin). Settings for direct proxy use are in [docs/install.md](docs/install.md).

**Python SDK**

```python
from virtual_context import VirtualContextEngine

engine = VirtualContextEngine(config_path="./virtual-context.yaml")
assembled = engine.on_message_inbound(message=user_message, conversation_history=messages)  # before the LLM call
report = engine.on_turn_complete(messages)                                                  # after it
```

**MCP server:** eight tools (`recall_context`, `compact_context`, `expand_topic`, `recall_all`, `remember_when`, `find_quote`, `search_summaries`, `domain_status`) plus topic resources, for Claude Desktop, Cursor or any MCP client.

## Features

| Feature | What it does | Docs |
|---|---|---|
| Topic memory | Each turn is tagged by topic; older turns become per-topic segment summaries and one digest per topic. Tags converge on an existing vocabulary instead of piling up synonyms. | [engine](docs/engine.md) |
| Structured facts | Compaction extracts facts (subject, verb, object, status, time, source turns). Newer facts supersede older ones, and facts can be retrieved by topic and by meaning. | [fact lifecycle](docs/fact-lifecycle.md) |
| Retrieval | Topics are ranked by tag overlap, full-text match and embedding similarity, with full-text and semantic search over stored turns as a fallback. | [engine](docs/engine.md) |
| Demand paging | The model sees topic digests for cold topics and segment summaries for relevant ones, and opens full text through tools when it needs detail. | [engine](docs/engine.md) |
| Time-scoped recall | "What changed between June and July?" resolves by date math on stored timestamps, not by model guessing. | [engine](docs/engine.md) |
| Group memory | Sender and channel are stored on every message, facts keep their author, members get person cards, and search can be scoped to one speaker. Direct-message content never appears in a server's context. Off by default. | [attribution](docs/attribution.md) |
| Tool and media compression | Finished tool exchanges become stubs restorable with `vc_restore_tool`; screenshots are recompressed with originals kept. | [proxy](docs/proxy.md) |
| Prompt-cache aware | Injected context sits after a cache breakpoint, and payload rewrites wait until the provider's cache would have expired anyway. | [engine](docs/engine.md) |
| A ceiling you choose | Set `context_window` to any size; the virtual window the client sees is independent of it. | [configuration](docs/configuration.md) |
| Truncation recovery | When a client trims its own history, the missing turns are restored from storage. | [proxy](docs/proxy.md) |
| History import | ChatGPT, Claude and Grok exports arrive tagged and searchable (`virtual-context import`). | [commands](docs/commands.md) |
| Native vector search | On PostgreSQL with pgvector, quote search, segment candidates and fact search rank inside the database instead of in each worker. | [native vector search](docs/native-vector-search.md) |
| Judgment layer (optional) | A typed judgment model can make individual decisions: which topics and segments to retrieve, which facts to keep, which existing tags a turn belongs to, and more. Each decision runs in `legacy`, `shadow` (logged side by side) or `jev` mode and falls back to the built-in behavior on any failure. Needs a TypeSafe API key. | [configuration](docs/configuration.md#judgment) |

## Commands in any conversation

Type these as ordinary messages in a connected client; the proxy handles them without calling the model.

| Command | What it does |
|---|---|
| `VCATTACH <label\|id>` | Reattach this client to another stored conversation |
| `VCLABEL <name>` | Set the conversation label (no argument shows it) |
| `VCSTATUS` | Conversation id, label, turns, segments, active topics |
| `VCRECALL <query>` | Search stored context and bring matching topics into the next turn |
| `VCCOMPACT` | Compact now |
| `VCLIST` | List conversations with labels and turn counts |
| `VCFORGET <tag>` | Delete a topic's segments and summaries |
| `VCMERGE INTO <label\|id>` | Merge this conversation's stored data into another |

`VCATTACH` is a durable redirect: nothing is deleted and old references keep resolving, so two clients (or two agents) can share one memory. Details: [docs/commands.md](docs/commands.md).

## Running it in production

- **Storage:** SQLite by default; PostgreSQL (`pip install "virtual-context[postgres]"`) for multi-worker deployments. With pgvector installed, `virtual-context admin migrate-semantic-vectors` prepares in-database vector search.
- **Multi-worker safety:** compactions run under leased, fenced operations, lifecycle changes are epoch-guarded, and a backlog sweeper compacts conversations whose traffic never triggers it inline.
- **Operator tooling:** `virtual-context admin` has 27 guarded backfill, repair and migration commands with dry-run modes.
- **Security:** dashboard endpoints are open until you set `VC_DASHBOARD_TOKEN`; the default bind is loopback only.
- **Observability:** per-stage timing logs, request captures in the dashboard, and per-call cost telemetry.

Architecture, including the canonical-turn model and the REST prepare/ingest surface: [docs/architecture.md](docs/architecture.md).

## Benchmarks

On 100 questions from [LongMemEval-500](https://github.com/xiaowu0162/LongMemEval), with the same reader model (Claude Sonnet 4.5):

| | virtual-context | Full raw history |
|---|---|---|
| Correct | 95/100 | 33/100 |
| Tokens per question | 52,347 | 117,582 |
| Cost per question | $0.16 | $0.36 |

These are **historical results**: the original run's provenance is incomplete and they are not a measurement of the current pipeline. Configuration, results by category, the per-question table and how new runs are recorded: [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

| Page | Covers |
|---|---|
| [architecture.md](docs/architecture.md) | Memory model, request pipeline, multi-worker coordination, identity |
| [engine.md](docs/engine.md) | Compaction, tagging, retrieval, code mode, cache awareness |
| [fact-lifecycle.md](docs/fact-lifecycle.md) | How facts are extracted, superseded and retrieved |
| [attribution.md](docs/attribution.md) | Group conversations: who said what, person cards, speaker search |
| [proxy.md](docs/proxy.md) | Proxy internals, routing, dashboard, streaming, endpoints |
| [native-vector-search.md](docs/native-vector-search.md) | PostgreSQL + pgvector ranking and its migration |
| [configuration.md](docs/configuration.md) | Every config key with its default, including the judgment layer |
| [commands.md](docs/commands.md) | In-conversation commands, the CLI, import, admin tooling |
| [install.md](docs/install.md) | Install paths, daemons, multi-instance setups |
| [design.md](docs/design.md) | Why it is built this way, and how it compares with RAG and plain compaction |
| [benchmarks.md](docs/benchmarks.md) | Suites, results, how to run them |

## Development

```bash
git clone https://github.com/virtual-context/virtual-context.git
cd virtual-context
python -m venv .venv && source .venv/bin/activate
python -m pip install uv==0.9.30
uv sync --locked --extra all --extra dev
.venv/bin/python scripts/check_contracts.py
```

## License

AGPL-3.0, Copyright Y. Ahmed Kidwai. For commercial licensing: ahmed@kidw.ai
