# Benchmarks

Virtual-context is evaluated against established long-conversation memory benchmarks (LongMemEval, LoCoMo, MRCR, AMB) and an internal stress test suite. All benchmarks run against the full pipeline: tagging, compaction, retrieval, assembly, and LLM response.

## Benchmark Suites

### LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) tests long-term memory across many chat sessions: a large haystack of prior sessions is ingested, then questions probe facts stated anywhere in that history. Questions span six categories: knowledge-update, multi-session, temporal-reasoning, single-session-user, single-session-assistant, and single-session-preference.

**Historical results** (100 questions from LongMemEval-500, 5 batches of 20): virtual-context answered 95/100 correctly vs. 33/100 for the same reader model given the full raw history. Average tokens per question dropped 55% (52,347 vs. 117,582), and average cost per question dropped from $0.36 to $0.16. Results by category and the full per-question table follow.

These are historical claims, not a measurement of the current source-bound
pipeline. The [frozen claim record](../benchmarks/longmemeval/historical-claims.yaml)
identifies the exact published source bytes and marks missing original run
provenance explicitly. It does not retroactively attest the old cache or scores.
New runs require the content-addressed pipeline manifest described below.

Configuration: MiMo-V2-Flash for ingestion, Claude Sonnet 4.5 as the reader and Gemini 3 Pro Preview as the judge. The baseline is Claude Sonnet 4.5 given the full conversation history (about 118K tokens) with the same judge. Questions: 100 random questions from LongMemEval-500 in 5 batches of 20 (seeds 42/99/777/1234/2025).

#### Accuracy by question type (historical)

| Category | Count | VC | Baseline |
|----------|-------|----|----------|
| knowledge-update | 17 | 100.0% (17/17) | 29.4% (5/17) |
| multi-session | 26 | 88.5% (23/26) | 15.4% (4/26) |
| temporal-reasoning | 28 | 92.9% (26/28) | 32.1% (9/28) |
| single-session-user | 13 | 100.0% (13/13) | 46.2% (6/13) |
| single-session-assistant | 11 | 100.0% (11/11) | 72.7% (8/11) |
| single-session-preference | 5 | 100.0% (5/5) | 20.0% (1/5) |

<details>
<summary>Click to expand the full results table (100 questions)</summary>

| ID | Type | BL | BL Tokens | BL Cost | VC | VC Tokens | VC Cost |
|----|------|-----|-----------|---------|-----|-----------|---------|
| `07741c44` | knowledge-update | FAIL | 116,404 | $0.35 | pass | 49,721 | $0.15 |
| `0977f2af` | knowledge-update | FAIL | 117,359 | $0.35 | pass | 49,734 | $0.15 |
| `0ddfec37` | knowledge-update | FAIL | 115,848 | $0.35 | pass | 43,780 | $0.13 |
| `2133c1b5_abs` | knowledge-update | pass | 116,186 | $0.36 | pass | 56,533 | $0.17 |
| `2698e78f_abs` | knowledge-update | FAIL | 118,841 | $0.36 | pass | 36,039 | $0.11 |
| `3ba21379` | knowledge-update | FAIL | 116,604 | $0.35 | pass | 46,034 | $0.14 |
| `4b24c848` | knowledge-update | pass | 117,107 | $0.35 | pass | 32,494 | $0.10 |
| `4d6b87c8` | knowledge-update | FAIL | 115,104 | $0.35 | pass | 47,262 | $0.14 |
| `50635ada` | knowledge-update | FAIL | 118,682 | $0.36 | pass | 41,677 | $0.13 |
| `5a4f22c0` | knowledge-update | pass | 118,775 | $0.36 | pass | 35,437 | $0.11 |
| `6071bd76` | knowledge-update | FAIL | 117,904 | $0.36 | pass | 36,618 | $0.11 |
| `6aeb4375` | knowledge-update | pass | 115,001 | $0.35 | pass | 38,984 | $0.12 |
| `89941a94` | knowledge-update | FAIL | 117,038 | $0.35 | pass | 45,347 | $0.14 |
| `8fb83627` | knowledge-update | pass | 115,488 | $0.35 | pass | 35,041 | $0.11 |
| `a1eacc2a` | knowledge-update | FAIL | 117,513 | $0.35 | pass | 46,401 | $0.14 |
| `cf22b7bf` | knowledge-update | FAIL | 115,784 | $0.35 | pass | 49,002 | $0.15 |
| `ed4ddc30` | knowledge-update | FAIL | 118,045 | $0.36 | pass | 37,708 | $0.11 |
| `099778bb` | multi-session | FAIL | 118,622 | $0.36 | pass | 33,375 | $0.10 |
| `09ba9854` | multi-session | FAIL | 115,128 | $0.35 | FAIL | 36,120 | $0.11 |
| `0ea62687` | multi-session | FAIL | 116,840 | $0.36 | pass | 36,910 | $0.11 |
| `21d02d0d` | multi-session | FAIL | 119,667 | $0.36 | pass | 44,069 | $0.13 |
| `36b9f61e` | multi-session | FAIL | 116,713 | $0.35 | pass | 42,919 | $0.13 |
| `3fe836c9` | multi-session | FAIL | 117,954 | $0.35 | pass | 45,463 | $0.14 |
| `46a3abf7` | multi-session | FAIL | 117,783 | $0.35 | pass | 132,933 | $0.40 |
| `6456829e_abs` | multi-session | FAIL | 117,467 | $0.35 | pass | 42,898 | $0.13 |
| `681a1674` | multi-session | FAIL | 118,545 | $0.36 | pass | 62,141 | $0.19 |
| `720133ac` | multi-session | FAIL | 120,053 | $0.37 | pass | 50,205 | $0.15 |
| `7405e8b1` | multi-session | FAIL | 118,694 | $0.36 | pass | 50,989 | $0.16 |
| `88432d0a` | multi-session | FAIL | 118,401 | $0.36 | pass | 46,391 | $0.14 |
| `88432d0a_abs` | multi-session | pass | 119,275 | $0.36 | pass | 55,463 | $0.17 |
| `9d25d4e0` | multi-session | FAIL | 117,978 | $0.36 | pass | 83,295 | $0.25 |
| `a11281a2` | multi-session | FAIL | 119,807 | $0.36 | pass | 49,939 | $0.15 |
| `a346bb18` | multi-session | FAIL | 118,452 | $0.36 | pass | 44,404 | $0.14 |
| `a96c20ee` | multi-session | FAIL | 117,282 | $0.35 | pass | 42,068 | $0.13 |
| `bf659f65` | multi-session | FAIL | 114,781 | $0.35 | FAIL | 41,952 | $0.13 |
| `d682f1a2` | multi-session | FAIL | 117,856 | $0.35 | pass | 48,821 | $0.15 |
| `dd2973ad` | multi-session | pass | 117,351 | $0.36 | pass | 56,463 | $0.17 |
| `e56a43b9` | multi-session | pass | 119,177 | $0.36 | pass | 47,528 | $0.14 |
| `e6041065` | multi-session | FAIL | 117,316 | $0.35 | pass | 38,473 | $0.12 |
| `eeda8a6d` | multi-session | FAIL | 118,197 | $0.36 | pass | 45,726 | $0.14 |
| `ef66a6e5` | multi-session | FAIL | 116,328 | $0.35 | pass | 152,680 | $0.46 |
| `gpt4_372c3eed` | multi-session | pass | 117,552 | $0.36 | FAIL | 46,299 | $0.14 |
| `gpt4_d84a3211` | multi-session | FAIL | 116,459 | $0.35 | pass | 51,487 | $0.16 |
| `0db4c65d` | temporal-reasoning | FAIL | 115,780 | $0.35 | pass | 45,639 | $0.14 |
| `2ebe6c90` | temporal-reasoning | FAIL | 115,113 | $0.35 | pass | 39,883 | $0.12 |
| `6613b389` | temporal-reasoning | pass | 119,268 | $0.37 | pass | 41,228 | $0.13 |
| `a3045048` | temporal-reasoning | FAIL | 116,689 | $0.35 | pass | 47,120 | $0.14 |
| `b29f3365` | temporal-reasoning | FAIL | 118,078 | $0.36 | pass | 43,563 | $0.13 |
| `c8090214_abs` | temporal-reasoning | pass | 116,460 | $0.35 | pass | 79,046 | $0.24 |
| `cc6d1ec1` | temporal-reasoning | pass | 116,218 | $0.35 | pass | 47,747 | $0.15 |
| `eac54adc` | temporal-reasoning | FAIL | 119,492 | $0.36 | pass | 40,470 | $0.12 |
| `f0853d11` | temporal-reasoning | pass | 116,117 | $0.35 | pass | 46,903 | $0.14 |
| `gpt4_18c2b244` | temporal-reasoning | FAIL | 119,183 | $0.36 | pass | 53,922 | $0.17 |
| `gpt4_1a1dc16d` | temporal-reasoning | FAIL | 120,646 | $0.37 | pass | 52,119 | $0.16 |
| `gpt4_1e4a8aec` | temporal-reasoning | pass | 118,208 | $0.36 | pass | 48,286 | $0.15 |
| `gpt4_21adecb5` | temporal-reasoning | FAIL | 119,249 | $0.36 | pass | 125,864 | $0.38 |
| `gpt4_483dd43c` | temporal-reasoning | FAIL | 117,942 | $0.35 | pass | 43,327 | $0.13 |
| `gpt4_4929293b` | temporal-reasoning | FAIL | 118,774 | $0.37 | pass | 58,869 | $0.18 |
| `gpt4_4cd9eba1` | temporal-reasoning | pass | 119,611 | $0.36 | pass | 46,083 | $0.14 |
| `gpt4_5438fa52` | temporal-reasoning | FAIL | 114,753 | $0.35 | pass | 51,194 | $0.16 |
| `gpt4_65aabe59` | temporal-reasoning | FAIL | 115,392 | $0.35 | pass | 39,931 | $0.12 |
| `gpt4_70e84552` | temporal-reasoning | FAIL | 117,453 | $0.35 | pass | 42,109 | $0.13 |
| `gpt4_7ca326fa` | temporal-reasoning | FAIL | 116,432 | $0.35 | pass | 51,589 | $0.16 |
| `gpt4_7de946e7` | temporal-reasoning | pass | 117,096 | $0.35 | pass | 44,183 | $0.14 |
| `gpt4_8279ba02` | temporal-reasoning | FAIL | 115,780 | $0.35 | pass | 156,923 | $0.47 |
| `gpt4_88806d6e` | temporal-reasoning | FAIL | 119,052 | $0.36 | pass | 33,463 | $0.10 |
| `gpt4_98f46fc6` | temporal-reasoning | pass | 117,366 | $0.36 | pass | 58,524 | $0.18 |
| `gpt4_d6585ce9` | temporal-reasoning | FAIL | 115,862 | $0.35 | pass | 50,320 | $0.15 |
| `gpt4_d9af6064` | temporal-reasoning | pass | 116,298 | $0.35 | pass | 48,037 | $0.15 |
| `gpt4_f420262c` | temporal-reasoning | FAIL | 116,610 | $0.35 | FAIL | 134,691 | $0.41 |
| `gpt4_f420262d` | temporal-reasoning | FAIL | 118,803 | $0.36 | FAIL | 52,815 | $0.16 |
| `001be529` | ss-user | FAIL | 117,394 | $0.35 | pass | 40,375 | $0.12 |
| `15745da0` | ss-user | FAIL | 120,384 | $0.37 | pass | 53,318 | $0.16 |
| `19b5f2b3` | ss-user | pass | 115,688 | $0.35 | pass | 42,046 | $0.13 |
| `19b5f2b3_abs` | ss-user | pass | 116,214 | $0.35 | pass | 44,256 | $0.14 |
| `37d43f65` | ss-user | FAIL | 117,911 | $0.35 | pass | 72,955 | $0.22 |
| `4fd1909e` | ss-user | FAIL | 119,200 | $0.36 | pass | 50,759 | $0.15 |
| `577d4d32` | ss-user | pass | 116,583 | $0.35 | pass | 48,225 | $0.15 |
| `60d45044` | ss-user | FAIL | 119,224 | $0.36 | pass | 47,125 | $0.14 |
| `853b0a1d` | ss-user | FAIL | 116,684 | $0.35 | pass | 48,110 | $0.15 |
| `8e9d538c` | ss-user | pass | 118,317 | $0.36 | pass | 42,345 | $0.13 |
| `ad7109d1` | ss-user | FAIL | 114,263 | $0.34 | pass | 49,802 | $0.15 |
| `af8d2e46` | ss-user | pass | 114,690 | $0.35 | pass | 53,504 | $0.16 |
| `f4f1d8a4_abs` | ss-user | pass | 118,760 | $0.36 | pass | 46,426 | $0.14 |
| `0e5e2d1a` | ss-assistant | pass | 118,067 | $0.35 | pass | 45,569 | $0.14 |
| `1de5cff2` | ss-assistant | FAIL | 118,432 | $0.36 | pass | 45,809 | $0.14 |
| `28bcfaac` | ss-assistant | pass | 118,509 | $0.36 | pass | 44,713 | $0.14 |
| `41275add` | ss-assistant | FAIL | 118,490 | $0.36 | pass | 51,010 | $0.16 |
| `58470ed2` | ss-assistant | pass | 118,116 | $0.36 | pass | 80,240 | $0.25 |
| `6222b6eb` | ss-assistant | pass | 118,378 | $0.36 | pass | 41,408 | $0.13 |
| `8aef76bc` | ss-assistant | pass | 118,739 | $0.36 | pass | 32,131 | $0.10 |
| `ceb54acb` | ss-assistant | pass | 118,463 | $0.37 | pass | 45,166 | $0.14 |
| `dc439ea3` | ss-assistant | pass | 118,782 | $0.36 | pass | 57,967 | $0.18 |
| `e3fc4d6e` | ss-assistant | FAIL | 115,974 | $0.35 | pass | 51,285 | $0.16 |
| `f523d9fe` | ss-assistant | pass | 119,321 | $0.36 | pass | 58,638 | $0.18 |
| `1a1907b4` | ss-preference | FAIL | 117,865 | $0.35 | pass | 51,663 | $0.16 |
| `1da05512` | ss-preference | FAIL | 120,425 | $0.37 | pass | 54,796 | $0.17 |
| `b0479f84` | ss-preference | FAIL | 117,425 | $0.36 | pass | 48,987 | $0.15 |
| `b6025781` | ss-preference | FAIL | 119,376 | $0.36 | pass | 46,189 | $0.14 |
| `fca70973` | ss-preference | pass | 117,421 | $0.36 | pass | 59,228 | $0.19 |
| **Total** | **100** | **33** | **11,758,181** | **$35.56** | **95** | **5,234,716** | **$15.99** |

</details>


### LoCoMo

A full LoCoMo run is not yet published; the headline accuracy figures in this documentation are LongMemEval results, reported above.

Tests memory accuracy over extended multi-turn conversations. Questions are categorized by type:

| Question Type | Description |
|--------------|-------------|
| Single-hop | Direct recall of a stated fact |
| Multi-hop | Combining facts from different conversation turns |
| Open-ended | Questions requiring synthesis across topics |
| Temporal | Questions about when events occurred |
| Adversarial | Questions designed to confuse retrieval (similar topics, contradictions) |

### MRCR (Multi-Round Conversational Retrieval)

Tests retrieval precision across topic switches. The conversation covers multiple distinct topics, then questions target specific topics to measure whether retrieval surfaces the right segments without cross-contamination.

This is where the context bleed gate and active tag skipping are tested: the system must retrieve "the database migration discussion" without also pulling in "the API design discussion" that happened in adjacent turns.

### AMB (Agent Memory Benchmark)

Tests memory in agentic contexts where the model uses tools, executes code, and maintains state across complex multi-step tasks. This is the most realistic benchmark: conversations include `tool_use`/`tool_result` pairs, chain collapses, and interleaved planning discussions.

AMB tests whether chain collapse preserves recoverable information, whether fact extraction captures decisions made during tool use, and whether retrieval handles the mixed content types in agentic conversations.

## Benchmark Infrastructure

Benchmarks live in `benchmarks/` and are structured as:

```
benchmarks/
  locomo/
    run.py             # Entry point (python -m benchmarks.locomo.run)
    baseline.py        # Full-history baseline runner
    vc_runner.py       # Runs questions through the virtual-context pipeline
    scoring.py         # Scores responses against ground truth
    dataset.py         # Dataset loading
  longmemeval/
    ...
  mrcr/
    ...
  amb/
    ...
```

The directory also contains `beam/` and `writ/` harnesses.

Each benchmark runner:
1. Loads a dataset of conversations and questions
2. Replays the conversation through the virtual-context engine
3. Asks each question via the retrieval pipeline
4. Scores the response against ground truth
5. Reports accuracy by question type

Benchmarks use the engine directly (not through the proxy) for reproducibility and speed.

## Stress Tests

The stress test suite validates pipeline behavior under adversarial conditions. It uses a set of prompt files that exercise edge cases:

| Test Category | What It Tests |
|--------------|--------------|
| Topic cycling | Rapid switches between 10+ topics, verifying retrieval stability |
| Compaction cascade | 200+ turns forcing multiple compaction events, checking for content loss |
| Tag explosion | Conversations that generate 100+ unique tags, testing index performance |
| Concurrent access | Multiple simultaneous requests against the same session |
| Large payloads | Messages with embedded images, code blocks, and tool results exceeding 50K tokens per turn |
| Empty turns | Messages with no semantic content, testing graceful degradation |
| Contradiction storms | Sequences of contradictory facts, testing supersession |

Stress tests are runnable via the dashboard's Replay feature: point it at a prompt file and watch metrics update live.

## Running Benchmarks

```bash
# Run a specific benchmark
python -m benchmarks.locomo.run --config virtual-context.yaml

# Run with a specific provider
python -m benchmarks.locomo.run --provider anthropic --model claude-sonnet-4-20250514

# Run stress tests via the proxy dashboard
# 1. Start the proxy
virtual-context proxy --upstream https://api.anthropic.com

# 2. Open http://localhost:5757/dashboard
# 3. Use the Replay panel with a stress test file
```

## Interpreting Results

**Accuracy by question type** is the primary metric. Overall accuracy can mask weaknesses: a system might score 90% overall but 40% on temporal questions.

**Tokens freed** measures compaction efficiency. Higher is better, but not at the cost of accuracy. The goal is maximum compression with minimum information loss.

**Retrieval precision** measures what fraction of retrieved segments were actually relevant to the question. Low precision means the system is wasting context budget on irrelevant content.

**Compression ratio** is the ratio of summary tokens to original tokens. Typical values are 0.15-0.25 (4x-7x compression). Below 0.10 risks losing detail; above 0.30 suggests summaries are too verbose.

## Regression Tests

The test suite includes regression markers tied to specific bugs (`BUG-NNN` and `PROXY-NNN` series; see `tests/REGRESSION_MAP.md` for the full index). Each regression test reproduces the exact scenario that triggered a bug and verifies the fix. Run them via the `regression` marker:

```bash
pytest tests/ -m regression
```

Example regression areas (full descriptions in `tests/REGRESSION_MAP.md`):
- **BUG-012**: Tag splitter collects empty or wrong text for turns in proxy history
- **BUG-018**: Recent context turns pollute the inbound tagger after compaction
- **BUG-036**: Sort-key insertion gaps exhaust under repeated mid-history inserts
- **BUG-041**: Compaction materialized tag summaries for only a subset of segment tags

### Cache provenance

LongMemEval memory caches live below a content fingerprint of the dataset,
engine and harness source bytes, and non-secret memory configuration. A cache
is reusable only after its ingestion/compaction pipeline records completion.
Model, prompt, code, or input changes select a separate cache; changing reader
settings alone can reuse matching memory. Legacy caches are preserved but are
not trusted without a manifest. Interrupted runs require `--fresh` for that
fingerprint. `--recompact` rebuilds compaction within a matching fingerprint;
it does not reuse old tagging across changed pipeline configurations.

## Offline context contracts and resource measurements

The [versioned context corpus](../benchmarks/context_contracts/README.md) exercises
corrections, plans, author and channel isolation, assistant decisions, tool
artifacts, Unicode and out-of-order source arrival against real SQLite storage.
It reports source recall, actual rendered attribution, correction coverage,
abstention, delivered tokens, paging and retrieval work across layer ablations.
Controlled embeddings and proposals make these reproducible mechanism tests;
they do not measure an external model's answer quality.

The companion `benchmarks.context_contracts.resources` command measures peak
process memory, payload hydration and timings using generated archives in fresh
processes. PostgreSQL runs require an explicitly named scratch DSN and an already
migrated vector cache. It never selects a production database by default.
