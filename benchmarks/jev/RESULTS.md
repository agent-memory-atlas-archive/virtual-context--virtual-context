# Jev judgment harness: first results

Run: 2026-09-16, late evening ET (result files stamped 2026-09-17T02:37Z to 02:38Z UTC).
Model returned by the API: `jev-1.13.0` (requested `jev-latest`). Legacy admission model:
`qwen/qwen3-235b-a22b-2507` through OpenRouter. Result files: `benchmarks/jev/results/`.
Every Jev request is cached by SHA-256 under `benchmarks/jev/cache/`; reruns are free.

Label sources. "Strong" rows are hand-written by the harness author (small sets; author
bias applies). "Weak" rows come from the LongMemEval 500-question file: for intent,
`knowledge-update` questions are labeled `current_state` and every other type `default`;
for temporal, `temporal-reasoning` questions are labeled true and every other type false.
Weak labels are approximate and are reported separately.

## rerank (n=140 cached LongMemEval questions, all scored)

| mode | gold in selected | gold found anywhere | mean rank of first gold | median rank | mean selected tokens | mean Jev ms | mean Jev input tokens |
|---|---|---|---|---|---|---|---|
| legacy | 94.3% | 94.3% | 2.33 | 1.0 | 1390 | 0 | 0 |
| jev | 94.3% | 94.3% | 1.14 | 1.0 | 1390 | 214 | 2341 |

| question type | n | legacy gold-in-selected | jev gold-in-selected |
|---|---|---|---|
| knowledge-update | 23 | 95.7% | 95.7% |
| multi-session | 37 | 97.3% | 97.3% |
| single-session-assistant | 12 | 75.0% | 75.0% |
| single-session-preference | 6 | 83.3% | 83.3% |
| single-session-user | 30 | 93.3% | 93.3% |
| temporal-reasoning | 32 | 100.0% | 100.0% |

First-gold rank per question: Jev better in 45, worse in 9, unchanged in 86. By type
(better/worse/same): knowledge-update 7/2/14, multi-session 8/3/26,
single-session-assistant 3/0/9, single-session-preference 1/0/5, single-session-user 10/1/19,
temporal-reasoning 16/3/13.

Observations from the numbers:
- The token budget never cut a candidate in any of the 140 questions (mean shortlist 14.3
  summaries, all selected), so membership of the selected set cannot differ between modes
  with the default budget; only order differs. The rerank moved the first gold summary from
  a mean position of 2.33 to 1.14.
- Jev latency and tokens are from the 46 live calls in this run; the rest were cache hits
  from an earlier partial run.
- One question, `2698e78f_abs`, returned no candidates in either mode despite 5 gold
  segments in its store.
- The engine logs `canonical turn bootstrap failed` with `database disk image is malformed`
  when opening these older cached stores (copied through the SQLite backup API; all 151
  stores pass `pragma quick_check` and their FTS tables pass `integrity-check`). Retrieval
  still returned candidates for 139 of 140 questions and both modes share the same engine
  state per question. The cause is not established.

## intent (S2, find_quote current_state vs default)

n=530 (strong=30, weak=500)

| side | accuracy all | accuracy strong | accuracy weak |
|---|---|---|---|
| legacy | 84.5% | 83.3% | 84.6% |
| jev_raw | 83.0% | 96.7% | 82.2% |
| jev_deployed | 83.8% | 96.7% | 83.0% |

Jev mean latency 191 ms, mean input tokens 392.
Strong misses: legacy 5 (h3, h5, h7, h8, h23: present-state questions without a trigger
word); Jev 1 (h17, the deliberate "led or am currently leading" disjunction, p=0.97
current_state). Weak confusion: legacy caught 23 of 94 knowledge-update questions and flagged
11 of 436 others; Jev caught 54 of 94 and flagged 50 of 436. Raising the deployed confidence
floor from 0.5 to 0.9 moves weak accuracy from 82.2% to 83.2% and leaves strong accuracy at
96.7%.

## temporal (S3, inbound temporal flag)

n=524 (strong=24, weak=500)

| side | accuracy all | accuracy strong | accuracy weak |
|---|---|---|---|
| legacy | 73.1% | 66.7% | 73.4% |
| jev_raw | 74.6% | 83.3% | 74.2% |
| jev_deployed | 74.6% | 83.3% | 74.2% |

Jev mean latency 181 ms, mean input tokens 364.
Strong misses: legacy 8 (every hand-written when/order question; the pattern list only
covers "first thing", "beginning", "early on", "initially"); Jev 4 (p4 at p=0.37, and three
false positives n2, n6, n12 at p=0.72, 0.68, 0.66). On weak rows legacy detected 4 of 133
temporal-reasoning questions with 0 false positives; Jev at the default 0.5 threshold
detected 127 of 133 with 123 false positives out of 367.

Threshold sweep from the saved probabilities (no new calls):

| noul_threshold | strong acc | weak acc | weak TP / 133 | weak FP / 367 |
|---|---|---|---|---|
| 0.5 | 83.3% | 74.2% | 127 | 123 |
| 0.6 | 83.3% | 76.2% | 120 | 106 |
| 0.7 | 87.5% | 81.8% | 114 | 72 |
| 0.8 | 91.7% | 84.6% | 112 | 56 |
| 0.9 | 91.7% | 88.6% | 108 | 32 |

## safety (S4, safety-critical personal evidence)

n=24 (all strong, hand-written)

| side | accuracy all |
|---|---|
| legacy | 66.7% |
| jev_raw | 100.0% |
| jev_deployed | 100.0% |

Jev mean latency 193 ms, mean input tokens 390.
Legacy misses 8 of 12 positives (starts, switches, corrections and moves phrased without the
patterns' stop/correction vocabulary) and none of the 12 negatives. Jev misses none.

## admission (S5, actor-card admission)

sets=10, candidates=13

| side | reason accuracy | admit/reject accuracy | coverage accuracy | mean ms | mean input tokens |
|---|---|---|---|---|---|
| legacy | 61.5% | 92.3% | 60.0% | 2074 | 3568 |
| jev | 61.5% | 76.9% | 80.0% | 264 | 1609 |

Legacy vs Jev reason agreement: 38.5%.

Per candidate (expected | legacy | jev):

| set | cand | expected | legacy | jev |
|---|---|---|---|---|
| broski | c1 | durable | durable | durable |
| broski | c2 | insufficient_evidence | insufficient_evidence | insufficient_evidence |
| greeting | c1 | not_durable | not_durable | insufficient_evidence |
| third_party | c1 | wrong_subject | wrong_subject | durable |
| agent_persona | c1 | wrong_kind | insufficient_evidence | wrong_subject |
| agent_persona | c2 | durable | expired | temporary |
| external_action | c1 | wrong_kind | not_person_card | wrong_kind |
| durable_history | c1 | durable | durable | completed |
| durable_history | c2 | durable | durable | durable |
| bot_test | c1 | test_probe | not_person_card | test_probe |
| privacy | c1 | explicit_privacy_request | explicit_privacy_request | explicit_privacy_request |
| refused | c1 | agent_refused | agent_refused | agent_refused |
| completed_goal | c1 | completed | contradicted | completed |

Observations from the table: the two admit/reject misses unique to Jev are `third_party`
(a fact about the actor's father admitted as the actor's own) and `durable_history c1` (a
finished marathon rejected as `completed` although the kind is relevant_history). Legacy's
reason errors all still land on reject. Coverage: legacy labeled four substantive
interactions `no_durable_context`; Jev labeled one (`privacy`).

## Not measured here
Tag select-instead-of-generate, summary faithfulness, retrieval gate, tag consolidation,
hint ranking. Production shadow mode has not been enabled anywhere; every deployment stays
on `legacy`.
