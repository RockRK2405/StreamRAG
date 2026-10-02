# Samsung PRISM Theme 4
# Phase 4 — Streaming Engine & Retrieval Controller

| | |
|---|---|
| Date | 2026-10-02 |
| Status | Phase 4 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** (unchanged) |
| Code | `src/streamrag/{streaming,controller,ledger,replay}` + `bench/streaming.py` |
| Tests | 213 passing (140 Phase 3 + 4 new isolation cases + 69 new Phase 4 tests) |
| Data used | **DEV SUITE**: 63 streaming cases authored by the team over the *fictional test-fixture corpus* (`eval/dev_streaming/`, `is_fixture: true`) |

**Read this first: what the numbers mean.**

1. Every benchmark number in this report comes from the dev suite on the fixture domain. It is **NOT an official benchmark result**, and every output is stamped `REPORTABLE: false`.
2. The dev suite was **written by the same team that designed the controller**. It is not held-out. A high score shows that the implementation behaves as designed. It does **not** show generalization.
3. Latency figures are labeled as one of:
   - **virtual (modeled)**: deterministic, with retrieval latency fixed at 10 ms;
   - **realtime (measured)**: asyncio and wall clock at speed 1.0 on the dev machine (Apple M5 Pro).

---

## 1. Objective

Turn the static Phase 3 retrieval foundation into a streaming system:
- transcript chunks are processed incrementally;
- a controller decides **WAIT / RETRIEVE / SKIP**;
- one active, versioned query per utterance is built and tracked in a **Query Ledger**;
- retrieval runs **asynchronously**, so the stream never blocks;
- stale queries are handled without discarding evidence;
- every step is recorded as structured, **replayable** telemetry.

**Explicitly out of scope:** multi-intent decomposition (Phase 5), session refinement and answers (Phases 6–7), UI.

---

## 2. Existing Retrieval Foundation

I verified Phase 3 before building on it:

| Check | Result |
|---|---|
| Test suite | 140/140 passing |
| Retrieval API | Fixture index load: 1.1 ms. Hybrid `retrieve()` p50: 2.9 ms. |
| Evidence | `EvidenceSet` round-trips exactly through canonical JSON |
| **Thread safety** (new requirement for async retrieval) | 16 concurrent `retrieve()` calls from threads: no errors, and results identical to sequential calls |
| Config and harness | Load and run as documented |

**Defect affecting Phase 4, fixed first.** `BM25Index` had no IDF lookup, which the controller's anchor-strength signal needs. That was flagged as a Phase 3 hand-off. I added `BM25Index.idf` (derived from CSC column counts, using the same formula as indexing) and `term_idf(term)`.

**Contract amendments,** recorded in ADR-012:
- `CONTROLLER_DECISION` renamed `RETRIEVAL_DECISION`;
- added `SESSION_STARTED`, `TRANSCRIPT_UPDATED`, `QUERY_UPDATED`, `RETRIEVAL_CANCELLED`, `UTTERANCE_FINALIZED`;
- added `t_session_ms` to the envelope;
- new schemas `RetrievalDecision` and `QueryRecord` (20 schemas in `docs/schemas/`).

Phase 3 components were otherwise left unchanged, and all Phase 3 tests still pass.

---

## 3. Streaming Architecture

Diagrams: `docs/architecture/08_streaming_engine.md` (Mermaid, parse-verified). Docs: `docs/streaming/01–08`.

```
Transcript source ─► TranscriptChunkManager ─► transcript state ─► RetrievalController ─► WAIT | SKIP | RETRIEVE
                                                     ▲ timers (quiet tick, endpoint timeout)          │
                                                                                QueryBuilder ◄────────┘
                                                                                     ▼
                                    QueryLedger (Qn supersedes Qn-1; queued Qn-1 cancelled) ─► Async executor
                                                                                     │  worker thread
                                                                                     ▼
                                                                  Phase 3 RetrievalService (unchanged)
                                                                                     ▼
                       completion ─► ledger (evidence, status, timestamps; stale lineage kept) ─► TURN_COMPLETED
every step ─► EventBus (validated TelemetryEvent JSONL, write-only) ─► metrics / ReplayEngine / CLI debug stream
```

**One session state machine, two schedulers** (ADR-014):

| Mode | Behavior | Use |
|---|---|---|
| **virtual** | Deterministic discrete-event simulation. Retrieval executes for real; completion is delivered after a modeled latency (10 ms). | Tests, benchmarks, **exact replay** |
| **realtime** | asyncio + wall clock; retrieval in `asyncio.to_thread` with a timeout | True latency; proves non-blocking behavior |

**How the two modes stay consistent:**
- All session state is mutated on one thread (the event loop), so there are no locks.
- Both schedulers order callbacks by **(time, priority, sequence)**: completion before input before timer, then scheduling order.
- **Two clocks** (ADR-014 §1b):
  - controller decisions, their timers and the cooldown run on **stream time** (when the triggering input was due);
  - event timestamps and latency telemetry use **actual time**.
- As a result, realtime decisions depend only on the input stream. Every realtime trace replays in virtual mode with identical behavior (§12).

---

## 4. Transcript Chunk Model

**Input contract:** unchanged from Phase 2 (§5).
- `TRANSCRIPT_CHUNK{session_id, utterance_id, chunk_index, timestamp_s, text, stability, replaces_chunk_index, utterance_offset_s}`, followed by a separate `UTTERANCE_END`, as in the guide.
- The brief's `received_at` maps to `ChunkRecord.received_at_ms` (session stream clock).
- The brief's `is_final` maps to `stability == "final"`.

**`TranscriptChunkManager`:**
- `append_chunk`, `get_current_transcript`, `get_recent_chunks`, `finalize_utterance`, `reset`.
- Statuses: `accepted`, `duplicate`, `revision`, `out_of_order`, `gap` (missing indices reported), `empty`, `rejected_finalized`, `rejected_closed`.
- The transcript is always rebuilt in index order, and nothing is silently dropped.

**Utterance completion:** `UTTERANCE_END`; the endpoint timeout (fallback); implicitly, by a new utterance; or session end.

**Simulator:**
- `stream(chunks, interval_ms)`, `timed(chunks, words_per_second, jitter, seed)`, `from_case(BenchmarkCase)`;
- JSONL input files;
- deterministic.

---

## 5. Retrieval Controller

`controller/` (docs/streaming/03) is a transparent, rule-first decision procedure. It contains no LLM.

1. **Act classification.** PRESENTATION, SOCIAL, BACKCHANNEL or META with confidence ≥ 0.8 → **SKIP (suppressed)**.
2. **QueryBuilder** → empty query → WAIT `no_content` (or SKIP `empty_content` at the utterance end).
3. **Signals:** semantic stability, retrieval-worthiness, novelty, anchors (§6–§7).
4. **Provisional gates:** dangling → WAIT; insufficient specificity → WAIT; stability < 0.65 → WAIT; worthiness < 0.5 → WAIT.
5. **At utterance end:** an UNKNOWN (non-request) act with worthiness < 0.5 → **SKIP (not worthy)**. Information requests are always retrieved at the end (Phase 2 K4).
6. **Novelty** < 0.25 against the ledger → **SKIP (redundant)**, with `ledger_ref`.
7. **Storm guards:**
   - budget of 4 per utterance, with one slot reserved for the final query;
   - cooldown of 400 ms;
   - optional in-flight guard.
8. **RETRIEVE** (provisional or final).

**Decision object** (`RetrievalDecision`, schema exported):
- decision, reason, reasons, skip_kind, confidence, signals, tick, trigger, query_text, ledger_ref, policy, timestamps.
- **Confidence is computed by a documented formula**, and a test checks it against the signals. For RETRIEVE it is `0.4·stability + 0.4·worthiness + 0.2·novelty`.

**Example decision**, generated by the implementation (fixture index): `streamrag stream --text "so I wanted to ask | how high the wicks | should be trimmed" --interval-ms 400`. This is the decision on chunk 3; the query is "how high the wicks should be trimmed".

```json
{"decision": "RETRIEVE", "reason": "stable_retrieval_worthy_request", "confidence": 0.853,
 "signals": {"act": "INFO_REQUEST", "semantic_stability": 0.967, "retrieval_worthiness": 1.0,
             "novelty": 0.333, "anchors": 3, "anchor_terms": "high,wick,trim", "dangling": false}}
```

Check: 0.4·0.967 + 0.4·1.0 + 0.2·0.333 = 0.853. `novelty` is 0.333 because Q1 "how high the wicks" (chunk 2) already existed.

---

## 6. Retrieval-Worthiness

Retrieval-worthiness does **not** rest on keyword matching alone. It combines three things:

1. **Dialog structure.** Question words, aux-inversion, request heads, and the *object* of a transformation verb ("summarize **that**" is presentation; "summarize **the ladder storage rule**" is a request).
2. **Corpus statistics.** `anchor_strength` = max IDF of the query terms that exist in the **indexed corpus vocabulary**, normalized by the IDF of a term found in one chunk. Content absent from the corpus cannot make a fragment retrieval-worthy early.
3. **A generic lexicon** (`configs/controller_lexicon.yaml`; domain-neutral; anti-hardcoding tests pass).

**Formula:** `worthiness = base(act) + 0.25·anchor_strength + 0.10·min(anchors,3)/3 + 0.10·explicit_question`, with base `INFO_REQUEST = 0.55`, `UNKNOWN = 0.20`, suppressed acts `= 0`.

**Model-based alternative:** `PrototypeActClassifier` (embedding nearest-centroid over generic seed phrases), behind the same interface. Compared in §15.

---

## 7. Semantic Stability

**Formula:** `stability = 0.35·complete + 0.20·min(1, content_terms/3) + 0.20·has_anchor + 0.15·request_form + 0.10·persistence`.

| Component | Meaning |
|---|---|
| `complete` | 0 when the transcript ends in a dangling function word ("I need", "about the", "and") or a comma |
| `persistence` | Term-Jaccard with the previous decision's query (has the query stopped changing?). 1 on a *quiet tick* (no new words for 600 ms), a sentence boundary, or the utterance end. |
| Sufficiency gate (provisional) | ≥ 1 anchor, and ≥ 2 content terms or one very rare term (`anchor_strength ≥ 0.8`) |

**The brief's examples, as produced by the implementation** (CLI run, fixture index):

| Transcript | Decision | Reason |
|---|---|---|
| "I need" | WAIT | `no_content` |
| "I need information" | WAIT | `no_content` |
| "I need information about" | WAIT | `no_content` |
| "I need information about the fog signal" | **RETRIEVE** | stability 0.833, worthiness 0.867 |
| "… during a storm" | WAIT | `cooldown`; retrieved at the utterance end as the refined Q2 |

---

## 8. Query Versioning

- Each RETRIEVE creates a new version, `Q<n>`.
- Within an utterance, the new version **supersedes** the active one.
- `relation = refines` when at least 60% of the old query's terms survive (Q2 evolves Q1); otherwise `replaces`.
- Each record keeps `query_text`, `transcript_snapshot`, `source_spans` (exact char spans in the transcript), analyzed `terms`, `trigger` / `trigger_chunk` / `tick`, `decision_reason`, and status timestamps.

**QueryBuilder:**
- removes fillers anywhere, and the leading preamble (greetings, request heads such as "can you tell me");
- strips trailing dangling function words;
- **never removes a negation or an auxiliary verb**;
- keeps entities, numbers, constraints and terminology verbatim;
- records the span of every kept token.

---

## 9. Query Ledger

`QueryLedger` tracks:
- versions, lineage (`supersedes`, `superseded_by`, `lineage_root`) and `stale` flags;
- status (`pending → queued → in_flight → completed | failed | cancelled`) and retrieval timestamps;
- evidence associations;
- session-wide similarity (for novelty).

`evidence_view(utterance)` returns the active query's evidence first, then stale lineage evidence. It is deduplicated, and each item lists the queries that found it. This is the hand-off point for Phase 6 refinement.

---

## 10. Async Retrieval

- **Executors:** FIFO queue with 2 running slots.
  - Realtime: `asyncio.to_thread` plus a timeout, with completion delivered on the event loop.
  - Virtual: modeled completion time.
- **Non-blocking, verified by test:** with a deliberately slow backend (400 ms per call) and chunks every 100 ms, at least 2 chunks were received and processed between Q1's start and completion.
- **Failures** surface as `RETRIEVAL_COMPLETED(status=error|timeout)` plus a structured `ERROR` (`action=keep_previous_evidence`). The utterance still completes.

---

## 11. Stale Query Handling

docs/streaming/06, ADR-014.

| Superseded query is… | Policy |
|---|---|
| **Queued** | Cancelled: `RETRIEVAL_CANCELLED(superseded_before_start)`. It never reaches the backend (tested). |
| **In flight** | Allowed to complete, marked `stale`, evidence **kept** with lineage (`stale_at_completion = true`; tested). Rationale: retrieval costs ~3–10 ms on the fixture index (measured, §16), so cancellation saves almost nothing, and running threads cannot be safely interrupted. |
| **Already completed** | Kept as stale lineage evidence |

`cancel_superseded: never` turns queued cancellation off.

**When the policy matters:** in the virtual dev-suite runs, **0 queries were cancelled at 10, 200 and even 1,000 ms modeled latency**. With 2 concurrent slots, Q1 and Q2 never wait. A queued Q3 could only be cancelled by a Q4. The controller produced at most 3 retrievals per utterance on this suite, so no Q4 ever existed. Cancellation activates only when slots are saturated (e.g. `max_concurrent_retrievals = 1` with slow retrieval). That path is covered by a unit test, not by the benchmark.

---

## 12. Streaming Telemetry

**Event types in use:** SESSION_STARTED, CHUNK_RECEIVED, TRANSCRIPT_UPDATED, RETRIEVAL_DECISION, QUERY_UPDATED, RETRIEVAL_STARTED, RETRIEVAL_COMPLETED, RETRIEVAL_SKIPPED, RETRIEVAL_CANCELLED, UTTERANCE_FINALIZED, TURN_COMPLETED, ERROR, SESSION_CLOSED.

**Every event carries:**
- `event_id` (deterministic `session:seq`), `type`, `session_id`, `utterance_id`;
- `t_session_ms` (stream clock), `t_stream_s` (utterance-relative), `t_wall_ms`;
- `component`, and a structured payload.

**Timestamps recorded:**

| Timestamp | Field |
|---|---|
| `chunk_received_at` | CHUNK_RECEIVED |
| `controller_decision_at` | RETRIEVAL_DECISION, plus `wall.controller_ms` |
| `retrieval_started_at` | RETRIEVAL_STARTED, plus `queue_wait_ms` |
| `retrieval_completed_at` | RETRIEVAL_COMPLETED, plus `latency_ms` and `wall.measured_ms` |
| `utterance_finalized_at` | UTTERANCE_FINALIZED |

**Derived metrics,** computed from structured events, never from logs:
- `retrieval_lead_time = finalized − first retrieval start`;
- `time_to_first_retrieval`;
- `post_final_retrieval_latency`;
- `evidence_ready_slack`;
- controller and retrieval latency.

**`TURN_COMPLETED`** carries the guide-style `retrieval_events[{timestamp_s, query, trigger}]`, `sub_queries` (the single active query), evidence view, citations, metrics, and `answer: null`.

**Manual trace inspection** (realtime, speed 1.0; `research/phase4/traces/example_realtime.{txt,jsonl}`, regenerated with the final code):

```
[00:00.600] u1 USER       #2: "about the fog signal"
[00:00.602] u1 CONTROLLER RETRIEVE stable_retrieval_worthy_request conf=0.88 stab=0.833 worth=0.867
[00:00.602] u1 QUERY      Q1 v1 "the fog signal"
[00:00.602] u1 RETRIEVAL  START Q1 [provisional] queue_wait=0.16ms
[00:00.613] u1 RETRIEVAL  DONE  Q1 status=ok 10.734ms
[00:00.900] u1 USER       #3: "during a storm"
[00:00.902] u1 CONTROLLER WAIT     cooldown
[00:01.400] u1 UTTERANCE  FINALIZED reason=endpoint
[00:01.401] u1 CONTROLLER RETRIEVE utterance_end_final_query
[00:01.402] u1 QUERY      Q2 v2 "the fog signal during a storm" supersedes Q1 (refines)
[00:01.413] u1 RETRIEVAL  DONE  Q2 status=ok 11.247ms
[00:01.413] u1 TURN       retrievals=2 lead_time=797.722ms ttfr=602.177ms post_final=12.894ms final_query=Q2
[00:03.201] u2 CONTROLLER SKIP     presentation_restructure   ("make that | shorter")
[00:03.701] u2 TURN       retrievals=0 lead_time=n/a ttfr=n/a post_final=n/a
```

Checked by hand against the JSONL `t_session_ms` values:
- first chunk 0.306, Q1 start 602.483, finalized 1400.205, Q2 completed 1413.099;
- lead time 1400.205 − 602.483 = 797.722 ms ✓
- TTFR 602.483 − 0.306 = 602.177 ms ✓
- post-final 1413.099 − 1400.205 = 12.894 ms ✓
- Q1's evidence was ready 787 ms before the user finished.
- The cooldown deferred the refined query to the end, at a cost of 12.9 ms post-final.
- The utterance ended by its own UTTERANCE_END (`reason=endpoint`), with no ERROR events.
- The second utterance was suppressed without any retrieval.

**Replay validation:**

| Check | Result |
|---|---|
| Virtual trace (`wick_height-authored`, benchmark run) | **identical**, 32/32 events, 0 differences (`results/replay_check.json`) |
| CLI virtual example (`streamrag replay`) | identical, 34/34 events, exit 0 |
| CLI realtime example | `behavior_identical: true`, exit 0. Exact comparison lists 20 timing-only differences, as expected. |
| **All 189 realtime dev-suite traces** (A/B/C × 63) replayed in virtual mode | **189/189 behavior-identical** (`results/realtime_replay_parity.json`) |
| Tampered trace (one RETRIEVE changed to WAIT) | detected (tested) |

---

## 13. Early Retrieval Benchmark

**Data:** 63 dev cases (33 utterances × up to 2 chunkings).
- 44 turns require retrieval.
- **41 are eligible** (retrieval required and ≥ 2 chunks).
- 19 require no retrieval.

**Gold:** fixture citations only.

**Virtual mode (deterministic; modeled retrieval latency 10 ms). Dev suite, NOT REPORTABLE:**

| Metric | Controller (C) |
|---|---|
| **early_retrieval_rate** (eligible) | **0.951** (39/41) |
| early rate on eligible in-vocabulary questions | 39/39 |
| useful_early_rate (first early retrieval shares top-5 with the final query) | 0.951 |
| missed_retrieval_rate | 0.000 |
| lead time p50 / p95 | 1,925 / 3,984 ms |
| time-to-first-retrieval p50 / p95 | 953 / 3,275 ms |
| evidence ready at utterance end | 0.886 (39/44) |
| post-final retrieval latency p50 / p95 | 0 / 10 ms (modeled) |
| fixture final-query success@5 (sanity, fixture gold) | 1.000 |

**The two non-early eligible cases** are both chunkings of "who won the cricket | championship yesterday". No word in it exists in the corpus vocabulary, so by design (K4) it is retrieved only at the utterance end and is never suppressed.

**Realtime mode** (asyncio, wall clock, speed 1.0, real retrieval in a worker thread). Measured on the dev machine; dev suite, NOT REPORTABLE (`research/phase4/results/realtime/`):

| Metric | Controller (C), realtime | (virtual, for comparison) |
|---|---|---|
| early_retrieval_rate (eligible) | **0.951** (39/41) | 0.951 |
| lead time p50 / p95 | **1,923 / 3,984 ms** | 1,925 / 3,984 ms |
| time-to-first-retrieval p50 / p95 | 954 / 3,276 ms | 953 / 3,275 ms |
| evidence ready at utterance end | 0.886 | 0.886 |
| post-final retrieval latency p50 / p95 / max | **0 / 10.1 / 11.3 ms** | 0 / 10 / 10 ms (modeled) |

- Realtime reproduces every virtual decision. For all 189 realtime traces (63 cases × A/B/C): the same per-turn retrieval counts, the same early and non-early cases, and per-turn lead times within 3.3 ms. Replaying each realtime trace in virtual mode gives identical behavior in 189/189 (`results/realtime_replay_parity.json`). Virtual mode is therefore a faithful stand-in for benchmarking.
- The two non-early cases are again the cricket utterance.
- **Early retrieval occurs only where it actually happened:**
  - `retrieved_early` is computed from the RETRIEVAL_STARTED and UTTERANCE_FINALIZED timestamps.
  - In realtime, end-only (A) retrieval starts *after* finalization in every case (lead time p50 −0.85 ms; all 41 values negative). It is correctly counted as not early, not rounded up to 0.

---

## 14. Retrieval Suppression Benchmark

19 non-retrieval turns:
- backchannel ("okay", "mm-hm | okay | right");
- social ("thanks | that helps a lot", "got it | thank you so much");
- presentation ("could you | repeat the previous answer | please", "make that | shorter", "can you summarize | the retrieved evidence | for me", "translate that | into hindi");
- meta ("what did I | ask you before");
- casual ("hmm | let me think | for a second", "I'm going to | grab a coffee | first").

| Metric (virtual, dev suite) | A end-only | B every-chunk | **C controller** | C prototype |
|---|---|---|---|---|
| false_retrieval_rate | 0.895 | 0.895 | **0.000** | 0.000 |
| suppression_rate | 0.105 | 0.105 | **1.000** | 1.000 |
| retrievals on non-retrieval turns | 17 | 31 | **0** | 0 |

**How C skipped the 19 turns** (identical in virtual and realtime):
- 15 turns only by `suppressed` (presentation, social, backchannel or meta);
- 3 turns only by `not_worthy` (casual statements; decided at the utterance end);
- 1 turn by both.

That is 64 SKIP decisions in total (60 suppressed, 4 not_worthy), because a suppressed act SKIPs on every tick it is evaluated.

A and B reach 0.895 rather than 1.0 only because two turns (`ack_okay-authored` "okay" and `ack_mmhm-authored` "mm-hm okay right") reduce to an empty query after filler removal.

Realtime: identical rates (C 0.000 false / 1.000 suppression; A and B 0.895 / 0.105).

---

## 15. Controller Ablation

**Virtual mode, identical inputs.** Dev suite, NOT REPORTABLE.

| Metric | **A** end-only (static) | **B** every chunk | **C** controller (rules) | C′ prototype classifier |
|---|---|---|---|---|
| Retrievals total (63 turns) | 61 | **174** | 89 | 77 |
| Retrievals per required utterance (mean / max) | 1.00 / 1 | **3.25 / 12** | 2.02 / 3 | 1.75 / 4 |
| Early retrieval (eligible) | **0.000** | 1.000 | 0.951 | 0.878 |
| False retrieval (non-retrieval turns) | 0.895 | 0.895 | **0.000** | **0.000** |
| Missed retrieval | 0.000 | 0.000 | 0.000 | **0.114** |
| Duplicate retrievals (identical query text) | 0 | 3 | **0** | 0 |
| Evidence ready at utterance end | 0.000 | 1.000 | 0.886 | 0.897 |
| Post-final latency p50 (modeled) | 10 ms | 0 | 0 | 0 |
| Controller decision cost p50 / p95 (wall, 411 decisions) | 0.06 / 0.16 ms | 0.08 / 0.21 ms | **0.14 / 0.33 ms** | 2.74 / 4.35 ms |
| Retrieval wall time p50 / p95 (real execution, back-to-back) | 3.15 / 4.00 ms | 3.25 / 4.49 ms | 3.04 / 3.54 ms | 3.09 / 4.00 ms |
| Fixture final-query success@5 (sanity) | 1.000 | 1.000 | 1.000 | 0.929 |

**Reading the ablation.**
- **A (static RAG):** never retrieves early, and retrieves on almost every non-retrieval turn.
- **B (naive streaming):** maximally early, but 2× C's retrievals (up to 12 per utterance), the same false-trigger problem as A, and duplicate searches.
- **C:** B's earliness on in-vocabulary questions at about half the retrievals, with **zero** false triggers and zero duplicates.
- **C′ (embedding prototype classifier, untuned seeds):** misses 5 required retrievals (wick_height ×2, tool_shed, cricket ×2). It is ~20× costlier per decision (p50 2.74 ms vs 0.14 ms) because it embeds every tick. **Rules win on this dev suite.** This is the rule-vs-model ablation the guide suggests [G§8 p6].

**Sensitivity to retrieval latency** (C vs A, virtual, modeled latency 10 / 200 / 1000 ms):

| Modeled latency | C: early rate | C: evidence ready at end | C: post-final p50 / p95 | A: post-final |
|---|---|---|---|---|
| 10 ms | 0.951 | 0.886 | 0 / 10 ms | 10 ms |
| 200 ms | 0.951 | 0.886 | 0 / 200 ms | 200 ms |
| 1000 ms | 0.951 | **0.136** | **500 / 1000 ms** | 1000 ms |

**Finding.** At slow retrieval (e.g. with the cross-encoder or a remote index), early retrieval still happens (0.951). However, the *final query's* evidence is rarely ready at the end. Either the last provisional query started less than 1 s before the end, or the final transcript yields a new query version that is retrieved only at the end. C then still halves A's post-final latency at p50 (500 vs 1,000 ms) but matches it at p95. Mitigation options for later phases:
- reuse the provisional evidence when the final query only *refines* it (high containment);
- answer from provisional evidence while the final retrieval completes.

**Realtime ablation** (asyncio, wall clock, speed 1.0; same 63 inputs; the prototype variant was run in virtual mode only):

| Metric (realtime, measured) | A end-only | B every chunk | **C controller** |
|---|---|---|---|
| Retrievals total | 61 | 174 | **89** |
| Retrievals per required utterance (mean / max) | 1.00 / 1 | 3.25 / 12 | **2.02 / 3** |
| Early retrieval (eligible) | 0.000 | 1.000 | **0.951** |
| False retrieval | 0.895 | 0.895 | **0.000** |
| Duplicate retrievals | 0 | 3 | **0** |
| Lead time p50 | −0.85 ms (after the end) | 2,628 ms | **1,923 ms** |
| Post-final retrieval latency p50 / p95 | 10.6 / 14.9 ms | 0 / 0 ms | **0 / 10.1 ms** |
| Retrieval wall time p50 / p95 | 9.5 / 13.4 ms | 9.7 / 11.9 ms | 9.3 / 11.5 ms |
| Controller decision wall time p50 / p95 | 0.32 / 0.80 ms | 0.31 / 0.68 ms | 0.58 / 1.35 ms |

**Takeaways:**
- The controller turns A's ~11 ms post-final retrieval wait into 0 ms for the median utterance.
- It does this with 51% of B's retrieval calls and no false triggers.
- At ~10 ms per retrieval, the absolute saving is small on this tiny fixture index. The value grows with retrieval latency (sensitivity table above).

---

## 16. Performance

**Approach:** identify the bottleneck first; nothing was optimized.

**Profiling pass:** realtime, controller, speed 1.0, all 63 dev cases. Per-stage wall time on the dev machine (Apple M5 Pro, CPU only), from `research/phase4/results/profile.json`:

| Stage | n | p50 | p95 | max |
|---|---|---|---|---|
| Chunk processing (append, transcript update, decision, timers; excluding retrieval) | 214 | 0.91 ms | 1.70 ms | 2.73 ms |
| Controller decision (act, query build, signals, guards) | 411 | 0.63 ms | 1.36 ms | 2.28 ms |
| Query construction alone (QueryBuilder) | 214 | 0.06 ms | 0.12 ms | 0.16 ms |
| Event emit (validate + record one TelemetryEvent) | 1,530 | 0.02 ms | 0.07 ms | 0.19 ms |
| Retrieval queue wait (submission → worker thread start) | 89 | 0.09 ms | 0.12 ms | 0.13 ms |
| **Retrieval wall time** (hybrid BM25 + dense, ONNX query embedding, fixture index) | 89 | **9.50 ms** | **12.06 ms** | 13.73 ms |

**Bottleneck:** retrieval itself, specifically query embedding plus search. It is about 15× the controller cost per decision. The streaming engine adds about 1 ms per chunk, which is negligible against the dev suite's inter-chunk gaps (328–2,130 ms, median 1,044 ms; 151 gaps).

**Why realtime retrieval (~9.5 ms) is slower than the same call in virtual mode (~3.0 ms).** I measured this rather than assumed it (`research/phase4/latency_idle_probe.py`, results in `results/latency_idle_probe.json`):

| Same hybrid `retrieve()` call (fixture index, n = 40) | p50 | p95 |
|---|---|---|
| Main thread, back-to-back | 2.99 ms | 3.29 ms |
| Worker thread (`asyncio.to_thread`), back-to-back | 3.17 ms | 3.65 ms |
| Main thread, after a 300 ms idle pause | 9.30 ms | 11.48 ms |
| Worker thread, after a 300 ms idle pause | 9.49 ms | 11.90 ms |

**Conclusions:**
- The thread hand-off costs about 0.2 ms.
- The 3× penalty comes from the **CPU being idle before the call**. In a live stream that is always the case, because the user speaks between chunks.
- The precise mechanism was not isolated (candidates: CPU frequency or core scheduling, cache state, ONNX Runtime thread-pool wake-up).
- **Consequence:** realtime numbers are the representative ones for live use. Virtual mode's 10 ms modeled latency sits between the realtime p50 (9.5 ms) and p95 (12.1 ms). A keep-warm strategy is a possible later optimization, *not* applied now.

**Scale:** all of this is measured on the 14-chunk, 3-document fixture index. Retrieval cost on the official corpus is unmeasured. The full benchmark ran in 1,020 s in total, most of it realtime cases running at speaking speed.

---

## 17. Failure Cases

**Found during Phase 4 and fixed:**

1. **Premature endpoint timeout truncated an utterance** (dev suite, `stack_height-authored`).
   - At 2.6 words/s with jitter, consecutive 5–6-word chunks arrive up to 2.10 s and 2.13 s apart (chunk 1 → 2 and 5 → 6 of this case). The 2,000 ms "no new chunk" timeout fired after the first chunk, and the remaining chunks were rejected as `rejected_finalized`.
   - **Fix:** default raised to 3,000 ms. It is a fallback only; recorded in ADR-014 and the config comment. Controller thresholds were **not** changed.
2. **Replay was not identical.**
   - (a) A wall-clock value (`controller_wall_ms_max`) sat in the TURN_COMPLETED metrics. It now lives under `wall`.
   - (b) Inputs that only produced an `ERROR` (a duplicate `UTTERANCE_END`) were not recoverable. ERROR now carries `input_event`.
   - Replay is now identical (tested; final results in the §12 replay validation table).
3. **QueryBuilder stripped meaningful trailing verbs** ("what do workers **do**"). Auxiliary verbs are no longer stripped.
4. **YAML parsed `no`, `yes` and `on` as booleans** in the lexicon. They are now quoted, and the loader fails loudly on non-strings.
5. **Realtime runner waited for no-op timers after session close.** Each realtime case took ~17 s instead of its true duration. `drain()` now waits only for the inputs and in-flight retrievals.
6. **Realtime scheduler dispatched same-timestamp events in arbitrary order** (found during final validation, by manual trace inspection).
   - SESSION_END shares the timestamp of the last UTTERANCE_END.
   - The old scheduler armed one `call_later` per callback, with a delay computed from a fresh `now`, so the later-scheduled SESSION_END usually fired first.
   - Result: the utterance was finalized as `session_end`, followed by a spurious `DuplicateUtteranceEnd` ERROR, in **109 of 189 realtime dev-suite runs**. Timing metrics were barely affected, because both events fall at the same instant.
   - **Fix:** the realtime scheduler now uses the same (time, priority, seq) heap as the virtual one.
   - Regression tests: the old logic misordered 20/20 trials; the new one passes. The realtime benchmark was rerun.
7. **Controller decisions depended on execution delay** (found by the new realtime-replay behavior check).
   - The cooldown was measured from the previous retrieval's *start*, which in realtime includes ~2 ms of processing and queue delay.
   - A chunk arriving exactly one cooldown (400 ms) later was RETRIEVE in virtual mode but WAIT in realtime.
   - **Fix:** the controller runs on stream time, and the cooldown counts from when the previous query was *issued*.
   - Virtual results are unchanged (identical counts, rates and lead times). Realtime traces now replay with identical behavior.

**Remaining known failure modes:**

8. **Trailing negation treated as dangling.** "are ladders allowed … overnight **or not**" ends in "not", so provisional retrieval of the complete question waits until the utterance end, and the final query is retrieved after the end (10 ms modeled). This is not fixed, to avoid tuning on the dev suite. It is a candidate rule: "or not" as an ending is complete.
9. **Words never seen in the corpus get no early retrieval** (by design). Those requests incur the full post-final latency.
10. **Common words can become anchors in a tiny corpus** ("night", "high", "first" in the fixture). This could cause premature provisional retrieval on a real but small corpus. Anchor floors must be re-checked on the official corpus.
11. **A late chunk after a *timeout* finalization is rejected,** with an explicit error, rather than reopening the utterance.
12. **The prototype classifier misclassifies** some plain questions. It missed required retrievals for `wick_height` ("so I wanted to ask | how high the wicks | should be trimmed | in the lighthouse", both chunkings) and `tool_shed-authored`.

---

## 18. Test Results

`.venv/bin/python -m pytest -q` → **213 passed, 0 skipped, 0 failed** (real-model tests included; ~7 s).

| Suite | Tests |
|---|---|
| Phase 3 suite (components unchanged) | 140 (as at the end of Phase 3) |
| Phase 3 isolation guard, parametrized over the 4 new packages | +4 |
| `tests/streaming/`: chunk manager, events, simulator, session integration CASES 4–10, async realtime (non-blocking, timeout, equal-time ordering, UTTERANCE_END before SESSION_END), replay (exact virtual, realtime behavior parity, tampered trace detected), metrics, benchmark harness, CLI | 35 |
| `tests/controller/`: query builder, decisions CASES 1–3, suppression, guards, signals and confidence formula, ablation policies, prototype interface | 30 |
| `tests/ledger/`: versions, lineage, status, issue-time cooldown reference, evidence view | 4 |
| **Total** | **213** |

**Other checks:**
- **Phase 3 regressions:** none. All 140 Phase 3 tests pass. The only Phase 3 code change is the additive `BM25Index.idf` / `term_idf` (§2).
- **Isolation:** the 4 new packages are included in the no-network import scan.
- **Anti-hardcoding:** the guard scans `src/` and `configs/` (including the controller lexicon) for every dev-suite utterance of ≥ 4 words, guide example vocabulary and benchmark-specific branches. It passes.
- **Lint / type checking:** not configured in this project (no ruff, mypy or flake8 in `pyproject.toml`). As a substitute, `compileall` passed and all modules import cleanly.
- **Replay:** see §12 and the benchmark's `replay_check.json` / `realtime_replay_parity.json`.

### Quality gate

| Gate | Status | Evidence |
|---|---|---|
| Phase 3 validated; defects fixed and recorded | ✅ | §2 (IDF accessor) |
| Chunk model + `TranscriptChunkManager` (duplicate, out-of-order, empty, gap, new utterance, session end) | ✅ | §4; tests |
| Event vocabulary (all required types, with event_id, type, session_id, utterance_id, timestamps) | ✅ | §12; schema-validated `TelemetryEvent` |
| Deterministic simulator `stream(chunks, interval_ms)` | ✅ | §4 |
| Controller WAIT/RETRIEVE/SKIP, with transparent decision objects (values computed, not authored) | ✅ | §5–§7; confidence formula tested |
| QueryBuilder: fillers removed; entities, negations and terminology kept; traceable spans | ✅ | §8; tests |
| Query versioning + QueryLedger (lineage, stale, duplicates, evidence) | ✅ | §8–§9 |
| Async, non-blocking retrieval; documented cancellation policy | ✅ | §10–§11; non-blocking test |
| Timestamps, lead time, TTFR, latency telemetry (from structured events only) | ✅ | §12 |
| Early-retrieval and suppression benchmarks | ✅ (dev suite only; **not reportable**) | §13–§14 |
| Controller ablation A/B/C (+ prototype variant) | ✅ (dev suite only) | §15 |
| Performance p50/p95, bottleneck identified, no premature optimization | ✅ | §16 |
| Replay: virtual exact; realtime behavior-identical | ✅ | §12 |
| Externalized config, one canonical schema | ✅ | `configs/default.yaml` (`controller:`, `streaming:`), `configs/controller_lexicon.yaml` |
| Docs `docs/streaming/01–08`, `docs/architecture/08` (Mermaid parse-verified), ADR-014 | ✅ | |
| Evaluation mapping updated without unsupported claims | ✅ | `docs/evaluation/evaluation_mapping.md` |
| No fabricated metrics; fixtures labeled; no official-result claims | ✅ | `REPORTABLE: false` on every output |
| Out of scope respected: no multi-intent, no answer generation, no grounding validator, no frontend | ✅ | `TURN_COMPLETED.answer = null` |
| Lint / type checking | ⚠️ not configured | compile + import check only |
| Official corpus / official streaming cases | ❌ blocked (external) | §19 |

---

## 19. Known Limitations

1. **No official data.** All controller numbers are dev-suite numbers on the fixture domain, and the dev suite is not held-out. Thresholds were set from Phase 2 reasoning, not tuned. Their real-world validity is unknown until official streaming cases exist.
2. **Virtual mode uses a constant modeled retrieval latency.** Realtime numbers come only from the dev machine at speed 1.0, not judge hardware.
3. **The controller lexicon is English-only and generic.** Unusual phrasings may misclassify, and "or not" endings are a known case.
4. **Single active query only.** Compound utterances are retrieved as one query, so multi-intent coverage waits for Phase 5.
5. **Cancellation only affects queued queries;** running retrievals always complete.
6. **No ASR integration.** Transcript replay only (allowed by [D s7]). Partial-hypothesis revisions are supported by the chunk manager but untested with a real ASR.
7. **TTFT cannot be measured yet,** because there is no generation. Post-final retrieval latency is the current proxy.

---

## 20. Phase 5 Requirements

**Ready for Phase 5:**
- streaming session and controller;
- a per-utterance transcript with chunk spans;
- the query ledger with lineage and evidence view;
- async executor (multiple concurrent slots already supported);
- structured telemetry and replay;
- dev-suite benchmark harness (multi-intent metrics can be added to `evaluate_run`).

**Exact prerequisites:**

1. **Segment-level state.** Extend the chunk manager or controller from "whole utterance = one query" to *segments* (clause boundaries are already detected for `dangling`). This is the unit Phase 5 decomposes.
2. **Ledger support for multiple active queries per utterance.** Today a new version supersedes the single active one. Phase 5 needs a per-intent active query (`intent_id` on `QueryRecord`) and parallel dispatch, which the executor already allows (2 slots, configurable).
3. **Multi-intent gold** in the streaming case format. `BenchmarkCase` already supports multiple `GoldIntent`s per turn; dev cases with compound utterances must be authored. The official ones are still needed (Q1/Q3).
4. **Per-intent fusion.** Phase 3 fusion is single-query. Phase 5 adds quota round-robin across intents (ADR-009) on top of `EvidenceSet.per_intent`.
5. **Address the slow-retrieval finding** (§15): provisional-evidence reuse for refined final queries, before enabling the cross-encoder in streaming.
6. **Still blocked externally:** the official corpus, official streaming test cases and labels, and the deadline (Q1, Q3, Q6).
