# Samsung PRISM Theme 4
# Phase 6 — Adaptive Streaming RAG

| | |
|---|---|
| Date | 2026-10-03 |
| Status | Phase 6 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** (unchanged) |
| Code | `src/streamrag/{session,context,delta,claims,answers}` + `bench/adaptive.py`. Extensions to `intents/` (cross-turn operations, elliptical follow-ups, topic return), `multi_retrieval/coordinator.py` (streaming integration behind `session.enabled`), `replay/`, `ledger/`, `retrieval/bm25.py` (co-occurrence), CLI `stream --session` |
| Tests | **396 passing, 0 skipped**: 320 from Phases 3–5 and 76 new (`tests/session` 29, `tests/context` 15, `tests/delta` 13, `tests/claims` 8, `tests/answers` 5, CLI +1, isolation scan +5 packages). Two Phase 5 tests were updated for documented behaviour changes. `compileall` passes for all 106 modules. Lint / type checking is not configured. |
| Data | **DEV SUITE:** 28 sessions / 64 turns (`eval/dev_adaptive/`). **STRESS SET:** 14 sessions / 31 turns (`eval/dev_adaptive_stress/`), written after the code freeze. Both are team-authored over the *fictional* fixture corpus plus a 2-document conflict fixture, with `is_fixture: true` |
| Output of the phase | A versioned, claim-level **answer state** per topic frame, with diffs and minimal-regeneration markers. No answer text, citation UI, voice, agents, web search or long-term memory. |

**Read this first: what the numbers mean.**

1. **Not official.** Every number comes from fixture-domain dev data (3 documents / 14 chunks, plus 2 conflict documents). All result files carry `REPORTABLE: false`.
2. **Not held-out.**
   - The dev suite was written after most of the session rules, by the same team, on utterances similar to those the rules were developed on. Its near-perfect scores are **optimistic**.
   - The stress set was written after the code freeze, with deliberately different phrasing, and run once with no code change in response. Its scores are the more honest signal: change types 0.903, need queries 0.867.
3. **Latency** is measured on the dev machine (Apple M5 Pro, CPU only, bge-small ONNX).
   - Each turn's time is the median of 7 repetitions.
   - The fixture corpus makes a retrieval cost about 5 ms, so absolute savings are small. They grow with retrieval cost, session length and the number of needs in a topic frame (§18, scaling run).
   - Across four full benchmark runs, the change-turn latency saving varied between 8.3 % and 10.1 %, and the 16-turn scaling saving between 63.2 % and 63.7 %. The final run is reported.
4. **No token savings for generation are claimed.** There is no generator yet. Token counts below are bge-WordPiece *proxies* of context or re-render size.

**Quality gate (brief §52):**

| Gate | Status | Where |
|---|---|---|
| SessionState, SessionMemory, session versions | ✅ | §3, §4 |
| Context changes detected, change types represented | ✅ | §5, §6 |
| DeltaPlanner, delta queries, query lineage | ✅ | §7, §8 |
| Existing evidence reused, evidence lifecycle tracked | ✅ | §9, §21 |
| Claim model, claim-evidence relationships, affected claims, targeted revalidation | ✅ | §10–§12 |
| Answer versions, answer diffs | ✅ | §13, §14 |
| Entity corrections, constraint addition / removal, new intents, follow-ups | ✅ dev suite; stress-set failures in §22 | §5, §17 |
| Context contamination controlled | ✅ (0 isolation leaks on both sets) | §16 |
| Memory reset; replay | ✅ (exact virtual replay 28/28 sessions; realtime by behaviour) | §4, §20 |
| Delta retrieval benchmark, full-restart baseline, actual measurements | ✅ | §17–§21 |
| Tests pass; no fabricated results | ✅ 396 passing; every number is in `research/phase6/results/` | — |

## 1. Objective

When the user adds or changes information, **change only what the change affects**. Do not restart the pipeline. Concretely:
- keep session memory;
- detect late-arriving details, corrections and retractions across turns;
- turn them into typed changes;
- plan targeted updates: delta queries, evidence reuse or invalidation, claim revalidation;
- version the answer state, recording exactly which claims and sections changed.

The plan is **CHANGE → IMPACT ANALYSIS → TARGETED UPDATE**, measured against a full-restart baseline.

## 2. Phase 5 Baseline

Phase 5 delivered:
- per-utterance multi-intent decomposition with versioned `IntentSet`s;
- delta retrieval *within* an utterance;
- per-intent ledger lineage;
- intent-aware fusion into a `UnifiedEvidenceSet`.

What it lacked (Phase 5 report §21):
- **Constraint-only follow-ups.** An utterance such as "especially for visitors" fell back to a single Phase 4 query; the constraint was recorded in `dropped`.
- **No cross-turn layers.** There was no constraint registry, no evidence or claim state, no answer state, and no notion of topic boundaries across turns.

Phase 6 keeps all Phase 3–5 components unchanged. With `session.enabled: false` the Phase 5 path is byte-for-byte the old behaviour (a test checks that no Phase 6 events appear). It extends them where the late-detail flow needs it:
- **Tracker:** cross-turn operations, a constraint lifecycle, and a corpus co-occurrence test for "what about Z?".
- **Ledger:** lineage fields.
- **BM25:** a co-occurrence count.
- **Coordinator:** delegates to the session engine.

## 3. Session State Architecture

`SessionState` is the canonical view (doc `docs/session/01`), built from four separate layers (§4).

`SessionStateVersion` records:
- the version id and its parent;
- the trigger: `interpretation` / `evidence` / `answer` / `reset` / `restore`;
- the context-change ids and a summary;
- a deterministic `SessionSnapshot` mapping ids to statuses for frames, needs, constraints, queries, evidence assignments and claims, plus the answer id.

**No version is created for a no-op** ("Okay." leaves the version chain untouched; tested).

Example chain (generated, `research/phase6/results/demo_trace.md`):

```
v6 evidence (u1) -> v7 answer A1 -> v8 interpretation CONSTRAINT_ADDITION(I1) -> v9 evidence via Q4 -> v10 answer A2
```

## 4. Session Memory

| Layer | Holds | Implementation |
|---|---|---|
| A transcript | last `transcript_window` (6) utterances, PII-redacted; older ones as SHA-1 + length | `SessionMemory.transcript` |
| B semantic | topic frames; needs + versions; constraint registry (active / retracted, `replaces`, `retracted_in`); entities | `IntentTracker`, `FrameManager` |
| C retrieval | query ledger with lineage; semantic cache; evidence records + per-need assignments | `QueryLedger`, `SemanticCache`, `EvidenceStore` |
| D answer | claims, links, transitions; answer versions | `ClaimGraph`, `AnswerStateManager` |

**API.** The brief's API exists:
- `initialize_session`, `update_session`, `get_current_state`;
- `get_intent_context`, `get_entity_context`, `get_relevant_evidence`;
- `create_snapshot`, `restore_snapshot`;
- plus `reset_session` and `archive_session`.

**Snapshot and restore cover the whole session.** That includes the tracker's per-utterance state and the engine's own state (the interpretation it last planned from, each need's previous query, id counters). The engine registers them as memory *extras*. Tested properties:
- restore returns the state exactly;
- the restored session keeps producing the same changes;
- reset empties every layer;
- archives keep no raw text.

**Two defects were found while writing these tests, and fixed:**
- restore initially missed the engine state;
- reset and restore replaced the counters dict that the engine holds a reference to.

## 5. Context Change Detection

Three steps (doc 03):

1. **Gate.** The Phase 4 controller decision, plus a late-detail gate. It opens for not-worthy / unstable decisions when:
   - a need is active;
   - the utterance carries a correction, retraction, focus, condition or restriction cue;
   - the utterance has content beyond the cue words.

   A bare "Specifically" mid-stream waits; backchannels never open it.
2. **Interpretation.** The tracker re-decomposes against the session context. Constraint-only utterances, value updates and retractions become **cross-turn operations** on earlier needs. "What about Z?" becomes a **constraint** if the corpus has a chunk containing Z together with the previous need's most specific topic term (max IDF), and otherwise a **parallel need** inheriting only the aspect. "Back to X, …" redirects the follow-up to need X and reactivates its frame.
3. **Detection.** Semantic comparison of need versions: term sets, topic, aspect, constraint ids. Raw strings are never compared, and a test checks that surface changes produce no diff.

**Confidence is computed, never authored.** It is the minimum of the decomposer's need confidence and the constraint scope confidence.

Each turn yields `ContextChange`s with:
- affected / new / superseded needs;
- added / removed constraints;
- affected queries;
- relation, frame action, semantic diffs and the cue.

| Result (fixture, NOT REPORTABLE) | Dev suite (64 turns) | Stress set (31 turns, written after freeze) |
|---|---|---|
| Change-type accuracy | 64/64 = 1.000 | 28/31 = 0.903 |
| … unambiguous turns only | 59/59 = 1.000 | 24/27 = 0.889 |
| Streaming (chunk by chunk, net per turn) | 64/64 = 1.000; provisional changes per turn mean 1.58, p95 3 | not streamed |

## 6. Change Taxonomy

The taxonomy is closed, with 9 types. A test pins the set.

| Type | Meaning | Example (fixture domain) |
|---|---|---|
| `NO_CHANGE` | nothing retrieval-relevant | "Okay." / "Thanks." |
| `REFINEMENT` | same need, terms added | streamed "What are the rules" → "… for ladders" |
| `CONSTRAINT_ADDITION` | new constraint on a need | "Specifically overnight." |
| `CONSTRAINT_REMOVAL` | a constraint retracted or replaced | "Ignore the overnight restriction." |
| `NEW_INTENT` | new need (follow-up or independent) | "What about crates?" / "Now explain how the telescope is recalibrated." |
| `INTENT_REMOVAL` | need vanished from the re-decomposition | mid-stream fragment that turned into a late detail |
| `CORRECTION` | explicit correction superseding a need (`entity_replacement` cue when the topics share no term) | "Sorry, I meant crates instead of ladders." |
| `ENTITY_CHANGE` | same need, topic replaced without an explicit correction | text change with disjoint topics |
| `QUESTION_CHANGE` | same need, aspect / type changed | "rules" → "cost" of the same topic |

**Value updates.** A constraint value update ("night shift" → "day shift") is reported as ADDITION + REMOVAL, and the new constraint records `replaces`.

**Within-utterance revisions.** While streaming, `net_change_types` summarises the provisional changes of a turn:
- changes to needs created in that turn fold into NEW_INTENT;
- fragments created and removed again vanish;
- a constraint added and revised within the turn cancels.

## 7. Delta Planning

`DeltaPlanner.plan(changes)` → `DeltaPlan` (doc 04) with:
- queries to create (`retrieve`), reuse (`reuse_active` = same semantic key as the active query, or `cache_hit` = an earlier completed query had this key) and supersede;
- evidence to retain / revalidate / discard (`evidence@need`);
- claims to revalidate and claims unaffected.

**Only affected needs are planned.** A new need adds one query, and nothing else is touched (CASE 3, CASE 8; tested).

**Streaming integration.**
- The coordinator executes plans under the Phase 5 guards. Budgets are now per (need, utterance), because the session-wide count blocked late details (a streaming defect found and fixed).
- A guarded action is **deferred** and re-dispatched while its need version is current. If a budget prevents it, the turn reports it in `deferred`; it is never silently dropped.

## 8. Delta Query Generation

`DeltaQueryGenerator` derives a need's next query from its previous query plus the semantic delta:
- drop retracted constraints' components;
- append new constraints;
- rebuild the need's own words only if they changed.

A test checks that the delta query has the same analyzed terms as a from-scratch build.

**Lineage.** Every created query records `parent_query_id`, `supersedes`, `derived_from_change_id` and `semantic_key`; cache hits create `reused` ledger records with `reused_from`. A correction links the corrected need's query to the superseded need's query and marks that query stale (`intent_superseded`). This was a gap found in the final manual validation and fixed (§22).

```
Q1 "What are the rules for ladders in the orchard"                       CH1 NEW_INTENT
Q2 "What are the rules for ladders in the orchard overnight"   parent Q1  CH2 CONSTRAINT_ADDITION
Q3 reused_from Q1 (cache_hit)                                  parent Q2  CH3 CONSTRAINT_REMOVAL   -> 0 retrievals
```

**Cache.**
- The key is sha1(index content hash | retrieval-options hash | sorted analyzed terms). It is not the raw string: "Tell me the orchard ladder rules." reuses Q1 (dev S19).
- Invalidation rules: index change, config change, source chunk unavailable. Entity, constraint and question changes simply produce new keys.

## 9. Evidence Lifecycle

Evidence records are never deleted. Applicability is tracked per (evidence, need) as one of:
- ACTIVE / RETAINED (usable);
- REVALIDATION_REQUIRED;
- STALE / SUPERSEDED / INVALID.

Each `EvidenceAssignment` keeps its full rule-named transition history; doc 05 tables the rules. They cover:
- constraint covered / other value of the dimension / general;
- specific to a removed constraint;
- refinement covering, or not covering, the refined topic;
- entity mentioned or replaced;
- correction: superseded, or carried as revalidation-required;
- intent removed;
- confirmed / not confirmed by the delta retrieval;
- source unavailable.

CASE 6 is tested exactly. Q1 → E1, E2, E3; a constraint touches only E2 ("day shift" vs "night shift"):
- E2 → REVALIDATION_REQUIRED;
- E1 and E3 → RETAINED;
- a delta retrieval without E2 → E2 STALE.

Final validation, scenario 8 (bge stack), shows the same pattern on the corpus:

```
Doc_07§3 @I1: ACTIVE(retrieved) -> REVALIDATION_REQUIRED(constraint_dimension_other_value) -> ACTIVE(confirmed_by_delta_retrieval)
```

## 10. Claim Model

Phase 6 claims are **extractive**: verbatim evidence sentences, with an evidence id and character span. SUPPORTS is therefore justified by construction (doc 06); no claim text is invented.

**Fields:**
- `claim_id`, `text`, `intent_id` (+ version, frame);
- `evidence_ids`, `source` span;
- `status` + `status_reason`;
- `confidence` (computed extraction relevance);
- `created_at` / `updated_at`;
- `introduced_in` / `modified_in` answer versions.

**Selection.** IDF-weighted relevance plus an anchor rule: a topic term of the need, or else the max-IDF query terms. A sentence sharing only a ubiquitous word is not a claim about the need.

**Id stability.** Ids are stable per (need lineage, evidence, span). Constraint changes re-evaluate the same claims; corrections create new ones.

## 11. Claim-Evidence Graph

| Relation | Set only when |
|---|---|
| SUPPORTS | verbatim in usable evidence and all active constraints of the need are addressed |
| PARTIALLY_SUPPORTS | verbatim in usable evidence, but an active constraint's terms are missing |
| CONTRADICTS | two selected claims of one need, from different documents, give different numbers for the same unit around a shared content word (potential conflict, flagged both ways, not resolved) |
| UNSUPPORTED | the linked evidence is no longer usable for the need |

**Conflict check.** On the conflict fixture ("below 25 kilometres per hour" vs "below 30"), the dev suite flagged 2 of 2 expected conflicts, with 0 flags without gold. The answer section carries uncertainty `conflict`. Non-numeric contradictions are not detected (§23).

## 12. Targeted Revalidation

A change affects:
- the live claims of the needs it affects or supersedes;
- the live claims of a need whose evidence assignment *for that need* changed.

Claims of other needs on the same chunk are not affected. The first version re-checked them too, and the incremental pipeline then validated more claims than the full restart (124 vs 100 on change turns). The fix is now pinned by a test.

**Statuses:** PENDING_VALIDATION → SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / SUPERSEDED / STALE, each with a reason and change id (doc 07).

**Invariant:** a claim is never SUPPORTED on unusable evidence (tested).

```
CH4 CONSTRAINT_ADDITION I1 +overnight -> C1, C2 PENDING_VALIDATION
Q4 result -> C1 PARTIALLY_SUPPORTED (does_not_address_constraint:K1), C2 SUPPORTED
```

## 13. Answer Versioning

The answer is the structured state of one topic frame (doc 08):
- one `AnswerSection` per active need, holding usable claims, evidence, active constraints and uncertainty (`no_evidence` / `constraint_not_covered` / `conflict`);
- `AnswerVersion` with `answer_id`, `version`, `supersedes_answer_id`, `session_version`, `claim_ids`, `evidence_ids`, `citations`, `frame_slots`, `diff`, `change_summary`, `delta_queries`, `full_rerun`;
- `text` stays `""` (Phase 7 renders).

**API:** `create_answer_state`, `update_answer_state` (commits only on a non-empty diff), `get_current_answer_state`, `compare_answer_versions`.

**Diff:**
- claims kept / modified (from → to status) / added / retracted;
- citations and evidence added or removed;
- sections added / removed / changed / unchanged;
- uncertainty introduced or resolved.

## 14. Incremental Answer Refinement

Sections whose status is new or changed carry `needs_regeneration = true`. Unchanged sections are reused verbatim: **minimal regeneration**.

The brief's 7 refinement cases (`answer_refinement.json`; fixture, NOT REPORTABLE) are shown below. Columns are new retrievals / affected / revalidated / unchanged claims, then the answer change.

| Case | Incremental | Full restart |
|---|---|---|
| 1 no meaningful change ("Okay.") | 0 / 0 / 0 / 2 kept; **no new version** | 0 retrievals; nothing rebuilt |
| 2 new constraint ("Specifically overnight.") | 1 / 2 / 2 / 1; A2: 1 kept, 1 modified SUPPORTED → PARTIALLY_SUPPORTED | 1 retrieval, 2 validations; new answer, 2 added |
| 3 new intent (unrelated topic) | 1 / 0 / 2 / 0; new frame answer | same work |
| 4 entity correction | 1 / 2 / 4 / 0; 2 retracted, 4 added, same section lineage | 1 retrieval, 4 validations, 4 added |
| 5 intent refinement ("In the orchard.") | 1 / 2 / 2 / 2; 2 kept | 1 retrieval; 2 added |
| 6 follow-up ("What about crates?") | 1 / 0 / 4 / 2; S-I1 unchanged, only S-I2 re-rendered | **2 retrievals, 6 validations, both sections re-rendered** |
| 7 evidence contradiction | 1 / 0 / 3 / 0; uncertainty `I2:conflict` introduced | same; conflict also flagged |

Over the 64 dev turns:
- **Incremental:** answers re-rendered 62 of 65 committed sections and kept 20 claims unchanged in diffs.
- **Full restart:** re-renders every section of every answer.

## 15. Context Compression

`ContextCompressor` (doc 09) represents the conversation as:
- per utterance: hash + length + derived needs and constraints with spans;
- verbatim text only inside the window.

The tracker drops old utterances' text too. `RelevantContextSelector` builds the package a generator may see, in three modes:
- `none`;
- `full_transcript`;
- `structured`: the active frame's needs, active constraints, entities, usable claims and evidence ids.

**Exclusions are counted by reason.** **Provenance:** every item carries utterance spans or evidence spans, and a test checks claims are verbatim at their spans. Sizes are characters, plus tokens when a tokenizer is supplied.

## 16. Context Isolation

**Topic frames.**
- A new need joins the active frame only when related to it (relation, follow-up decision, correction, cross-turn change, or shared topic terms).
- Otherwise the frame goes dormant and a new one opens.
- Dormant frames are reactivated only when their topic is named ("Back to the ladders, …"; tested CASE 9).

**Constraint scope.** Constraints apply only to the needs in their `applies_to`. A parallel follow-up inherits the previous need's aspect, not its constraints (tested).

**Measured:**
- 0 isolation leaks (terms the gold forbids in new queries) on the dev suite and on the stress set;
- in the memory ablation, full-transcript memory produced 8 queries carrying earlier topics' words; structured memory produced 0 (§19).

**Memory safety (doc 10).**
- Redaction runs at ingestion, so interpretation, queries, ledger, snapshots and archives never contain the redacted PII (tested with an e-mail address).
- Archives keep hashes and counts only.

## 17. Delta Retrieval Benchmark

Benchmark A (full re-retrieval = full-restart pipeline) vs B (delta), over the 32 dev follow-up turns that changed something (`delta_retrieval.json`; fixture, NOT REPORTABLE):

| Metric | B delta (incremental) | A full re-retrieval |
|---|---|---|
| Retrieval calls | **30** (+3 semantic-cache hits) | 36 |
| Evidence assignments kept usable without retrieval | **93** | 0 |
| New evidence records fetched | **45** | 161 |
| Claim validations | 88 | 91 |
| Retrieval-stage ms, p50 / p95 (per turn, median of 7 reps) | 4.82 / 6.38 | 5.21 / 6.99 |
| Turn total ms, p50 / p95 / mean | 6.35 / 8.65 / 6.28 | 6.85 / 9.06 / 6.91 |

Example (S01, constraint added):
- **Delta:** 1 retrieval, 5 assignments retained, 1 new record, 2 claims re-validated, answer diff = 1 kept + 1 modified.
- **Restart:** 1 retrieval, 5 records re-fetched, a fresh answer.

On single-need turns both make one retrieval. The delta advantage there is retained evidence, a targeted diff and cache hits for "undo". On multi-need frames the restart re-retrieves every need.

**Retrieval-decision correctness against gold** ("required / reuse / none"):
- delta: 64/64;
- full restart: 61/64 (3 unnecessary retrievals, all where the gold expects evidence reuse).

## 18. Full Restart Baseline

`FullRestartPipeline` (brief §41). Every turn builds a fresh session over the whole conversation:
- full intent analysis of every turn;
- full query generation and retrieval of every active need **of the current frame**;
- no cache and no retention;
- every claim re-extracted and re-validated;
- a fresh `initial` answer.

It uses the same gate. Equivalence check: on a 6-turn session, the queries it serves equal the incremental pipeline's (test). Quality is identical on the dev suite (change types 1.000, need queries 0.984). The difference is work.

**Conservative choice.** The baseline only re-retrieves the current frame. A restart that re-retrieved every need of the session would look worse; that variant was not measured.

**Session-length scaling** (`scaling.json`; one 16-turn fixture workload with a frame growing to 3 needs, late details, retractions and backchannels; median of 7 reps per turn):

| | Incremental | Full restart |
|---|---|---|
| Retrieval calls (16 turns) | **9** | 25 |
| Sum of per-turn median latency | **57.8 ms** | 159.1 ms |
| Latency saving | **63.7 %** (range over 3 runs: 63.2–63.7 %) | — |
| Turn 15 "Forget the sunset part." | 2.44 ms (cache hit, 0 retrievals) | 18.78 ms (3 retrievals) |
| Turn 16 "What about the crate count?" | 5.57 ms (1 retrieval) | 19.33 ms (3 retrievals) |
| Backchannel turns ("Okay.", "Right.") | 0.06–0.10 ms | 1.3–6.1 ms (re-analysis of the whole conversation) |

The restart cost grows with conversation length and frame size. The incremental cost depends only on what changed.

## 19. Ablation Results

Memory ablation (brief §40; `ablation.json`; 64 dev turns; fixture, NOT REPORTABLE). Tokens are bge-small WordPiece **proxies**.

| Arm | Need-query correctness | Context leaks | Retrieval calls | Wall ms / turn p50 (mean) | Conversation context tokens / turn p50 (mean) | Gold-evidence recall (fixture) |
|---|---|---|---|---|---|---|
| A no session memory | 0.594 | 0 | 49 | 5.20 (4.43) | 7 (7.6) | 0.579 |
| B full transcript memory | 0.938 | **8** | 81 | 5.52 (6.60) | 13 (14.0) | 0.895 |
| C structured session memory | **0.984** | 0 | 59 | 5.63 (5.51) | 13 (11.8) | **1.000** |

- **A** cannot apply any late detail.
- **B** recovers most needs, but leaks finished topics into new queries and re-retrieves everything.
- **C** is the most accurate and has no leaks.

**Context size.** On these 2–5-turn sessions C's conversation context is about the size of B's. B grows with session length by construction; this was not measured beyond these sessions.

C additionally carries a retrieval context (claims + evidence ids) of 69.5 tokens per turn (p50), which a generator would consume.

## 20. Latency Results

Stage latencies (`latency.json`) on the 32 follow-up turns with a change. Each figure is the per-turn median over 7 repetitions, then p50 / p95 across turns, in ms. Brief §43 names in brackets.

| Stage | Incremental | Full restart |
|---|---|---|
| interpretation (current turn) | 0.21 / 0.33 | 0.19 / 0.31 |
| change_detection [change_detection_latency] | 0.10 / 0.16 | 0.09 / 0.16 |
| delta_planning [delta_planning_latency] | **1.03 / 1.69** | 0.08 / 0.13 |
| retrieval [delta_retrieval_latency] | 4.76 / 6.14 | 4.87 / 7.03 |
| evidence update | 0.05 / 0.08 | 0.03 / 0.05 |
| claim_revalidation [claim_revalidation_latency] | 1.90 / 3.13 | 1.73 / 2.91 |
| answer_update [answer_update_latency] | 0.06 / 0.07 | 0.05 / 0.06 |
| **turn total** [incremental_latency / full_restart_latency] | **6.28 / 8.06** | **6.59 / 8.81** |

**latency_saving**, computed only from these measurements:
- **Changed turns:** sum of per-turn medians 199.6 ms (incremental) vs 217.9 ms (restart), a **8.4 %** saving.
  - Per-turn saving: p50 −0.9 %, mean 6.2 %, p95 64.8 %.
  - The median turn costs about the same in both pipelines. The saving comes from multi-need turns and cache hits.
- **All 64 turns:** 5.2 %.
- **16-turn session:** 63.7 % (§18).

**What incremental pays more for.** Delta planning costs about 1 ms (analyzing evidence text for the validity rules the first time it is seen). The restart has no evidence to check. This was not optimized (brief: do not optimize prematurely).

**Streaming (virtual) integration.**
- 28 sessions, 64 turns, 90 queries, of which 87 were issued provisionally while the user was still speaking.
- Exact replay: 28/28 sessions.
- A realtime trace at 4× speed (161 events) replays with identical behaviour (`replay_check.json`).

## 21. Compute/Evidence Reuse

Totals over the 64 dev turns (`compute.json`):

| | Incremental | Full restart |
|---|---|---|
| Retrieval calls | 59 | 65 (**6 avoided, 9.2 %**) |
| Queries reused (semantic cache) | 3 | 0 |
| Evidence assignments retained without retrieval | **93** | 0 |
| Claim validations | 158 | 161 |
| Claims carried unchanged in answer diffs | 20 | 0 |
| Answer sections to re-render / committed | 62 / 65 | 65 / 65 |
| Re-render input tokens (claim text of sections to re-render; bge proxy) | 2,326 | 2,474 (**148 avoided, 6.0 %**) |

**Evidence reuse rate on changed turns** = retained (evidence, need) assignments / (retained assignments + newly fetched evidence records) = 93 / (93 + 45) = **67 %**. The restart retained nothing and re-fetched 161 records.

The short dev sessions understate the effect. The 16-turn workload avoided 16 of 25 retrievals (§18).

**Tokens.** No generation tokens were measured, because no generator exists yet. The re-render figure is an input-size proxy, not a generation saving.

## 22. Failure Cases

**Stress set (written after the code freeze, run once, not fixed):**

| Turn | Expected | Got |
|---|---|---|
| "Make that the day shift instead." (after "Only for the night shift.") | constraint update | NEW_INTENT "make day shift instead", plus a retrieval |
| "Not the lens, the wick." | correction | NEW_INTENT "not len wick" |
| "Can you drop the overnight condition?" | constraint removal, evidence reuse | NEW_INTENT plus an unnecessary retrieval: the request head "can you" wins over the retraction word |
| "Is that also true at night?" (ambiguous) | anaphoric constraint on the ladders need | NEW_INTENT "true night" |
| "Scratch that, I'm asking about crates." | correction; detected correctly | the query carries noise words: "rule m ask crate" |

**Ambiguous reading taken by the corpus rule.** "And what about the tool shed?" after a ladders question (scaling workload, turn 6) became a *constraint* on the ladders need. The corpus says ladders "must be returned to the tool shed". A user may have meant a new question about the shed.

**Defects found during Phase 6 development and fixed** (disclosed because they shaped the design):
1. A correction opened a new topic frame.
2. A topic-only evidence match produced no claims: two query terms tie on IDF, so the tie-break picked the wrong anchor term. Fixed with topic anchors.
3. A parallel follow-up lost the aspect inherited from an earlier utterance.
4. In streaming, a session-wide per-intent query budget blocked late-detail queries, and a mid-stream fragment ("Specifically") opened the gate and retrieved junk.
5. `reuse_active` did not re-confirm evidence that was pending revalidation.
6. Fragment frames stayed active.
7. Claim revalidation over-targeted other needs' claims (§12).
8. Snapshot / restore and counter aliasing (§4).
9. The correction query had no lineage to the superseded need's query, and the synchronous pipeline did not mark that query stale. Found during the final manual validation; fixed and tested.

**Operational.**
- One intermittent native abort (`libc++abi … recursive_mutex lock failed`) was seen at interpreter exit in 1 of about 10 benchmark runs, after all results were written. It was not reproduced in later runs and is likely ONNX Runtime / thread-pool teardown.
- **Repository:** the `.gitignore` patterns `models/` and `corpus/` also matched `src/streamrag/models/`, `src/streamrag/corpus/` and `tests/fixtures/corpus/`. Those directories are **missing from the Phase 1–5 commit `8ce6502`**. The patterns are now anchored to the repo root; the next commit must include the three directories.

## 23. Known Limitations

1. **No official corpus or official multi-turn cases.** All results are fixture-domain dev results. The dev suite is optimistic (see "Read this first"); the stress set is small (14 sessions).
2. **Rule- and lexicon-driven cross-turn operations.** Phrasings outside the lexicon fail (§22). There is no LLM fallback, because no backend is configured.
3. **Anaphoric late details** ("is that also true at night?") are not resolved into constraints.
4. **Extractive claims only.** Relevance is lexical (IDF); support status is term presence, not entailment; PARTIALLY_SUPPORTED is a term-coverage test.
5. **Contradiction detection is numeric only.** Polarity conflicts are not detected.
6. **Topic frames use topic-term overlap.** Unrelated questions sharing a generic topic word may share a frame.
7. **Delta scope is the whole corpus.** Document-scoped delta retrieval (`session_docs_first`) is not implemented.
8. **Fusion is not included in the synchronous comparison.** Phase 5 fusion runs in streaming on the turn's needs.
9. **Small absolute savings on a tiny corpus.** Retrieval costs about 5 ms on the fixture corpus. The full-restart baseline is conservative (current frame only).
10. **Pattern-based PII redaction.** Names and addresses are not detected. Phase 4 telemetry traces still contain the raw input (needed for replay).
11. **Restore granularity.** Restore is between turns; the mid-utterance decomposition state is not persisted. There is no long-term memory (out of scope).
12. **Streaming cost.** Streaming issues provisional queries while the user speaks (87 of 90 in the dev run). This is bounded by the per-need budgets, and it costs retrievals.

## 24. Phase 7 Requirements

**Ready for Phase 7:**
- `AnswerVersion` sections with `needs_regeneration`, claim ids, evidence ids, citations, constraints and uncertainty items per need.
- Extractive claims with exact evidence spans: ready-made grounding units and citation targets.
- `RelevantContextSelector` structured packages: generator context with provenance and counted exclusions.
- Answer diffs (kept / modified / added / retracted) that tell the renderer which text can be reused verbatim.
- Telemetry and replay covering the whole session evolution.

**Prerequisites:**
1. **LLM backend decision** (ADR-007 Q2: hosted / local / extractive fallback). This also decides whether an LLM-assisted change classifier can back up the rule-based one for the §22 phrasings, behind deterministic validation.
2. **Generation contract.** Render only sections with `needs_regeneration`; keep unchanged sections verbatim. Every generated sentence must cite claim evidence (claim ids → citations). Express PARTIALLY_SUPPORTED claims and the `constraint_not_covered` / `conflict` / `no_evidence` uncertainty explicitly, without asserting beyond the claims.
3. **Citation validation** against claim source spans (ADR-008 L0–L2), plus a grounding metric (G4 ≥ 85 %). This needs gold answers and evidence on the official corpus.
4. **Token accounting with the generator's tokenizer.** This replaces the bge proxy; only then can token savings of minimal regeneration be claimed.
5. **The official Theme 4 corpus and official multi-turn evaluation cases.** These are needed to report G5 session refinement and G4 grounding; still blocked.
6. **Commit the working tree.** The Phase 6 code plus the three previously git-ignored directories (`src/streamrag/models/`, `src/streamrag/corpus/`, `tests/fixtures/corpus/`), so a clean clone runs.
