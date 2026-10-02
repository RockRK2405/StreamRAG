# Samsung PRISM Theme 4
# Phase 5 — Multi-Intent RAG

| | |
|---|---|
| Date | 2026-10-02 |
| Status | Phase 5 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** (unchanged) |
| Code | `src/streamrag/{intents,multi_retrieval,fusion}` + `bench/multi_intent.py` + streaming integration |
| Tests | **320 passing, 0 skipped** (213 from Phases 3–4; 107 new: `tests/intents` 71, `tests/multi_retrieval` 18, `tests/fusion` 13, isolation scan +3 packages, contract samples +2). Lint / type checking not configured; `compileall` and the import of all 85 modules pass. |
| Data | **DEV SUITE:** 53 cases / 60 utterances / 94 gold needs, team-authored over the *fictional test-fixture corpus* (`eval/dev_multi_intent/`, `is_fixture: true`) |
| Output of the phase | `UnifiedEvidenceSet`. No answer generation, claim extraction, citation validation, late-detail refinement, UI or ASR. |

**Read this first: what the numbers mean.**

1. **Not official.** Every number comes from the dev suite on the fixture domain (3 documents, 14 chunks). All outputs carry `REPORTABLE: false`.
2. **Not held-out.** The same team that wrote the decomposition rules wrote the suite. Two runs are reported:
   - a **blind run** (frozen code, first contact with the suite);
   - a **post-fix run**, after fixing 7 defects found by the blind run and the streaming evaluation (§20).

   The post-fix numbers are optimistic.
3. **Retrieval quality on 14 chunks says little.** Any reasonable query finds its section within the top 5. Retrieval and fusion results are plumbing and sanity checks, measured at a small k (3), where differences are at least visible.
4. **Latency** is measured on the dev machine (Apple M5 Pro, CPU only). The conditions are stated per number: back-to-back, or after a 300 ms idle gap as in live speech.

**Quality gate (brief §45):**

| Gate | Status | Where |
|---|---|---|
| Intent model, IntentSet | ✅ | §3 |
| Multi-intent decomposition works | ✅ dev suite | §4, §17 |
| Over-decomposition controlled | ✅ dev suite (category I: F1 1.0; brief TESTS 1, 8, 9) | §4, §17 |
| Under-decomposition controlled | ✅ dev suite (category J: F1 1.0; B3 miss documented) | §4, §17, §20 |
| Constraints represented (global / local) | ✅ | §5 |
| Relationships where needed | ✅ (recall 0.50: misses documented) | §6, §20 |
| Queries per intent, keeping relevant context | ✅ (context retention 0.79) | §7 |
| Parallel retrieval; Phase 3 foundation reused | ✅ | §8 |
| Evidence provenance with intent / query lineage | ✅ | §9 |
| Duplicate evidence handled | ✅ | §10 |
| Evidence fusion; intent-aware ranking | ✅ | §11, §12 |
| Incremental updates, versioning, supersession, delta retrieval | ✅ | §13–§15 |
| Telemetry (intent / query ids, 9 new event types) | ✅ | §9, ADR-015 |
| Benchmark harness, ablations | ✅ | §16–§18 |
| Tests pass | ✅ 320 / 320 | Header |
| No fabricated results | ✅ every number is from `research/phase5/results/` (dev suite, `REPORTABLE: false`) | — |
| No answer generation implemented | ✅ (`TURN_COMPLETED.answer = null`) | — |
| Official data | ❌ blocked (external) | §21 |

---

## 1. Objective

Turn Phase 4's *single active query per utterance* into **multiple related intents**, preserving streaming behaviour, query lineage, session state, retrieval provenance, latency telemetry and evidence traceability:

```
Transcript → Retrieval Controller (gate) → Intent Analyzer/Decomposer → IntentSet (versioned)
  → per-intent queries → parallel retrieval (Phase 3) → candidate evidence → deduplication
  → evidence fusion → cross-intent reranking (optional) → UnifiedEvidenceSet
```

While the user is still speaking, the set of needs must update incrementally:
- only **new or changed** needs are retrieved (delta retrieval);
- valid evidence is reused;
- corrected needs are superseded, with their provenance kept.

---

## 2. Existing Streaming Foundation

**Validation before any change:**

| Check | Result |
|---|---|
| Full test suite | 213/213 passing |
| TranscriptChunkManager, RetrievalController, QueryLedger, async executors, Evidence model, hybrid retrieval, telemetry, replay | Verified against their APIs and tests. CLI stream and replay worked (virtual exact, realtime behaviour-identical). |
| Phase 3 retrieval API | `retrieve` / `retrieve_batch` unchanged. Thread safety had been verified in Phase 4. |

**Phase 4 defects or gaps fixed for Phase 5:**

| # | Issue | Fix |
|---|---|---|
| 1 | `configs/default.yaml` still justified the 3,000 ms endpoint timeout with the superseded "≈ 2.65 s" estimate | Corrected to the measured 2.10–2.13 s (as in the Phase 4 report and ADR-014) |
| 2 | `QueryLedger` supported one active query per utterance (Phase 4 prerequisite #2) | Supersession is now scoped to (utterance, intent) when `intent_id` is set. Added `for_intent`, `latest_completed`, `mark_intent_stale` and `lineage_tree`. Phase 4 behaviour with `intent_id=None` is unchanged; its tests pass. |
| 3 | The telemetry envelope had no intent or query fields | Optional `intent_id` / `query_id` added to every `TelemetryEvent`. Phase 4 query events now fill `query_id` too. |

**Phase 4 prerequisites addressed:**

| # | Prerequisite | Status |
|---|---|---|
| 1 | Segment-level state | Clause segmentation inside the decomposer |
| 2 | Per-intent ledger | Done (row 2 above) |
| 3 | Multi-intent gold | Fixture dev suite only; official cases are still needed |
| 4 | Per-intent fusion | Done |
| 5 | Slow-retrieval finding | Partly. Unchanged intents are never re-retrieved, and reranking runs only on the final fusion. A refined final query is still re-retrieved; see §22. |
| 6 | External blockers | Unchanged |

---

## 3. Intent Model

`models/intents.py`; full field list in `docs/multi_intent/01`; JSON schemas exported (24 total).

**Definition used throughout:** an intent is one retrieval-relevant information need. Linguistic components ("tell", "about", "information") and constraints ("especially during a storm") are **not** intents.

| Object | Key fields |
|---|---|
| `Intent` | Session-scoped `I<n>`, `version`, `text`, `resolved_text`, `components[]` (verbatim pieces with `SourceSpan`), `intent_type`, `entities`, `topic` + `topic_span`, `aspect`, `constraint_ids`, `inherited_context[]` (reason + source span), `unresolved_references`, `order` (mention order), `priority` / `priority_score`, `confidence` + `confidence_signals`, `status`, `provenance`, lineage (`supersedes`, `superseded_by`, `lineage_root`), timestamps, `query_ids` |
| `IntentSet` | `original_text`, active `intents`, `global_constraints`, `local_constraints`, `relationships`, `decomposition_confidence`, `superseded`, `merged[]`, `dropped[]` (each with a reason) |
| `IntentSetDelta` | `added`, `modified` (with `changed`: text / constraints / context), `removed`, `superseded`, `constraints_added` / `_removed`, `affected_intents` |

**Intent types (8):** REQUIREMENT, TIMELINE, PROCEDURAL, EXCEPTION, COMPARISON, DEFINITION, FACTUAL, OTHER.
- Types are assigned from cue words in the need's own words, in lexicon order.
- They inject nothing into queries; their cue words are already part of the need.

**Confidence is computed, not authored:**

```
confidence = 0.35*explicit + 0.30*specificity + 0.20*separation + 0.15*resolution
```

A test checks every reported confidence against its signals.

---

## 4. Intent Decomposition

`intents/decomposer.py`; `docs/multi_intent/02`. Rule-first, deterministic, generic English lexicon (`configs/intent_lexicon.yaml`; scanned by the anti-hardcoding test).

1. **Tokens with punctuation and character spans.** Fillers are masked, except inside correction phrases ("i mean").
2. **Clause segmentation:**
   - hard punctuation;
   - ", " followed by a clause opener;
   - "and" followed by a question word, request head or correction;
   - addition markers ("and also", "plus", "as well as");
   - question restarts without punctuation ("how …"; "what/when/where/why" + auxiliary);
   - focus / condition / correction markers.

   A question directly after a marker stays inside that clause ("specifically how high …", "no wait how often …").
3. **Clause roles:** REQUEST, CONSTRAINT, CORRECTION, CONTEXT, IGNORE. Only REQUEST clauses produce intents. A facet noun phrase without a first-person subject counts as an elliptical request ("the requirements for ladders, how long …").
4. **Guarded coordination split:**
   - list commas and "and" split a request;
   - parts are merged back when they have no substantive content (generic nouns), when "a and b" occurs **in the indexed corpus** (fixed phrase), or for comparisons and "between …";
   - "or" never splits;
   - in ASR-style lists without commas, an article after a content word starts a new item, unless the item contains an auxiliary (a relative clause).
5. Constraints (§5), corrections (§13), anaphora and ellipsis (§6), dedup and REFINEMENT, type, entities, confidence, priority, and the `max_intents` budget (dropped with a reason).
6. **Validation** (`intents/validation.py`):
   - empty output, duplicate ids, duplicate intents, identical queries, unsupported types, dangling references;
   - **every source span must equal the transcript substring.**

   Rule output passing validation is a tested invariant.
7. **Fallback.** No request clause but retrieval-worthy content → one intent from the Phase 4 QueryBuilder (Phase 2 K4: information requests are never left unretrieved).

**Optional LLM check** (`intents/llm_check.py`, spec §9.6):
- gated (G-a…G-d), at most once per utterance;
- strict JSON schema, one retry, fallback to the rules;
- every proposed need must quote a verbatim `source_text`; ungrounded or unsupported needs are rejected; rule intents are never removed.

**Status: off by default; no backend is configured** (ADR-007, decision Q2 open). It is exercised only with scripted fake backends in 7 tests and **has not been measured with a model.**

---

## 5. Constraint Handling

`docs/multi_intent/03`. Constraints are first-class objects:
- `kind`: focus / condition / restriction;
- `marker`, `text`, `source_span`;
- `scope`: global / local; `applies_to`; `scope_reason`; `scope_confidence`.

| Situation (first rule wins) | Scope | Confidence |
|---|---|---|
| Trailing PP after a coordination whose conjuncts have their own topics ("the dome and the telescope **for visitors from abroad**") | global | 1.0 |
| Explicit "for both / all of them" | global | 1.0 |
| Focus + question ("the harvest limits, **specifically how high …**") | local, previous need | 0.8 |
| Focus sharing a term with exactly one need ("requirements for ladders, **especially the safety requirement**") | local | 0.9 |
| Restriction or condition after ≥ 2 needs ("… and also the lens **especially during a storm**") | global | **0.6 (ambiguous)** |
| Fronted restriction ("**For visitors,** …") | global | 1.0 |
| Bare focus noun phrase | local, nearest preceding need | 0.6 |

- **Ambiguity has consequences:** it lowers `decomposition_confidence` and is the G-c gating signal for the optional LLM check.
- **Shared topics are not constraints:** "the rules and the schedule **for pruning**" is a shared *topic*, inherited as context.

**Results:** §17 (dev suite).

---

## 6. Intent Relationships

| Type | Produced when | Evidence string |
|---|---|---|
| `DEPENDENT` | A pronoun or facet-only need resolved to an earlier need in the same utterance | `pronoun 'it'`, `aspect without topic` |
| `FOLLOW_UP` | The same, resolved to a need of an earlier utterance | Same |
| `REFINEMENT` | A need's own terms are a strict superset of another's ("the tool shed" ⊂ "the tool shed's register process") | `narrower need on the same terms` |
| `CONSTRAINT_OF` | Represented by `Constraint.applies_to` | — |
| `COMPARISON_WITH` | In the vocabulary but not produced: a comparison is one intent | — |

`INDEPENDENT` is the default (no edge).

**Context carry-over:**
- **Pronouns:** replaced by the antecedent's topic words. "there" is never guessed; it goes to `unresolved_references`.
- **Facet-only follow-ups** ("and what about the application process?") inherit the previous need's topic.
- **No blind copying:** a follow-up naming its own topic inherits nothing (tested).
- **Traceability:** inherited words keep the span of the utterance where they were said.

---

## 7. Query Generation

`intents/query_builder.py`; `docs/multi_intent/04`.

```
query = resolved need words + inherited context not yet present + constraint text in scope not yet present
```

- **Verbatim speech only:** every component is verbatim speech with a source span, so nothing is invented. Tested: every query word occurs in the session's transcripts, and every component span equals its transcript substring.
- **What survives:** entities, numbers and negation (tested).
- **One string per intent:** it feeds the unchanged Phase 3 hybrid retrieval.
- **Example:** for "For visitors, what are the rules and the schedule for the telescope?" the queries are:
  - I1 `what are the rules for the telescope For visitors`;
  - I2 `the schedule for the telescope For visitors`.

**Context retention on the dev suite** (gold "required query terms", 24 cases): **0.792 post-fix** (blind: 0.750). The misses are in §20.

---

## 8. Parallel Retrieval

`multi_retrieval/retriever.py` (offline) and `multi_retrieval/coordinator.py` (streaming, Phase 4 executor with `max_concurrent_retrievals` = 3).

- **One retrieval implementation.** All intents go through Phase 3 `RetrievalService`. A test asserts that sequential, parallel and batched dispatch return exactly `RetrievalService.retrieve`'s results.
- **Budget:**
  - `max_intents` 4;
  - `max_queries_per_utterance` 10;
  - `max_queries_per_intent` 3;
  - `max_candidates_per_intent` 5;
  - provisional per-intent cooldown 400 ms.

  Every skip emits `RETRIEVAL_SKIPPED` with its reason.
- **Prioritisation:** priority = 0.35·confidence + 0.25·specificity + 0.15·explicitness + 0.10·constraints + 0.10·novelty + 0.05·dependents. Mention order is kept separately for answer organisation.

**Dispatch latency** (bge-small ONNX, fixture index, 30 repetitions per cell, p50 ms; `results/dispatch_latency.json`):

| Queries | Condition | Sequential | Parallel (3 threads) | Batched (one embedding call) |
|---|---|---|---|---|
| 1 | back-to-back | 2.69 | 2.66 | 2.49 |
| 2 | back-to-back | 5.44 | 4.47 | **4.29** |
| 3 | back-to-back | 8.25 | 6.99 | **5.24** |
| 4 | back-to-back | 11.84 | 9.10 | **6.56** |
| 1 | after 300 ms idle | 10.54 | 10.93 | 11.66 |
| 2 | after 300 ms idle | 17.11 | **13.79** | 14.23 |
| 3 | after 300 ms idle | 22.33 | 18.66 | **15.94** |
| 4 | after 300 ms idle | 27.09 | 20.48 | **18.36** |

**Findings:**
- **Parallel threads overlap but contend.** Max concurrency is 3, yet the speed-up over sequential is only 1.2–1.3× at 3–4 queries. The critical path grows with concurrency (2.9 → 6.6 ms back-to-back) because the ONNX intra-op threads compete for the same cores.
- **Batching wins for ≥ 3 same-tick queries,** by 10–28% over parallel.
- **The idle penalty from Phase 4 (§16 there) applies to every mode:** about 10 ms for a single query after an idle gap.
- **Memory:** the Python-heap peak grows by about 24 KiB per query in every mode. ONNX native allocations are not visible to `tracemalloc`, and RSS growth after warm-up was 0 KiB.

**Decision: parallel stays the streaming default.** In the streaming dev run, **114 of 124** same-tick batches held a single query; 6 held two and 4 held three. Needs arrive one at a time while the user speaks, so batching would rarely apply. Parallel jobs also complete independently. Batching is a candidate for end-of-utterance batches of ≥ 3 (§22).

---

## 9. Evidence Provenance

**Retrievals:** every retrieval event carries `session_id`, `utterance_id`, `intent_id` and `query_id` in the envelope.

**Fused evidence:** every `FusedEvidence` lists:
- `hits[]`: `intent_id`, `query_id`, `retrieval_method`, rank, score, BM25/dense ranks, RRF/rerank scores, `stale`;
- `supporting_intents`;
- `supporting_queries`;
- `selected_for`.

**Lineage:**
- `QueryLedger.lineage_tree(utterance)` returns `intent → query versions → evidence ids`. It is included in `TURN_COMPLETED.lineage`.
- An integration test walks every unified item back through its hit's query to the ledger's evidence ids and the owning intent.

Excerpt from `research/phase5/traces/B1.txt` (three needs; fixture index):

```
[00:07.644] u1 INTENTS    v2 +I2,I3 ~I1 | I1[REQUIREMENT] 'What are the rules for ladders in the orchard';
                          I2[PROCEDURAL] 'how are the wicks trimmed'; I3[TIMELINE] 'when is the telescope recalibrated'
[00:07.644] u1 QUERY      Q2 for I2 v1 "how are the wicks trimmed"
[00:07.644] u1 QUERY      Q3 for I3 v1 "when is the telescope recalibrated"
[00:07.644] u1 QUERY      Q4 for I1 v2 "What are the rules for ladders in the orchard" supersedes Q1 (refines)
[00:07.644] u1 MULTI      B2 start Q2,Q3,Q4 (parallel)
[00:07.654] u1 RETRIEVAL  DONE  Q4/I1 status=ok 10.0ms top: Doc_07 §2.1, Doc_07 §2.2, Doc_07 §1
```

---

## 10. Evidence Deduplication

| Duplicate kind | Handling |
|---|---|
| Same chunk retrieved by several intents | **One** `FusedEvidence` with `supporting_intents: [I1, I2]` and `supporting_queries: [Q1, Q2]` |
| Phase 3 near-duplicate *alternate* of a kept chunk, retrieved by another intent | Collapses into the kept chunk; `DUPLICATES` relation |

**Measured duplicate-item rate in the unified set** (dev suite, k = 8):
- **0.0** for every deduplicating strategy;
- **0.125** for naive concatenation (one item in eight repeats).

**Per set:** `EVIDENCE_DEDUPLICATED` reports `input_hits`, `unique_chunks`, `cross_intent_duplicates` and `near_duplicates_merged`.

---

## 11. Evidence Fusion

`fusion/engine.py`; `docs/multi_intent/06`.

**Strategies compared** (brief §23). Metric: **Intent Coverage@k**, the share of gold needs with ≥ 1 gold citation in the unified top-k (59 utterances with retrieval; overall recall equals coverage on this suite):

| Strategy | k = 3 | k = 5 | k = 8 | Duplicates @8 | Fusion time p50 |
|---|---|---|---|---|---|
| A. Concatenation | 0.910 | 0.935 | 0.977 | 0.125 | 0.10 ms |
| B. Global score ranking | **0.983** | 0.983 | 0.983 | 0 | 0.07 ms |
| C. RRF across intents | 0.946 | 0.977 | 0.983 | 0 | 0.06 ms |
| D. **Intent-aware** (floor 2 per intent, then fill; section cap 2) | **0.983** | 0.983 | 0.983 | 0 | 0.06 ms |

**What the comparison shows:**
- **Concatenation starves later intents:** its first intent takes the budget.
- **RRF rewards shared chunks** over an intent's own best evidence. Both effects are measurable at k = 3.
- **Intent-aware and global-score tie on this suite.** The fixture corpus is too small to separate them.

**Decision: intent-aware is the default** (ADR-015, = ADR-009 quota round-robin with a global budget). The choice rests on its construction guarantee, not on a measured win:
- it is the only strategy that guarantees each need a share by construction;
- a unit test shows a strong intent taking the whole budget under concatenation but not under intent-aware;
- global-score ranking compares scores that are not comparable across queries.

**Conflicts.** A conservative numeric check flags `potential` conflicts when two chunks from different documents state different values for the same unit around a shared content word.
- Unit tests trigger it, and one false-positive source (the unit word itself counted as shared context) was found and fixed.
- No conflict was flagged on the fixture corpus.
- Non-numeric contradictions are **not** detected (§21).

---

## 12. Intent-Aware Reranking

`docs/multi_intent/07`. Pluggable; scores candidates against **each intent's query**, never the whole utterance. In streaming it runs only on the final fusion.

| Mode | Coverage@3 | Coverage@5 | Rerank time p50 / p95 (profile run, 30 compound utterances) |
|---|---|---|---|
| none (intent-aware fusion) | 0.983 | 0.983 | 0 |
| `intent_ce` (cross-encoder, each intent's own candidates) | 0.983 | 0.983 | 22.3 / 28.4 ms |
| `cross_intent_dense` (cosine of each candidate × each intent) | 0.983 | 0.983 | 3.5 / 5.8 ms |
| `cross_intent_ce` (cross-encoder, each candidate × each intent) | 0.977 | 0.983 | 30.8 / 54.7 ms |

**Result:** on this suite, reranking adds cost and no measurable coverage. `cross_intent_ce` loses one item at k = 3.

**Default:** `rerank: none`. Reranking must be re-evaluated on the official corpus, where a 14-chunk ceiling no longer hides differences (Phase 3 Exp 3 is still pending).

**Failure behaviour:** a missing model or an error falls back to retrieval order with a warning (tested).

---

## 13. Incremental Intent Updates

`multi_retrieval/coordinator.py`; `docs/multi_intent/08`.

**Gate.** The Phase 4 controller decides whether to work:
- **Suppression is unchanged:** presentation, social, backchannel and meta turns produce no intents (tested).
- **Storm guards are replaced:** the controller's single-query storm guards give way to per-intent guards.
- **Corrections:** a correction utterance ("actually I meant the lamp, not the lens") opens the gate when there is an active need to correct. The Phase 4 act classifier rates such utterances not retrieval-worthy; this was found by the streaming run (§20).

The brief's TEST 6, from the actual trace (virtual clock, fixture index):

```
0.4 s  INTENTS v1 +I1 'the fog signal'           -> Q1 (I1)                    -> provisional fused set
1.2 s  INTENTS v2 +I2 'the lens'                 -> Q2 (I2) only; I1 evidence reused
1.6 s  "especially during"                       -> marker without content: no constraint yet, no version
2.0 s  INTENTS v3 ~I1,I2  K1 'during a storm'    -> Q3 (I1 v2), Q4 (I2 v2) in parallel
2.5 s  end: no query changed -> no retrieval; final fusion; post-final latency 0 ms
```

**Corrections (TEST 7):**
- The target becomes `SUPERSEDED`.
- Its queued queries are cancelled; its completed queries are kept with `stale_reason=intent_superseded`.
- It is excluded from fusion, and the corrected need is retrieved.
- This works within an utterance and across utterances.
- Example: after u1 "Tell me about the lens", u2 "And how often is it cleaned?", u3 "Actually, I meant the lamp instead of the lens", I2 is superseded by "how often is the lamp cleaned". That query is built entirely from verbatim spans of u2 and u3.

**Session ledger reuse (REQ-MI-005):** a repeated question reuses the earlier evidence (`RETRIEVAL_SKIPPED reason=ledger_hit`). In the streaming dev run the repeated question (K3) made 2 reuse decisions, one per version of the need, and **0 retrievals**.

---

## 14. Intent Versioning

`intents/tracker.py`; `docs/multi_intent/09`. Each open gate re-decomposes the current transcript (p50 0.28 ms) and reconciles it with the previous version.

| Draft vs previous | Result |
|---|---|
| Matches (containment ≥ 0.6 or Jaccard ≥ 0.5), unchanged | Same id and version |
| Matches, changed | Version + 1, with `changed: [text / constraints / context]` |
| No match | Added |
| Superseded by a correction | Maps back to its existing id |
| Vanished | `DROPPED`, reason `no_longer_in_decomposition` |

- A new `IntentSet` version is created only for a non-empty delta. A final "." creates none; nor does an empty decomposition such as "actually I meant" before its content.
- Streaming dev run: **p50 2, p95 4 versions** per utterance (max 4).

---

## 15. Delta Retrieval

**Rule:** retrieve exactly the `affected_intents` (added or modified) whose query text changed; every other intent keeps its evidence.

| Measured (streaming dev run, 60 utterances) | Value |
|---|---|
| Re-retrieval of an unchanged intent (same intent, same query text) | **0** |
| Retrievals per utterance (p50 / p95 / max) | 2 / 4 / 5 (mean 2.30) |
| Final distinct needs per utterance (gold `expected_query_count`, mean) | 1.57 |

The extra 0.73 retrievals per utterance are **refinements** of a need as its words arrive. Example (S1): "how many crates" → "… can a picker fill" → "… in one shift". Each is a new version of the same intent within the per-intent budget of 3 and the 400 ms cooldown. This is the Phase 4 provisional/final pattern, now per intent.

**Lineage:** a modified intent's new query supersedes only that intent's previous version (`relation: refines | replaces`); other intents' lineages are untouched.

---

## 16. Benchmark Design

**`MultiIntentBenchmarkCase`** (`models/benchmark.py`, schema exported), per utterance:
- chunks with timestamps;
- `expected_intents` (description, paraphrases, gold fixture citations, `required_query_terms`, `superseded`, `answerable`);
- `expected_constraints` (text, scope, `applies_to`);
- `expected_relationships`;
- `expected_query_count`.

No answers and no retrieval results are stored.

**Suite** (`research/phase5/build_dev_suite.py` → `eval/dev_multi_intent/`):

| Count | Value |
|---|---|
| Cases / utterances / gold needs | 53 / 60 / 94 (4 more are gold superseded needs) |
| Constraints / relationships | 6 / 12 |
| Compound utterances (≥ 2 needs) | 30 |

**Categories:** A two independent · B three · C incremental second need · D need + constraint · E refinement · F follow-up · G pronoun · H shared context · I over-decomposition traps · J under-decomposition traps · K duplicates · L changed / corrected need · S single-need controls.

**Chunk timing:** 2.6 words/s with ±15% seeded jitter, as in Phase 4.

**Scoring** (`bench/multi_intent.py`; spec §22.6, fixed before scoring):
- **Intent match:** 0.5·cosine (**all-MiniLM-L6-v2**, a separate model from the system's bge-small, to avoid self-preference) + 0.5·content-term F1, maximised over description and paraphrases. One-to-one Hungarian assignment; match iff ≥ **τ = 0.5** (pre-registered).
- **Final state:** each utterance is scored in the session's *final* state, so a later correction counts.
- **Constraints:** matched by term Jaccard ≥ 0.5. Then attachment (`applies_to` mapped through the intent matching) and scope (scored only for utterances with ≥ 2 needs) are checked.
- **Retrieval:** citation-level.
- **Status labels:** MEASURED = dev suite (not reportable) · BLOCKED = official data · ESTIMATED = none (no number in this report is estimated).

**Anti-cheating:**
- The guard now also scans every dev-suite utterance, gold description and paraphrase (≥ 4 words) against `src/` and `configs/`, and passes.
- No benchmark-specific branches or query mappings exist.
- Fixed phrases come from the *indexed corpus*, not a list.

---

## 17. Benchmark Results

**Decomposition** (offline: final transcripts, sequential utterances; MEASURED, dev suite):

| Metric | Blind run | Post-fix |
|---|---|---|
| Intent precision / recall / F1 (92 matched of 94) | 0.958 / 0.979 / 0.968 | **0.989 / 0.979 / 0.984** |
| Exact intent count per utterance | 0.933 | 0.983 |
| G3 lenient (≥ 2 gold needs matched; 30 compound utterances) | **1.000** | **1.000** |
| G3 strict (all matched, no extra) | 0.967 | 0.967 |
| Constraint P / R | 0.833 / 0.833 | 0.833 / 0.833 |
| Constraint attachment accuracy (matched constraints) | 1.000 | 1.000 |
| Constraint scope accuracy (≥ 2 needs) | 1.000 | 1.000 |
| Relationship P / R | 1.000 / 0.417 | 1.000 / **0.500** |
| Context retention (24 required) | 0.750 | 0.792 |
| Supersession recall (4 gold) | 0.250 | **1.000** |

**Per category (post-fix):**
- **Perfect F1** in A, C, D, E, G, H, I (over-decomposition traps), J (under-decomposition traps), K, L and S.
- **B (three needs):** F1 0.957. B3's "…cleaned what goes into the logbook" has no auxiliary after "what" and stays merged.
- **F (follow-ups):** F1 0.875. "and the opening hours?" names no facet noun and inherits nothing.

**Retrieval** (fixture sanity):
- **Per-intent Recall@3: 0.968** (blind: 0.957).
- The 3 misses follow from decomposition or context misses, not from retrieval (§20).

**Streaming** (virtual clock, cases streamed chunk by chunk, MI enabled):

| Metric | Blind | Post-fix |
|---|---|---|
| Matched needs retrieved **before** the utterance ended (eligible: ≥ 2 chunks) | 0.978 (91) | **0.978** (92) |
| … or served from an earlier utterance's evidence (ledger hit) | — | **0.989** |
| Unified Coverage@8 at turn end | 0.966 | **0.983** |
| Duplicate retrieval of the same intent | 0 | **0** |
| Lead time p50 / p95 (first retrieval start → utterance end) | 2,313 / 5,538 ms | 2,237 / 5,480 ms |
| Post-final retrieval latency p50 / p95 (modeled 10 ms retrieval) | 0 / 10 ms | 0 / 10 ms |

**The one matched need not retrieved early** is G2's "when is it filled in": its last chunk carried it.

---

## 18. Ablation Results

Brief §36. Accuracy is Intent Coverage@3 and @5 (59 utterances). Latency is end-to-end per compound utterance: 30 utterances × 3 repetitions, 300 ms idle before arms A–C as in live speech, p50 / p95.

| Arm | Coverage@3 | Coverage@5 | Retrievals per compound utterance | Duplicate items @8 | Latency p50 / p95 |
|---|---|---|---|---|---|
| **A** single query (Phase 4 QueryBuilder on the whole utterance) | 0.961 | 0.983 | 1 | 0 | 11.6 / 14.0 ms |
| **B** multi-intent, sequential, concatenated | 0.910 | 0.935 | 1 per need (2–3) | 0.125 | 18.7 / 24.9 ms |
| **C** multi-intent, parallel, concatenated | 0.910 | 0.935 | Same | 0.125 | 17.3 / 22.4 ms |
| **D** C + intent-aware fusion | **0.983** | 0.983 | Same | 0 | 17.7 / 22.9 ms |
| **E** D + rerank (`intent_ce`) | 0.983 | 0.983 | Same | 0 | 47.7 / 58.4 ms |

**Reading:**
- **What helps:** decomposition alone (B/C) *hurts* when results are concatenated. Decomposition plus fusion (D) beats the single query at k = 3 (0.983 vs 0.961). On this suite that corresponds to **2 utterances (B1, F4)**, where one whole-utterance query missed a need that per-intent queries covered.
- **At k ≥ 5:** all fusing arms tie at the fixture ceiling.
- **Cost:** decomposition + fusion costs about 6 ms more than one query (D vs A, p50). The cross-encoder adds about 30 ms with no measurable gain here.
- **What would change the picture:** the decisive comparison needs the official corpus, where a single query cannot cover several sections inside a small budget so easily.

---

## 19. Performance

Profile (dev machine, CPU; `results/profile.json`, `offline/metrics.json`):

| Stage | Condition | n | p50 | p95 |
|---|---|---|---|---|
| Decomposition, per streaming update | In session | 308 | **0.28 ms** | 0.73 ms |
| Decomposition, final transcript | Offline | 60 | 0.50 ms | 0.91 ms |
| Query generation (all intents of an utterance) | Offline | 60 | 0.09 ms | 0.16 ms |
| Multi-query retrieval, parallel, all intents of an utterance | Back-to-back | 59 | 4.5 ms | 6.7 ms |
| Multi-query retrieval, 2–4 queries | After 300 ms idle | 30 reps | 13.8–20.5 ms | 15.7–22.7 ms |
| Fusion (intent-aware), per call in session | In session | 211 | **0.11 ms** | 0.20 ms |
| Rerank `intent_ce` / `cross_intent_dense` / `cross_intent_ce` | Profile run | 30 | 22.3 / 3.5 / 30.8 ms | 28.4 / 5.8 / 54.7 ms |
| End-to-end, compound utterance: decompose + parallel retrieve + fuse (arm D) | Idle start | 90 | 17.7 ms | 22.9 ms |

**Bottlenecks:**
- **Retrieval (ONNX query embedding)** dominates, about 10 ms per query after idle, as found in Phase 4.
- **The optional cross-encoder** is the largest optional cost.
- **Decomposition and fusion** are under 1 ms each and are not worth optimising.
- **What helps:** batching embeddings for ≥ 3 same-tick queries saves 10–28% (§8). Keeping reranking off until it is shown to help on real data saves 20–30 ms.

**Nothing was optimised in this phase.** The full benchmark ran in 226 s.

---

## 20. Failure Cases

**Defects found by the dev suite and fixed** (the blind-run numbers above predate fixes 1–3; fixes 4–7 came from the streaming evaluation):

| # | Defect | Found by | Fix |
|---|---|---|---|
| 1 | A question right after a correction phrase ("… cleaned **no wait how often** is the lens polished", "**scratch that where** are …") opened its own clause, so the correction was lost | Blind run (L3, L4) | A marker or correction phrase absorbs the following question or request head |
| 2 | A follow-up's inherited topic was re-searched by its normalised text ("using ladder" vs "using **a** ladder") and lost | Blind run (F4) | Intents store `topic_span` |
| 3 | The evaluator scored supersession per utterance, so a correction in the *next* utterance could never count | Blind run (L2) | Score the final session state (evaluator bug, not a system bug) |
| 4 | A correction utterance ("actually I meant the lamp, not the lens") was judged not retrieval-worthy by the Phase 4 controller, so the gate stayed closed | Streaming run (L2) | Correction gate (§13) |
| 5 | The whole-utterance fallback minted a junk intent "I meant" from the partial "actually I meant" | New test for #4 | No fallback while a correction is pending |
| 6 | After a correction superseded its target, the next quiet tick re-decomposed the utterance, found no target and dropped the corrected need | Streaming trace (L2) | Targets superseded by the current utterance stay in the correction context (never as anaphora antecedents) |
| 7 | "X, not Y" was not parsed as a replacement, so "lens" stayed in the corrected query | Same | A bare "not" between new and old content in a correction clause separates them |
| 8 | Conflict check counted the unit word as shared context (false positive) | Unit test | Unit words excluded |
| 9 | Realtime-replay helper, conftest name clash and a YAML `on: true` lexicon entry | Development | Fixed (tests and loader validation) |

**Remaining failure modes** (measured, *not* tuned away):

1. **Wh-word without an auxiliary.** "how often is the lamp cleaned **what goes** into the logbook" is under-split (B3): the restart rule needs an auxiliary after "what".
2. **Follow-ups that name a new noun.** "and what about the **storage** rules", "and the **opening hours**?" inherit nothing (F1, F2): only facet-only follow-ups inherit. The lexicon was frozen and not extended to fit the suite.
3. **Refinements without lexical overlap.** "pruning" vs "how thick branches are cut", and "logbook rules" vs "what ink the entries use", are not linked (E2, E3).
4. **Pronoun blocked by an earlier word.** A content word earlier in the same need blocks resolution: "what must **never** be used on **it**" (G3), "before they leave **it**" (G4). Without part-of-speech information a verb or adverb looks like a possible antecedent, so G3's evidence for "it" (the lens) is missed (coverage 0 for that utterance).
5. **Restriction across clauses.** A restriction inside a later clause without a marker ("… and how many can a picker fill **for the night shift**") is not distributed to the earlier question (D3).
6. **Shared topic vs restriction.** "cleaning and logging **for the lamp**" is classified as a restriction constraint rather than a shared topic (H3, constraint false positive).
7. **Comma-ended chunks delay retrieval.** A chunk ending in a list comma ("…, how are the wicks trimmed,") keeps the Phase 4 controller's *dangling* rule closed for the whole transcript, so complete earlier needs wait for the next chunk (B1 trace). They are still retrieved before the utterance ends.
8. **Too-generic first query.** The first provisional query of a growing need can be generic ("What are the rules", B1) and is superseded a few seconds later. Its retrieval cost is spent (counted in the 2.30 retrievals per utterance).

---

## 21. Known Limitations

1. **No official data.** All quality numbers are dev-suite numbers on a fictional 14-chunk corpus, written by the rule authors. The post-fix numbers are optimistic. Retrieval and fusion comparisons hit the corpus ceiling at k ≥ 5.
2. **English, rule-based, no part-of-speech tagging.** Anaphora, topic and coordination heuristics fail on the patterns in §20. The optional LLM check is unmeasured because there is no backend (ADR-007 Q2).
3. **Conflict detection is numeric-only and potential-only.** Contradictions in wording are not detected.
4. **One query string per intent** for both BM25 and dense; no separate lexical and dense query forms.
5. **Batched dispatch is not used in streaming,** although it is faster for ≥ 3 same-tick queries.
6. **Cross-utterance refinement** (a constraint-only follow-up such as "especially for visitors" as its own utterance) is deferred to Phase 6. It falls back to one Phase 4-style query and is recorded in `dropped`.
7. **No answer generation, claim extraction or citation validation** (out of scope by design).

---

## 22. Phase 6 Requirements

**Ready for Phase 6:**
- versioned `IntentSet`s with `IntentSetDelta.affected_intents`;
- per-intent query lineage with stale evidence retained;
- supersession;
- session ledger reuse;
- `UnifiedEvidenceSet` with per-intent coverage and full provenance;
- `TURN_COMPLETED.lineage`;
- deterministic replay of multi-intent traces.

**Exact prerequisites:**

1. **Cross-turn constraint updates.** Map a constraint-only follow-up utterance onto the previous turn's intents as `MODIFIED` (with `op: set / update / retract`) instead of the current fallback. This is the core of late-detail refinement (spec §14).
2. **Claim / answer versions keyed by intent.** Phase 6 needs `AnswerVersion` / `Claim` records (contracts exist from Phase 3) with an `intent_id`, so `affected_intents` decides which claims are carried verbatim and which are regenerated (ADR-006).
3. **Provisional-evidence reuse for refined queries.** When a need's final query only refines its last provisional one (containment ≥ 0.6, `relation: refines`), reuse that evidence instead of re-retrieving. This is the open Phase 4 slow-retrieval finding, and it matters before any cross-encoder or remote index is enabled.
4. **Per-clause stability.** Replace the whole-transcript dangling rule with a per-clause one, so complete earlier needs are not held back by a trailing comma (§20 #7).
5. **Official multi-intent and refinement cases** (spec §22 categories 2, 3, 6, 7, 10) authored on the official corpus, with a held-out test split. All Phase 5 thresholds must then be re-checked: τ_eval, the scope rules and `min_per_intent`.
6. **Still blocked externally:**
   - the official corpus;
   - official streaming / multi-intent cases and labels;
   - the deadline;
   - the LLM backend decision (Q1, Q2, Q3, Q6).
