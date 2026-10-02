# Samsung PRISM Theme 4
# Streaming Live RAG
# Phase 2 — Problem Analysis & System Specification

| | |
|---|---|
| Date | 2026-10-02 |
| Status | Specification. Nothing implemented. |
| Inputs | `PHASE_1_RESEARCH_DOSSIER.md`; *Theme 4 Guide_RAG.pdf* (Samsung); hackathon deck; `research/phase1/`, `research/phase2/` measurements |
| Companion artifacts | `docs/architecture/01–06` (diagrams), `docs/decisions/ADR-001…012` |
| Normative language | **SHALL** = mandatory, **SHOULD** = expected unless justified, **MAY** = optional |

**Source tags.** These are the same as Phase 1.

| Tag | Meaning |
|---|---|
| `[G§n pX]` | Theme 4 guide, section *n*, page *X* |
| `[D sN]` | Hackathon deck, slide *N* |
| `[P1 §n]` | Phase 1 dossier |
| `[M]` | Measured on the dev machine (scripts in `research/`) |
| `[E]` | Engineering estimate; must be measured in Phase 3 |
| `[T]` | Team target, not an official threshold |
| `[INF]` | Our inference |

**What this document does not contain.** It contains no benchmark results and no corpus facts. The corpus is still unavailable as of this date. Every threshold marked *calibrate* gets its value from the tune split in Phase 3.

---

## 1. Problem Definition

### 1.0 Specification baseline: Phase 1 re-checked against the official documents

**Carried forward from Phase 1** (re-verified against the guide and the deck):

| Item | Phase 1 position | Re-check |
|---|---|---|
| Architecture | B: streaming hybrid RAG, rule-first controller, query ledger, versioned answers [P1 §18] | Consistent with the guide's 5-component pipeline [G§2 p2] |
| Technologies | BM25 (scipy sparse) + small dense embedder (bge-small default, MiniLM fallback), RRF, optional MiniLM-L6 cross-encoder, exact numpy search, asyncio, FastAPI/SSE demo, pluggable LLM (hosted / local / extractive) [P1 §12] | Consistent. The embedder is still unproven on the corpus (decided by Exp 1). |
| Hard constraints | Corpus isolation, no hardcoding or precomputation, rigorous grounding, session-bound state, parsimony, full duplex, single-command reproducibility [P1 §5] | Consistent with [G§3 p3], [D s7], [G§5 p4] |
| Benchmark requirements | G1–G6 [G§5 p4–5]; baseline comparison, ≥2 ablations, ≥3 edge-case failure analyses [G§8 p6]; report recall, groundedness, TTFT, cost per turn [D s7] | Consistent |
| Corpus characteristics | **Missing.** Implied `Doc_NN` IDs with `§N` sections; enterprise policy domain (events/venues, travel reimbursement) [P1 §4] | Still missing as of 2026-10-02 (re-checked) |
| Metrics | [P1 §16] | Kept, with corrections K5 and K6 below |
| Risks | R1–R14 [P1 §20] | Still valid. R1 (corpus) and R2 (deadline) remain open. |
| Unresolved questions | Q1–Q12 [P1 §19] | All still open. None have been answered yet. |

**Contradictions and corrections.** Where Phase 1 conflicts with an official requirement, the official requirement wins. Each correction is recorded here and in an ADR.

| ID | Phase 1 said | Official source says | Resolution (now normative) | ADR |
|---|---|---|---|---|
| **K1** | Baseline B0 is *dense-only* static RAG [P1 §14] | The roadmap defines the baseline retrieval pipeline as **dense/sparse hybrid** [G§7 p5]. Dense-only appears only as an *ablation* example [G§8 p6]. | **B0 is a static, turn-based *hybrid* (BM25+dense RRF) pipeline.** Dense-only becomes an ablation arm (Exp 2). The ladder becomes B0 hybrid static → B1 + cross-encoder → B2 + decomposition and quota fusion → B3 + streaming controller → B4 + session refinement → B5 + grounding verifier. | ADR-011 |
| **K2** | One command can be `docker compose up` *or* a CLI runner [P1 §7.3] | G1's validation explicitly says a **container** launches with one command [G§5 p4]. The deliverables list allows either [G§8 p6]. | **A container (`docker compose up`) is the mandatory primary path.** The CLI runner is secondary, for development. | ADR-007 |
| **K3** | Reranking means a cross-encoder, and it is optional [P1 §8.3] | The roadmap names "RRF and deduplication reranker" [G§7 p5]. The pipeline requires re-rank and dedup [G§2 p2]. | **The mandatory re-rank stage is RRF fusion plus deduplication**, which is always on. The cross-encoder is an *optional enhancement*, gated by Exp 3. The official requirement is met even with the cross-encoder disabled. | ADR-003 |
| **K4** | Out-of-corpus questions might be answered "not in KB" *without* retrieval [P1 §8.5, §17-H] | If evidence is insufficient for a sub-intent, the system must emit uncertainty [G§3 p3], and that judgment needs a retrieval attempt. False triggers are measured on *no-retrieval cases* [G§5 p5], which the guide exemplifies with presentation turns [G§4 p4]. | **Every information-request segment gets retrieval at the latest at the end of the utterance.** Corpus-vocabulary signals affect only *how early* retrieval fires, never *whether*. Suppression is limited to presentation, social, backchannel and meta acts (§15). | ADR-004 |
| **K5** | G3 pass formula `(#matched ≥ 2) ∧ (#matched == #gold or ≥ 2)` [P1 §10] | (internal error: the second conjunct is redundant) | Corrected definitions in §21 and §22.6: *lenient* G3 = ≥2 distinct gold intents matched; *strict* = all gold intents matched and no unmatched predictions. | — |
| **K6** | "Corpus affinity" could decide NO_RETRIEVE [P1 §8.5.1] | Same as K4 | Corpus affinity is renamed *anchor strength*. It is used only for early-trigger sufficiency. | ADR-004 |
| **K7** | Presentation turns create answer version v(n+1) with kind=presentation [P1 §9.3] | The guide only says prior citations are retained and no new ones are fabricated [G§4 p4]. It says nothing about versioning. | Kept as **our design choice**, labeled as such. | ADR-006 |
| **K8** | TTFT ≤ 1.5 s [P1 §7] | No official TTFT threshold; the deck only asks that TTFT be reported [D s7] | Kept as team target `[T]`. Never presented as official. | — |
| **K9** | Implicitly a linear pipeline | The guide's components run *while* the user speaks [G§1–2] | Replaced by a **hierarchical state machine with concurrent regions** (§7). This is a design refinement, not a contradiction. | ADR-010 |

### 1.1 Formal problem statement

**Givens.**

- **Corpus.** A corpus 𝒞 of documents *d*, each with ordered sections *s*, and each section split into chunks *c* that carry a citation key `Doc_ID §Section`.
- **Session.** A session *S* is a sequence of turns *u₁…uₙ*.
- **Turn.** Each turn arrives as a stream of chunks `(tⱼ, xⱼ)`, with *tⱼ* in seconds relative to the start of the utterance and *xⱼ* a text delta. The stream ends with an end-of-utterance mark at *t_end*.

**For every turn the system SHALL produce:**

1. A decision sequence *dⱼ ∈ {WAIT, RETRIEVE, NO_RETRIEVE}*, one per chunk plus the end-of-utterance mark.
2. A set of intents *I(u)* and, for each retrieval, the time it started.
3. An answer version *A_v*. This is a set of claims, each mapped to ≥1 chunk of 𝒞, or carrying an explicit uncertainty record. It also records its parent version *A_{v−1}* when the turn refines an earlier answer.
4. A complete telemetry trace.

**Subject to:**

- Evidence only from 𝒞.
- State scoped to *S*.
- No benchmark-specific logic.

**Optimized for, in priority order:**

1. Groundedness: ≥85% of claims supported, 0 invalid IDs (G4).
2. Early retrieval: ≥80% of eligible turns (G2).
3. Intent isolation: ≥70% of compound turns (G3).
4. Refinement continuity (G5).
5. Latency and cost.

### 1.2 What is wrong with conventional RAG

Conventional RAG computes `answer = LLM(q, topk(retrieve(q)))` once per turn. That rests on six assumptions, and Theme 4 violates every one:

| # | Assumption of conventional RAG | How Theme 4 violates it | Source |
|---|---|---|---|
| A1 | The query is complete when retrieval starts | The query arrives incrementally over seconds | [G§1 p1] |
| A2 | One query is one information need | One utterance packs several needs | [G§1 p1] |
| A3 | The query is phrased for the retriever | Speech is disfluent and context sits in other clauses ("…in Pune for 30 people, and I need the cancellation policy") | [D s7], [G§4 p3] |
| A4 | Every turn needs retrieval | Formatting and restatement turns must not query the corpus | [G§4 p4], [G§6 p5] |
| A5 | Turns are independent, or history is plain text | Later turns modify earlier answers | [G§4 p4] |
| A6 | Each answer is regenerated from scratch | The answer must be refined, not restarted, keeping citations | [G§1 p1], [G§5 p5 G5] |

### 1.3 Why static RAG fails here (concrete)

**Latency serialization.** Static RAG puts every stage after the end of the utterance:

`TTFT_static = T_retrieve + T_rerank + T_decompose + T_assemble + TTFT_llm`

Streaming moves the first three stages into the time the user is still speaking:

`TTFT_stream ≈ T_assemble + TTFT_llm + spill`

Measured on our hardware:

| Stage | Cost | Status |
|---|---|---|
| Hybrid retrieval | < 10 ms per query | [M] |
| MiniLM-L6 rerank | 105–124 ms per intent at 20 candidates, so 0.31–0.37 s for 3 intents | [M] |
| LLM decomposition check | 0.3–1.5 s | [E] |

So streaming removes roughly **0.3–1.9 s** from the post-utterance critical path. Almost none of that saving comes from first-stage retrieval, which is already cheap. It comes from reranking, decomposition and evidence assembly.

**Quality failures.**

- A compound utterance sent as one query creates a *centroid problem* (§1.5).
- A refinement sent as concatenated text creates *answer drift* (§1.7).
- A presentation request sent to the retriever creates *citation pollution* (§1.6).

### 1.4 Why streaming retrieval is necessary

1. **It is a scored requirement.** G2 requires retrieval to start before the transcript is complete in ≥80% of eligible queries [G§5 p5]. The deck makes full duplex a scope constraint [D s7].
2. **It moves expensive work off the critical path.** Reranking (measured above) and the optional LLM decomposition check can run in the silence between the last chunk and the end of the utterance: 0.5 s in the guide's example [G§4 p3].
3. **It detects insufficiency early.** If an intent has no supporting evidence, the system knows *before* the user finishes. It can prepare an uncertainty statement or a targeted clarification [G§3 p3] at no added latency.
4. **The benefit grows with heavier retrieval.** With a larger corpus, a remote index or a larger reranker, retrieval costs grow and so does the benefit. The design must not assume retrieval stays at milliseconds `[INF]`.

### 1.5 Why query decomposition is necessary

Embedding a 3-intent utterance produces roughly a weighted mixture of the three intents. Nearest neighbors of that mixture tend to be:

- **(a)** chunks for the dominant intent, or
- **(b)** overview chunks that mention all three facets shallowly.

BM25 has the same problem: summing term scores across facets rewards chunks that match the most facets, not the chunk that answers each facet best. A shared top-k budget then starves the weaker intents.

Per-intent retrieval with **per-intent quotas** (§12) guarantees each intent gets its own best evidence. It is also the only way to support:

- per-intent uncertainty ("catering for venue A could not be verified" [G§4 p4]);
- G3 isolation [G§5 p5].

### 1.6 Why retrieval suppression is necessary

Presentation turns contain topical words. "Put the reimbursement answer in two bullets" contains "reimbursement", which the retriever would happily match. Retrieving here causes three problems:

1. **New evidence enters the prompt.** The model may add facts or citations that were not in the prior answer. The guide requires prior citations to be kept and none to be fabricated [G§4 p4].
2. **Wasted tokens and drift** [G§6 p5 #4].
3. **A false trigger** in G2's no-retrieval cases [G§5 p5].

### 1.7 Why session-aware refinement is necessary

The naive approach to a late detail is to concatenate the turns, re-retrieve and regenerate. That causes four problems:

1. **Duplicate retrieval** of everything already found.
2. **Ranking shift.** The constraint terms dominate the new combined query, so evidence for the original facets can fall out of top-k. Previously correct claims and their citations then *disappear* (non-monotonic answers).
3. **Roughly double the latency and tokens.**
4. **A G5 violation:** re-executing full-corpus search [G§5 p5]. The guide names this pitfall directly [G§6 p5 #2].

### 1.8 Why ordinary conversation memory is insufficient

A message-history buffer stores *text*. Refinement needs *structure*:

| Needed for refinement | Plain chat memory | Structured session state (§13) |
|---|---|---|
| What was already retrieved (avoid re-search) | ✗ | Query ledger |
| Which evidence supports which statement | ✗ | Claim → evidence graph |
| Which statements a new constraint affects | ✗ | Claim ↔ intent ↔ slot links |
| What the active topic and its constraints are | Implicit in text | Topic frame with slots |
| Proof of continuity (v1 → v2, citations retained) | ✗ | Version lineage + diff |
| Guarantee that preserved facts stay verbatim | ✗ (an LLM rewrites everything) | Deterministic carry-over of unaffected claims |

Summarizing history with an LLM also drops citations. Summaries cannot be grounded.

### 1.9 Why we cannot simply wait for the final transcript

1. G2 would be 0% by construction.
2. The silence between the last speech and the end of the utterance is pure waste in a static design. That is exactly where per-intent reranking fits (0.31–0.37 s for 3 intents [M] vs the guide's 0.5 s gap).
3. The final transcript does not exist as a clean object in a full-duplex system. Endpointing is a guess, and users keep adding details. A system that waits for "the end" waits forever, or answers late.

### 1.10 Why we cannot simply retrieve on every transcript chunk

1. **Chunks are arbitrary cut points.** The guide's first chunk ends on a dangling preposition [G§4 p3]. Fragment queries are noisy, and their results pollute the candidate pool [G§6 p5 #1].
2. **Retrieval storms.** A 30 s utterance at ~0.8 s per chunk is ~38 chunks. With N intents, a per-chunk policy issues ~38·N queries and, if reranking each, ~38·N·0.1 s CPU [M]. That saturates a 2-core judge machine.
3. **Duplicate events** make telemetry unreadable and inflate cost.
4. **False triggers** on presentation and social turns (G2).
5. **LLM per chunk** would multiply cost and latency, violating parsimony [G§3 p3].

The answer is a **controller** that retrieves on *semantic events*: clause closure, stability, novelty. It does not retrieve on *transport events* (chunk arrival). That is §8.

---

## 2. System Objectives

### 2.1 Primary objective (single)

> **For every turn of a streamed conversation, deliver one answer in which every factual statement is backed by a valid citation to the supplied corpus or is explicitly marked as unverifiable, while starting the retrieval that the answer depends on before the user finishes speaking.**

Measured by: G4 (citation support ≥85%, 0 invalid IDs) **and** G2 (early-retrieval rate ≥80%), on the same replay run.

### 2.2 Secondary objectives

| ID | Objective | Serves | Measured by |
|---|---|---|---|
| SO-1 | Identify and isolate every distinct information need in a compound utterance | G3 | Sub-query recall and precision, G3 rate |
| SO-2 | Retrieve relevant evidence per intent with hybrid search | G4, recall | Recall@k per intent |
| SO-3 | Fuse and rerank evidence into one budgeted, deduplicated, diverse set | G4 | Intent coverage@k, context tokens |
| SO-4 | Suppress retrieval when the turn needs none | G2 (false triggers) | False-trigger rate |
| SO-5 | Refine the existing answer for late details without restarting | G5 | Re-issue count = 0, retained-citation ratio, lineage |
| SO-6 | Keep all memory strictly session-scoped | C4 | Isolation test |
| SO-7 | Emit explicit uncertainty or clarification instead of guessing | C3/G4 | Uncertainty precision and recall |
| SO-8 | Make every decision observable and costed | G6 | Trace coverage 100% |
| SO-9 | Do all this with minimal components and LLM calls, reproducibly on CPU | C5, G1 | LLM calls per turn, single-command replay |

### 2.3 Non-objectives

Not pursued in this project:

- Speech recognition quality, TTS, wake word.
- UI polish beyond a demo view.
- Fine-tuning.
- Multilingual support (unless Q7 changes).
- Open-domain question answering.
- Cross-session personalization (forbidden).
- Agentic tool use beyond retrieval.

---

## 3. Functional Requirements

**Categories.** This section contains the behavioral categories:

- **A.** Functional (`REQ-FUNC`)
- **C.** Retrieval (`REQ-RET`)
- **D.** Streaming (`REQ-STREAM`)
- **E.** Multi-intent (`REQ-MI`)
- **F.** Session (`REQ-SESS`)
- **G.** Grounding (`REQ-GRD`)

Categories B, H, I and J are in §4.

**Priority.** **Critical** = gate-bearing or a hard rule (failure means a gate fails or disqualification). **High** = strongly affects the score. **Medium** = quality or demo. **Low** = optional.

### 3.A Functional (`REQ-FUNC`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-FUNC-001 | The system SHALL accept `TRANSCRIPT_CHUNK`, `UTTERANCE_END`, `SESSION_START`, `SESSION_END` input events (§5) and process them in `(session_id, utterance_id, chunk_index)` order | Critical | Streaming input [G§2 p2] | 100% of replayed input events produce a `CHUNK_RECEIVED` (or end/session) acknowledgment | Reconcile input event count against `CHUNK_RECEIVED` + lifecycle events in the trace |
| REQ-FUNC-002 | For every accepted chunk and every utterance end, the controller SHALL emit exactly one `CONTROLLER_DECISION` per open segment, with decision ∈ {WAIT, RETRIEVE, NO_RETRIEVE} and ≥1 reason code | Critical | Controller contract [G§2 p2]; explainability | 0 chunks without a decision; 0 decisions without reasons | Trace join on `(utterance_id, chunk_index)` |
| REQ-FUNC-003 | For each turn classified query, refinement, presentation, meta or social, the system SHALL commit exactly one `ANSWER_COMMITTED`. Backchannel turns SHALL commit none. | Critical | One unified response [G§4 p3] | Commit count = expected count per turn type on the replay set | Count by `turn_type` |
| REQ-FUNC-004 | Every committed intent SHALL end in status `answered` (≥1 supported claim) or `uncertain` (an uncertainty record) | Critical | No silently dropped sub-intent [G§3 p3] | 100% of intents in `TURN_COMPLETED.intent_status` ∈ {answered, uncertain} | Schema check |
| REQ-FUNC-005 | The system SHALL emit a `TURN_COMPLETED` record that is a superset of the guide's output record (`retrieval_events`, `sub_queries`, `answer`, `citations`, `uncertainty`) | Critical | Guide output format [G§4 p4] | 100% of turns validate against `TurnResult` schema | JSON-Schema validation |
| REQ-FUNC-006 | The system SHALL support multi-turn sessions with turn types query, refinement, presentation, social, backchannel and meta (§8.4) | High | Examples 2 and 3 [G§4 p4] | Turn-type accuracy reported on Categories 5–9 | Compare against gold `turn_type` |
| REQ-FUNC-007 | When a critical ambiguity blocks answering (unresolved referent, missing decisive slot), the system SHALL emit a targeted clarification (`kind=clarification`) instead of guessing | Medium | C3 allows clarification [G§3 p3] | ≥ target share of Category 8 cases produce clarification or uncertainty; 0 fabricated resolutions | Gold `clarification_expected` |
| REQ-FUNC-008 | A replay runner SHALL feed recorded sessions in real-time (speed factor) or virtual-time mode | Critical | G1 automated replay [G§5 p4] | Replay of the full dev suite completes unattended in both modes | Exit code + run manifest |
| REQ-FUNC-009 | An interactive mode SHALL accept live typed or streamed text chunks for the demo | Medium | Demo video [G§8 p6] | Demo script runs end to end | Manual demo checklist |
| REQ-FUNC-010 | The system MAY accept microphone audio via streaming ASR | Low | Voice simulated is acceptable [D s7] | — | — |

### 3.C Retrieval (`REQ-RET`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-RET-001 | Retrieval SHALL search only the indexed supplied corpus. No network retrieval, no web. | Critical | Corpus isolation [G§3 p3] | 100% of evidence `chunk_id`s exist in the index manifest; 0 outbound requests except to the configured LLM endpoint | Trace check + egress test |
| REQ-RET-002 | Each retrieval SHALL run lexical (BM25) and dense search and fuse them by RRF | Critical | Hybrid mandated [G§2 p2, §7 p5] | Every `RETRIEVAL_COMPLETED` lists lexical and dense ranks | Trace schema |
| REQ-RET-003 | A re-rank stage SHALL always run: RRF + dedup at minimum, plus an optional cross-encoder | Critical | [G§2 p2], [G§7 p5] (K3) | `EVIDENCE_RERANKED.method` ∈ {rrf_dedup, cross_encoder} on every committed intent | Trace |
| REQ-RET-004 | Chunking SHALL respect section boundaries. Each chunk maps to exactly one `(doc_id, section_id)`. | Critical | Citation unit `Doc_ID §Section` [G§3 p3] | 0 chunks spanning two sections | Index audit script |
| REQ-RET-005 | Document and section IDs SHALL be preserved from the corpus when present, and otherwise derived deterministically (same corpus → identical IDs) | Critical | Citation traceability; reproducibility | Two index builds produce identical ID sets and hashes | Build twice, diff manifests |
| REQ-RET-006 | Queries SHALL be normalized (case, Unicode, fillers, numerals, stopwords for lexical) by generic rules | High | Spoken input [P1 §8.1.2] | Normalization unit tests; ablation shows no Recall@k loss vs raw | Exp 1 ablation |
| REQ-RET-007 | Retrieval SHALL support metadata pre-filters (doc_id, section_id, doc metadata) | Medium | Scoped delta retrieval (§14) | Filtered queries return only matching chunks | Unit test |
| REQ-RET-008 | Hybrid retrieval SHALL achieve Recall@5 ≥ max(BM25-only, dense-only) Recall@5 on the dev test split | High | Justifies hybrid (C5) | Inequality holds, with a 95% bootstrap CI reported | Exp 2 |
| REQ-RET-009 | The index SHALL be cached by `(corpus_hash, embedder_revision, chunker_version)` and rebuilt automatically on mismatch | Medium | G1 start-up time | Second start skips the build; a changed corpus triggers a rebuild | Start-up log |
| REQ-RET-010 | The cross-encoder, if enabled, SHALL rerank at most `M_rerank` (default 20) candidates per intent | High | Latency (§20) [M] | `EVIDENCE_RERANKED.n_candidates ≤ 20` | Trace |

### 3.D Streaming (`REQ-STREAM`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-STREAM-001 | The system SHALL begin retrieval before utterance completion when a retrieval-worthy segment becomes complete or stable | Critical | G2 [G§5 p5] | Early-retrieval rate ≥ 80% of eligible turns (eligible = needs retrieval ∧ ≥2 chunks; our definition, Q4) | `t_stream(first RETRIEVAL_STARTED) < t_stream(UTTERANCE_END)` per eligible turn |
| REQ-STREAM-002 | The system SHALL NOT retrieve on no-retrieval turns | Critical | G2 "low false-trigger" [G§5 p5] | False-trigger rate ≤ 5% `[T]` on Category 5 | Turns with gold `retrieval_required=false` that have any `RETRIEVAL_STARTED` |
| REQ-STREAM-003 | Retrieval SHALL be dispatched only for segments in state CLOSED or STABLE that pass sufficiency (§8). Never for OPEN dangling fragments. | High | Pitfall 1 [G§6 p5] | 100% of `RETRIEVAL_STARTED` reference a CLOSED or STABLE segment | Trace join |
| REQ-STREAM-004 | Provisional evidence SHALL be re-scored against the final intent set. Evidence whose only source is a dropped intent SHALL NOT be cited. | High | Prevents premature noise from reaching the answer | 0 citations whose evidence is linked only to `DROPPED` intents | Trace join |
| REQ-STREAM-005 | Every event SHALL carry `t_stream_s` (utterance-relative) and `t_wall_ms` (monotonic run clock) | Critical | G2 is measured on stream time; latency on wall time | 100% of events carry both (`t_stream_s` null only for session-level events) | Schema |
| REQ-STREAM-006 | In-flight LLM work superseded by new content SHALL be cancelled and logged with `status=cancelled` | High | Cost and race safety | Every cancellation logged; no committed answer built on a superseded intent set | Race tests (§23) |
| REQ-STREAM-007 | Chunk handling SHALL be idempotent on `(utterance_id, chunk_index)` and tolerate a reordering window of 1 chunk | High | Robust transport | Duplicate and out-of-order fixture tests pass | Unit/integration tests |
| REQ-STREAM-008 | If no `UTTERANCE_END` arrives within `endpoint_timeout_s` (default 2.0 s stream time) after the last chunk, the system SHALL synthesize one with `reason=timeout` | High | Stream ends unexpectedly | Fixture without an end marker still commits an answer | Test |
| REQ-STREAM-009 | Answer synthesis SHALL start only after `UTTERANCE_END` (speculative prefill MAY happen earlier) | High | One unified answer for the full utterance [G§4 p3] | `SYNTHESIS_STARTED.t_stream_s ≥ utterance_end_s` | Trace |
| REQ-STREAM-010 | Retrieval for all committed intents SHOULD be complete at `UTTERANCE_END` | High | TTFT | Evidence-ready-at-end rate reported; target ≥ 80% `[T]` | `RETRIEVAL_COMPLETED.t_stream_s ≤ utterance_end_s` for all committed intents |

### 3.E Multi-intent (`REQ-MI`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-MI-001 | The decomposer SHALL identify and isolate ≥2 distinct intents in compound utterances | Critical | G3 [G§5 p5] | Lenient G3 rate ≥ 70% on compound cases | §22.6 matching procedure |
| REQ-MI-002 | Each intent SHALL get its own retrieval query, enriched with the shared context slots it depends on | High | Context retention [G§2 p2] | Context-retention rate ≥ 90% `[T]` | Gold `shared_slots` contained in the intent query |
| REQ-MI-003 | The decomposer SHALL merge over-split candidates (§9.5) | High | Pitfall 5 [G§6 p5] | Sub-query precision ≥ 0.8 `[T]` | §16.3 of P1, gold intents |
| REQ-MI-004 | Constraints ("especially if Z", "for 30 people") SHALL attach to intents as constraints, not become separate intents | High | Avoid over-splitting; refinement needs constraints | Constraint-as-intent error rate reported | Gold constraint annotations |
| REQ-MI-005 | Intents SHALL be deduplicated against the session query ledger before dispatch (reuse instead of re-search) | High | Example 1 reuse [G§4 p4]; G5 | Exact-duplicate retrieval count = 0; near-duplicate rate ≤ 5% `[T]` | Ledger stats |
| REQ-MI-006 | Retrieval for independent intents SHALL be dispatched concurrently | High | Parallel retrieval [G§1 p1, §2 p2] | Overlapping `RETRIEVAL_STARTED` timestamps for same-tick intents | Trace |
| REQ-MI-007 | Dependent intents (anaphora to another intent) SHALL record `depends_on` and inherit the referenced intent's context | Medium | Correct sub-queries | Dependency annotations in the test suite | Gold |
| REQ-MI-008 | At most one LLM decomposition call per utterance, invoked only under the gating rules (§9.6) | High | Parsimony [G§3 p3] | `LLM_CALL(purpose=decompose)` ≤ 1 per utterance | Trace |
| REQ-MI-009 | Each intent SHALL record provenance (`rule`, `llm` or `reconciled`) and the source segment span | Medium | Explainability, ablation | 100% of intents carry provenance | Schema |

### 3.F Session (`REQ-SESS`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-SESS-001 | All mutable state SHALL be keyed by `session_id` and owned by exactly one session actor. The only shared object SHALL be the read-only corpus index (plus pure-function caches). | Critical | C4 [G§3 p3] | Isolation test (Category 11): 0 cross-session evidence, claim or intent references | Concurrent replay + reference audit |
| REQ-SESS-002 | Session state SHALL be in memory only, destroyed on `SESSION_END` or TTL, and never persisted | Critical | C4 | No session data on disk after a run, other than write-only telemetry | Filesystem diff test |
| REQ-SESS-003 | The pipeline SHALL never read telemetry back | Critical | C4 (telemetry is not memory) | Static check: no telemetry reader imports in the pipeline package | Lint rule / test |
| REQ-SESS-004 | Every committed answer SHALL carry `version` and `parent_version` (null for a new topic thread) | Critical | G5 lineage [G§5 p5] | 100% lineage-valid sessions | Version-graph check |
| REQ-SESS-005 | Refinement turns SHALL issue only delta queries: 0 re-issues of prior ledger queries, `full_rerun=false` | Critical | G5 [G§5 p5] | Re-issue count = 0 on Categories 6 and 7 | Ledger stats |
| REQ-SESS-006 | Unaffected claims SHALL be carried into the new version verbatim, with their citations | High | Preserve established facts [G§4 p4] | Retained-claim fidelity = 100% (exact text and citations) for gold-unaffected claims | Diff check |
| REQ-SESS-007 | The system SHALL classify each turn's type and topic continuity (refinement vs new topic) | High | Example 2 [G§4 p4] | Turn-type accuracy reported | Gold |
| REQ-SESS-008 | Session state SHALL be bounded (evidence store ≤ 500 items, answers ≤ 50 versions, configurable) | Medium | Memory safety | Bounds enforced under the long-session fixture | Test |
| REQ-SESS-009 | Constraints arriving while a synthesis is in flight SHALL be queued (`pending_constraints`) and applied as a refinement after commit | High | Race safety, linear lineage | Fixture with a mid-synthesis detail yields v(n+1) with parent v(n) | Test |

### 3.G Grounding (`REQ-GRD`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-GRD-001 | Every factual claim SHALL carry ≥1 citation | Critical | [G§3 p3] | 0 committed factual claims without a citation | Validator report |
| REQ-GRD-002 | Every emitted citation SHALL resolve to an existing `Doc_ID §Section` in the index | Critical | Zero fabricated IDs [G§5 p5] | Citation validity = 100% | Validate against the manifest |
| REQ-GRD-003 | Cited evidence SHALL support its claims | Critical | G4 ≥ 85% | Support rate ≥ 85% on sampled claims (automatic verifier calibrated against human labels) | §16, Exp 9 |
| REQ-GRD-004 | Every intent without sufficient evidence SHALL produce an uncertainty record naming the unverified aspect | Critical | [G§3 p3], [G§4 p4] | Uncertainty recall ≥ 90% `[T]` on unanswerable gold intents | Gold `answerable=false` |
| REQ-GRD-005 | Presentation turns SHALL NOT add citations or new numeric facts | Critical | [G§4 p4 Ex3] | `citations(v_new) ⊆ citations(v_prev)` and no new numbers, on 100% of Category 5 presentation cases | Validator |
| REQ-GRD-006 | Evidence conflicts SHALL be surfaced as uncertainty with both sources cited | High | Fusion challenge [G§2 p2] | Conflict fixtures produce `uncertainty.kind=conflict` | Test (needs a corpus pair) |
| REQ-GRD-007 | Every number, date or amount in a claim SHALL appear in its cited evidence | High | Cheap high-precision hallucination detector | Numeric mismatch → claim flagged or dropped (100%) | Validator unit tests |
| REQ-GRD-008 | Citation strings SHALL use the format `Doc_ID §Section` | Critical | [G§3 p3, §4 p4] | Regex conformance 100% | Validator |
| REQ-GRD-009 | Answer streaming SHALL NOT present unvalidated factual sentences as final. Default is sentence-gated streaming (§16.6). | High | Grounding guarantee under streaming | Committed text == concatenation of emitted validated sentences | Trace |

---

## 4. Non-Functional Requirements

Categories **B.** Performance (`REQ-PERF`), **H.** Observability (`REQ-OBS`), **I.** Reproducibility (`REQ-REPRO`), **J.** Deployment (`REQ-DEP`).

**Reference hardware.** Every latency requirement is stated for:

- **RH-dev:** the dev machine, M5 Pro, 15 threads; and
- **RH-judge:** a container limited to 2 CPUs and 4 GB `[T]`, as a judge-class proxy.

Both are reported. Acceptance is judged on RH-judge.

### 4.B Performance (`REQ-PERF`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-PERF-001 | Controller decision latency per chunk p95 < 20 ms | High | Retrieval timestamp ≈ chunk timestamp (Ex1) [G§4 p3] | p95 < 20 ms on RH-judge | `CONTROLLER_DECISION.latency_ms` |
| REQ-PERF-002 | Hybrid retrieval (embed + BM25 + dense + RRF) p95 < 50 ms per query at ≤ 50k chunks | High | Streaming budget | p95 < 50 ms on RH-judge | `RETRIEVAL_COMPLETED.latency_ms` |
| REQ-PERF-003 | Time-to-first-token (validated sentence) after `UTTERANCE_END`: p50 ≤ 1.5 s with the hosted backend `[T]`; reported for the local and extractive backends | High | Deck asks for TTFT [D s7] | Target met on the hosted backend; others reported honestly | §20 definitions |
| REQ-PERF-004 | Rerank cost per intent ≤ 150 ms p95 on RH-dev at 20 candidates | Medium | [M] 105–124 ms | p95 ≤ 150 ms | `EVIDENCE_RERANKED.latency_ms` |
| REQ-PERF-005 | LLM calls per turn: query or refinement ≤ 2; presentation, meta or social ≤ 1; backchannel 0 | High | Parsimony [G§3 p3] | 100% of turns within budget | `LLM_CALL` count |
| REQ-PERF-006 | Retrieval calls per utterance ≤ `n_intents + provisional_budget` (default budget 4) | High | Prevents storms [G§6 p5 #1] | 100% of turns within budget | Trace |
| REQ-PERF-007 | Index build ≤ 120 s per 10k chunks on RH-judge `[T]` | Medium | G1 start-up | Measured and reported | Start-up log |
| REQ-PERF-008 | App container RSS ≤ 2 GB, excluding a local LLM sidecar | Medium | Clean-machine fit | Peak RSS measured | `docker stats` / psutil |
| REQ-PERF-009 | Context sent to synthesis ≤ 2,000 evidence tokens per turn by default | Medium | Prefill cost, especially on a local CPU LLM | 100% within budget | `EVIDENCE_FUSED.token_count` |

### 4.H Observability (`REQ-OBS`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-OBS-001 | 100% of turns SHALL produce a schema-valid event trace and `TURN_COMPLETED` record | Critical | G6 [G§5 p5] | Trace coverage = 100% | Schema validation over the run |
| REQ-OBS-002 | Traces SHALL capture timestamps, retrieval triggers, citations, answer version lineage and token cost | Critical | G6 field list [G§5 p5] | Required fields present in 100% of turns | Field-presence check |
| REQ-OBS-003 | Every component stage SHALL record its latency | High | Latency model (§20) | Stage-latency table derivable from traces alone | Eval harness |
| REQ-OBS-004 | Every LLM call SHALL record backend, model, tokens in/out, latency, TTFT, cost and status | Critical | Token cost [G§5 p5], cost per turn [D s7] | 100% of `LLM_CALL` complete | Schema |
| REQ-OBS-005 | Controller decisions SHALL include reason codes and feature values | High | Explainability; brief [G§8 p6] | 100% | Schema |
| REQ-OBS-006 | The telemetry writer SHALL be non-blocking and flush fully on shutdown, including on error paths | High | Trace completeness | Kill-test: SIGTERM mid-run loses 0 committed-turn records | Test |
| REQ-OBS-007 | Each run SHALL write a manifest: git SHA, config hash, model revisions, corpus hash, backend, seeds, hardware | High | Reproducibility | Manifest present and complete | Schema |
| REQ-OBS-008 | All metrics in §20–§22 SHALL be computable from traces plus gold labels alone | High | Auditable evaluation | Eval harness reads only traces and labels | Code review |

### 4.I Reproducibility (`REQ-REPRO`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-REPRO-001 | One command SHALL launch the container, and the replay suite SHALL complete without manual intervention | Critical | G1 [G§5 p4] (K2) | Fresh-clone test passes on amd64 and arm64 | CI script on a clean VM |
| REQ-REPRO-002 | Dependencies SHALL be pinned in a lockfile. Model weights SHALL be pinned by revision hash. | Critical | [G§8 p6] | Lockfile + revisions in the manifest | Review |
| REQ-REPRO-003 | The replay suite SHALL complete with no API keys present (auto-fallback: local → extractive) | Critical | G1 on judge machines (Q2) | Keyless run passes | CI with an empty env |
| REQ-REPRO-004 | Non-LLM stages SHALL be deterministic: same inputs and config give identical decisions, retrieval ranks and evidence sets | High | Comparable experiments | Two runs → identical non-LLM trace fields | Trace diff |
| REQ-REPRO-005 | LLM calls SHALL use temperature 0 (or the provider's deterministic setting) and a fixed seed where supported | High | Variance control | Variance over 3 runs reported | Repeated runs |
| REQ-REPRO-006 | The source tree SHALL contain no benchmark or test utterances, expected answers or corpus-specific strings | Critical | No hardcoding [G§3 p3] | Guard test: no eval string appears in `src/` or `configs/` | Automated grep test |
| REQ-REPRO-007 | Thresholds and prompts SHALL live in versioned config, with the config hash in the manifest | High | Traceability; C2 transparency | Manifest contains the config hash | Review |

### 4.J Deployment (`REQ-DEP`)

| ID | Requirement | Priority | Rationale | Acceptance criterion | Measurement method |
|---|---|---|---|---|---|
| REQ-DEP-001 | `docker compose up` SHALL start the app (and the optional Ollama sidecar profile) | Critical | G1 (K2) | Single command works | Clean-machine test |
| REQ-DEP-002 | The corpus SHALL be mounted as a volume (`./corpus`), and the index built on start if no cache exists | Critical | Held-out corpus swap | Swapping the corpus folder changes the index hash and answers | Test |
| REQ-DEP-003 | Model weights SHALL be downloaded at image build time, not at run time | High | Offline-safe runs | Container runs with network disabled (extractive or local backend) | Test |
| REQ-DEP-004 | Secrets SHALL come only from environment or `.env`, never committed. `.env.example` provided. | Critical | Hygiene | Secret scan passes | gitleaks-style check |
| REQ-DEP-005 | No GPU dependency. CPU-only wheels. | Critical | Clean machine [P1 §5.2] | Image contains no CUDA libraries | Image inspection |
| REQ-DEP-006 | Images SHALL build for amd64 and arm64 (or build on host from the Dockerfile) | High | Dev is arm64; judges likely amd64 | Both builds pass | CI matrix |
| REQ-DEP-007 | A demo surface (CLI stream + minimal SSE web view of the event timeline) SHALL be provided | Medium | Demo video [G§8 p6] | Demo script works | Manual |
| REQ-DEP-008 | The submission commit SHALL be tagged `PRISM_GENAI_HACKATHON_Y2026` and contain every referenced artifact | Critical | [D s13] | Tag exists; link check passes | Release checklist |

---

## 5. Input Event Specification

### 5.1 Derivation principles

1. **Only what a real streaming front-end could know may be an input.** The system must *infer* intents, turn types, refinements and retrieval needs. If the harness supplied labels such as `is_refinement` or `intents`, that would leak gold answers. So these are **not** input fields.
2. **Timestamps follow the guide.** `timestamp_s` is relative to the start of the utterance (Ex1: 0.0, 0.8, 1.6, end 2.1) [G§4 p3–4].
3. **Utterance end is a separate event, not a flag on a chunk.** The guide's end marker arrives 0.5 s after the last chunk and carries no text [G§4 p3]. Endpointing is a timing fact, not a property of text.
4. **ASR stability is a different concept from end of utterance.** A chunk may be a *partial* hypothesis that a later chunk replaces. The guide's chunks are final and append-only (Q9), so `stability` defaults to `final`.
5. **No outside state injection.** "Previous session state", "retrieved evidence", "user intent" and "late-arriving details" are **not** input types. State and evidence are internal (C4). Late details are just later chunks or turns that the system must recognize [G§4 p4].

### 5.2 Envelope (all inputs)

```json
{
  "schema_version": "1.0",
  "type": "TRANSCRIPT_CHUNK",
  "session_id": "sess-0001",
  "utterance_id": "u2",
  "payload": {}
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema_version` | string | yes | Input contract version; rejected if the major version is unknown |
| `type` | enum | yes | `SESSION_START`, `TRANSCRIPT_CHUNK`, `UTTERANCE_END`, `SESSION_END` |
| `session_id` | string (1–128, `[A-Za-z0-9._:-]`) | yes | Isolation namespace. All state hangs off it. Never reused across runs. |
| `utterance_id` | string | yes for chunk and end | Unique within the session. Order is defined by the first-seen chunk. |
| `payload` | object | yes | Type-specific (below) |

### 5.3 `SESSION_START` (optional; auto-created on the first chunk if absent)

```json
{"schema_version":"1.0","type":"SESSION_START","session_id":"sess-0001",
 "payload":{"locale":"en","client":"replay","session_started_wall":"2026-10-02T10:00:00Z"}}
```

| Field | Meaning |
|---|---|
| `locale` | Language hint (default `en`). Informational unless multilingual is enabled. |
| `client` | `replay`, `interactive` or `asr`. Informational. |
| `session_started_wall` | Optional ISO time, for logs only. Never used in logic (determinism). |

There are no fields for user identity or profile, by design (C4).

### 5.4 `TRANSCRIPT_CHUNK`

```json
{"schema_version":"1.0","type":"TRANSCRIPT_CHUNK","session_id":"sess-0001","utterance_id":"u1",
 "payload":{"chunk_index":1,"timestamp_s":0.8,"text":"Pune for 30 people, and I need",
            "stability":"final","replaces_chunk_index":null,"utterance_offset_s":12.4}}
```

| Field | Type | Required | Meaning / rules |
|---|---|---|---|
| `chunk_index` | int ≥ 0 | yes | Position within the utterance. The idempotency key together with `utterance_id`. |
| `timestamp_s` | float ≥ 0 | yes | Arrival time relative to the start of the utterance (guide semantics). Must be non-decreasing in `chunk_index`; a violation is logged and the chunk accepted. |
| `text` | string (may be empty) | yes | **Delta** text: new words only, not cumulative. Leading and trailing whitespace is normalized. Empty text is allowed (a keep-alive). |
| `stability` | enum `final` \| `partial` | no (default `final`) | `partial` = an ASR hypothesis that may be replaced. Controller rules treat partial text as OPEN. |
| `replaces_chunk_index` | int \| null | no | ASR revision: this chunk replaces the earlier chunk with that index. Segments are recomputed. |
| `utterance_offset_s` | float \| null | no | Start of the utterance relative to the start of the session. Used only for session-level timelines. |

A replay file MAY carry cumulative transcripts instead, if declared in its header (`"text_mode":"cumulative"`). The Transcript Streamer converts them to deltas.

### 5.5 `UTTERANCE_END`

```json
{"schema_version":"1.0","type":"UTTERANCE_END","session_id":"sess-0001","utterance_id":"u1",
 "payload":{"timestamp_s":2.1,"reason":"endpoint","last_chunk_index":2}}
```

| Field | Meaning |
|---|---|
| `timestamp_s` | End of the utterance on the same clock as the chunks. **The reference point for G2 and TTFT.** |
| `reason` | `endpoint` (VAD or harness), `explicit` (user pressed send), `timeout` (system-generated, REQ-STREAM-008), `eof` (stream closed) |
| `last_chunk_index` | Lets the system detect missing chunks. Gaps are logged as `ERROR(recoverable)`. |

### 5.6 `SESSION_END`

```json
{"schema_version":"1.0","type":"SESSION_END","session_id":"sess-0001","payload":{"reason":"client_closed"}}
```

On `SESSION_END` the system finalizes any in-flight turn, emits `SESSION_CLOSED`, and destroys the state (REQ-SESS-002).

### 5.7 Load-time inputs (not streamed)

| Input | Structure | Notes |
|---|---|---|
| **Corpus** | Folder of documents (txt, md, pdf, docx, html, json) | Native `Doc_ID` and section markers are preserved when present (§10.1) |
| **CorpusManifest** (derived) | `{corpus_id, corpus_hash, n_docs, n_sections, n_chunks, chunker_version, embedder_revision, built_wall}` | Written by the indexer; included in every run manifest |
| **SystemConfig** | YAML: backends, thresholds, budgets, prompts (templates), price table, verbosity | Hash recorded (REQ-REPRO-007). Contains no benchmark strings. |
| **Replay file** | JSONL of input events, optional header line `{"replay_header":{"text_mode":"delta","clock":"real","speed":1.0}}` | Produced by the eval harness (§22) |

### 5.8 Why the example schema in the brief was changed

| Brief field | Decision | Reason |
|---|---|---|
| `timestamp` | Renamed `timestamp_s`, utterance-relative | Matches the guide's examples and output record [G§4 p4] |
| `is_final` | **Split** into `stability` (ASR hypothesis) and a separate `UTTERANCE_END` event | The two concepts differ. The guide's end marker is its own timed event. |
| `chunk_index` | Kept; part of the idempotency key | Dedup and reordering |
| (new) `replaces_chunk_index` | Added | ASR revisions (Q9) |
| (new) `utterance_offset_s` | Added | Session-level timelines without breaking utterance-relative G2 |

---

## 6. Output Event Specification

### 6.1 Design

**Every output is an event on one bus.** "Telemetry" is not a separate event type. Every event *is* telemetry, written by the Telemetry Manager as JSONL. Each event serves both as a pipeline signal (consumed by downstream components) and as a trace record (consumed by the eval harness and the demo view).

**Envelope:**

```json
{
  "schema_version": "1.0",
  "event_id": "sess-0001:000042",
  "type": "RETRIEVAL_STARTED",
  "session_id": "sess-0001",
  "utterance_id": "u1",
  "seq": 42,
  "t_stream_s": 0.8,
  "t_wall_ms": 15234.7,
  "component": "retrieval_controller",
  "payload": {}
}
```

| Field | Meaning |
|---|---|
| `event_id` | Deterministic `session_id:seq`. No random UUIDs (REQ-REPRO-004). |
| `seq` | Per-session monotonic counter. It gives a total order inside a session. |
| `t_stream_s` | Stream time relative to the utterance (null for session-level events) |
| `t_wall_ms` | Monotonic milliseconds since the run started (`perf_counter`), for latency |
| `component` | Producer component name (§18) |

### 6.2 Minimum event vocabulary (16 types)

The brief's suggested names map onto this vocabulary as follows:

| Brief's name | Covered by | Why merged |
|---|---|---|
| `QUERY_DECOMPOSED` | `INTENTS_UPDATED` | Intents change several times per utterance; one event type with a `source` field covers all cases |
| `EVIDENCE_RETRIEVED` | `RETRIEVAL_COMPLETED` | One retrieval = one start/complete pair, which makes latency trivial to compute |
| `ANSWER_PARTIAL` | `ANSWER_DELTA` | — |
| `ANSWER_FINAL`, `ANSWER_REFINED` | `ANSWER_COMMITTED` with `kind` | One commit path keeps lineage linear |
| `CITATION_ADDED`, `UNCERTAINTY_DETECTED` | `GROUNDING_CHECKED` + `ANSWER_COMMITTED.diff` | Citations and uncertainty are properties of a validated version, not free-floating events |
| `TELEMETRY_EVENT` | The envelope itself | — |

| # | Type | Purpose | Producer | Consumers |
|---|---|---|---|---|
| 1 | `CHUNK_RECEIVED` | Acknowledge, dedup and order an input chunk | Chunk Manager | Controller, Intent Analyzer, Telemetry |
| 2 | `CONTROLLER_DECISION` | WAIT / RETRIEVE / NO_RETRIEVE per segment, with reasons | Retrieval Controller | Decomposer, Orchestrator, Telemetry, demo |
| 3 | `INTENTS_UPDATED` | Current intent set (rule, LLM or reconciled) | Query Decomposer | Retrieval dispatch, Fusion, Synthesizer, Telemetry |
| 4 | `RETRIEVAL_STARTED` | A search was issued (the guide's `retrieval_events`) | Retrieval dispatcher | Telemetry, eval (G2) |
| 5 | `RETRIEVAL_COMPLETED` | Candidates and latency for one search | Retriever | Reranker, Fusion, Telemetry |
| 6 | `RETRIEVAL_SKIPPED` | Retrieval needed but satisfied (ledger hit) or blocked (budget) | Ledger / dispatcher | Telemetry, eval (G5) |
| 7 | `EVIDENCE_RERANKED` | Per-intent rerank (cross-encoder or RRF+dedup) | Reranker | Fusion, Telemetry |
| 8 | `EVIDENCE_FUSED` | Final budgeted evidence set for synthesis | Evidence Fusion | Synthesizer, Grounding, Telemetry |
| 9 | `SYNTHESIS_STARTED` | Synthesis began (any backend) | Answer Synthesizer | Telemetry (TTFT start reference) |
| 10 | `ANSWER_DELTA` | Streamed validated sentence(s) or raw tokens | Synthesizer / Grounding | Client, Telemetry |
| 11 | `LLM_CALL` | Usage and cost for one LLM request | LLM Gateway | Telemetry, eval (cost) |
| 12 | `GROUNDING_CHECKED` | Claim verdicts, dropped labels, uncertainty | Grounding Validator | Citation Manager, Session, Telemetry |
| 13 | `ANSWER_COMMITTED` | Validated answer version with lineage and diff | Session State Manager | Client, Telemetry, eval (G4, G5) |
| 14 | `TURN_COMPLETED` | Guide-format per-turn record (superset) | Orchestrator | Client, eval |
| 15 | `ERROR` | A failure and the degradation applied | Any | Telemetry, Orchestrator |
| 16 | `SESSION_CLOSED` | Session teardown with counters | Session State Manager | Telemetry |

### 6.3 Payload specifications and examples

The examples use placeholder evidence IDs (`Doc_A §1`). They are not corpus facts.

**1. `CHUNK_RECEIVED`**

Required fields: `chunk_index`, `timestamp_s`, `status` ∈ {accepted, duplicate_ignored, revision, out_of_order_accepted, rejected}, `cumulative_chars`.

```json
{"type":"CHUNK_RECEIVED","payload":{"chunk_index":1,"timestamp_s":0.8,"status":"accepted","cumulative_chars":63}}
```

**2. `CONTROLLER_DECISION`**

Required fields: `segment_id`, `decision`, `reasons[]`, `trigger` (when RETRIEVE), `segment_state`, `features{}`, `latency_ms`, `turn_type_hypothesis`.

```json
{"type":"CONTROLLER_DECISION","payload":{"segment_id":"u1.s1","decision":"RETRIEVE","trigger":"provisional",
 "reasons":["clause_closed","anchor_present","novel"],"segment_state":"CLOSED","turn_type_hypothesis":"query",
 "features":{"content_tokens":6,"anchor_strength":2.7,"dangling":false,"stability":1.0,"novelty":1.0,"act":"INFO_REQUEST"},
 "latency_ms":1.9}}
```

**3. `INTENTS_UPDATED`**

Required fields: `intent_set_version`, `source` ∈ {rule, llm, reconciled}, `intents[]` (the §9.2 schema, abbreviated), `merged[]`, `dropped[]`.

```json
{"type":"INTENTS_UPDATED","payload":{"intent_set_version":2,"source":"rule",
 "intents":[{"intent_id":"I1","text":"venue for 30 attendees in Pune","status":"PROVISIONAL"},
            {"intent_id":"I2","text":"cancellation policy","status":"COMMITTED"},
            {"intent_id":"I3","text":"catering options","status":"COMMITTED"}],
 "merged":[],"dropped":[]}}
```

**4. `RETRIEVAL_STARTED`** (maps to the guide's `retrieval_events[]`)

Required fields: `retrieval_id`, `timestamp_s`, `query`, `trigger` ∈ {provisional, multi_intent, final, refinement}, `intent_ids[]`, `filters{}`, `scope` ∈ {corpus, session_docs}.

```json
{"type":"RETRIEVAL_STARTED","payload":{"retrieval_id":"u1.r2","timestamp_s":1.6,
 "query":"cancellation policy workshop venue pune","trigger":"multi_intent","intent_ids":["I2"],"filters":{},"scope":"corpus"}}
```

**5. `RETRIEVAL_COMPLETED`**

Required fields: `retrieval_id`, `status` ∈ {ok, partial, timeout, error}, `latency_ms{embed,lexical,dense,fuse,total}`, `results[]` (chunk_id, lexical_rank, dense_rank, rrf_score), `n_candidates`.

```json
{"type":"RETRIEVAL_COMPLETED","payload":{"retrieval_id":"u1.r2","status":"ok",
 "latency_ms":{"embed":4.1,"lexical":0.3,"dense":0.9,"fuse":0.2,"total":5.8},"n_candidates":50,
 "results":[{"chunk_id":"Doc_A§4#1","lexical_rank":1,"dense_rank":3,"rrf_score":0.0323}]}}
```

**6. `RETRIEVAL_SKIPPED`**

Required fields: `reason` ∈ {ledger_hit, budget_exhausted, suppressed}, `intent_ids[]`, `ledger_ref` (retrieval_id reused), `similarity`.

```json
{"type":"RETRIEVAL_SKIPPED","payload":{"reason":"ledger_hit","intent_ids":["I1"],"ledger_ref":"u1.r1","similarity":0.97}}
```

**7. `EVIDENCE_RERANKED`**

Required fields: `intent_id`, `method` ∈ {cross_encoder, rrf_dedup}, `n_candidates`, `latency_ms`, `top[]` (chunk_id, score), `covered` (bool), `best_score`.

```json
{"type":"EVIDENCE_RERANKED","payload":{"intent_id":"I2","method":"cross_encoder","n_candidates":20,"latency_ms":112.0,
 "top":[{"chunk_id":"Doc_A§4#1","score":7.2}],"covered":true,"best_score":7.2}}
```

**8. `EVIDENCE_FUSED`**

Required fields: `evidence_set_id`, `items[]` (label, evidence_id, citation, intent_ids), `per_intent{}` (covered, evidence_ids), `conflicts[]`, `dedup_dropped`, `token_count`.

```json
{"type":"EVIDENCE_FUSED","payload":{"evidence_set_id":"u1.E","token_count":1420,"dedup_dropped":3,
 "items":[{"label":"E1","evidence_id":"Doc_A§4#1","citation":"Doc_A §4","intent_ids":["I2"]}],
 "per_intent":{"I1":{"covered":true,"evidence_ids":["Doc_B§2#1"]},"I3":{"covered":false,"evidence_ids":[]}},
 "conflicts":[]}}
```

**9. `SYNTHESIS_STARTED`**

Required fields: `mode` ∈ {initial, refinement, presentation, meta, social, clarification}, `backend` ∈ {hosted, local, extractive}, `evidence_set_id`, `prompt_tokens_est`, `kept_claims` (refinement).

```json
{"type":"SYNTHESIS_STARTED","payload":{"mode":"initial","backend":"hosted","evidence_set_id":"u1.E","prompt_tokens_est":1900,"kept_claims":0}}
```

**10. `ANSWER_DELTA`**

Required fields: `delta_seq`, `text`, `validated` (bool), `claim_ids[]`. The first delta defines TTFT.

```json
{"type":"ANSWER_DELTA","payload":{"delta_seq":0,"text":"Venue options for 30 attendees are listed in the venue guide [Doc_B §2].","validated":true,"claim_ids":["C1"]}}
```

**11. `LLM_CALL`**

Required fields: `call_id`, `purpose` ∈ {synthesis, decompose_check, presentation, meta, social, controller_ablation}, `backend`, `model`, `tokens_in`, `tokens_out`, `cached_tokens`, `ttft_ms`, `latency_ms`, `cost_usd`, `status` ∈ {ok, error, timeout, cancelled}.

```json
{"type":"LLM_CALL","payload":{"call_id":"u1.L1","purpose":"synthesis","backend":"hosted","model":"claude-haiku-4-5",
 "tokens_in":1934,"tokens_out":212,"cached_tokens":0,"ttft_ms":0.0,"latency_ms":0.0,"cost_usd":0.0,"status":"ok"}}
```

Numeric values here are placeholders; they are not measurements.

**12. `GROUNDING_CHECKED`**

Required fields: `claims[]` (claim_id, verdict ∈ {SUPPORTED, PARTIAL, UNSUPPORTED, INVALID_CITATION}, checks), `dropped_labels[]`, `uncertainty[]`, `policy_actions[]`.

```json
{"type":"GROUNDING_CHECKED","payload":{"claims":[{"claim_id":"C1","verdict":"SUPPORTED","checks":{"labels_valid":true,"numbers_ok":true,"support":0.71}}],
 "dropped_labels":[],"uncertainty":[{"intent_id":"I3","kind":"no_evidence","aspect":"catering options"}],"policy_actions":[]}}
```

**13. `ANSWER_COMMITTED`**

Required fields: `version`, `parent_version`, `topic_id`, `kind` ∈ {initial, refinement, presentation, meta, social, clarification}, `text`, `claims[]`, `citations[]`, `uncertainty[]`, `diff{}`, `delta_queries[]`, `full_rerun`.

```json
{"type":"ANSWER_COMMITTED","payload":{"version":2,"parent_version":1,"topic_id":"T1","kind":"refinement",
 "text":"…","citations":["Doc_A §1","Doc_C §3"],
 "claims":[{"claim_id":"C1","op":"kept"},{"claim_id":"C4","op":"added"}],
 "uncertainty":[],"diff":{"kept":["C1","C2"],"modified":[],"added":["C4"],"retracted":[],
 "citations_added":["Doc_C §3"],"citations_removed":[],"uncertainty_resolved":[],"uncertainty_introduced":[]},
 "delta_queries":["u2.r1","u2.r2"],"full_rerun":false}}
```

**14. `TURN_COMPLETED`**

This is the guide's record (§6.4) plus metrics. Required fields: everything in §6.4.

**15. `ERROR`**

Required fields: `component`, `error_class`, `recoverable`, `action` ∈ {retry, fallback_backend, lexical_only, rrf_only, extractive, skip_stage, abort_turn}, `detail`.

```json
{"type":"ERROR","payload":{"component":"reranker","error_class":"Timeout","recoverable":true,"action":"rrf_only","detail":"rerank > 600 ms"}}
```

**16. `SESSION_CLOSED`**

Required fields: `reason`, `turns`, `versions`, `llm_calls`, `tokens_in`, `tokens_out`, `cost_usd`, `retrievals`.

### 6.4 `TURN_COMPLETED`: the guide-compatible record

The keys `retrieval_events`, `sub_queries`, `answer`, `citations` and `uncertainty` are **exactly** the guide's [G§4 p4]. All other keys are additive.

```json
{
  "session_id": "sess-0001", "utterance_id": "u1", "turn_type": "query",
  "retrieval_required": true, "reason": null,
  "retrieval_events": [
    {"timestamp_s": 0.8, "query": "customer workshop pune 30 people", "trigger": "provisional", "retrieval_id": "u1.r1", "intent_ids": ["I1"]},
    {"timestamp_s": 1.6, "query": "cancellation policy workshop venue pune", "trigger": "multi_intent", "retrieval_id": "u1.r2", "intent_ids": ["I2"]},
    {"timestamp_s": 1.6, "query": "catering options workshop pune", "trigger": "multi_intent", "retrieval_id": "u1.r3", "intent_ids": ["I3"]}
  ],
  "sub_queries": ["venue for 30 attendees in Pune", "cancellation terms and refund policy", "catering options"],
  "answer": "…",
  "citations": ["Doc_B §2", "Doc_A §4"],
  "uncertainty": "Catering options could not be verified from the retrieved corpus.",
  "intent_status": {"I1": "answered", "I2": "answered", "I3": "uncertain"},
  "answer_version": {"version": 1, "parent_version": null, "kind": "initial", "topic_id": "T1"},
  "controller_summary": {"decisions": 4, "wait": 2, "retrieve": 2, "no_retrieve": 0},
  "timings": {"utterance_end_s": 2.1, "first_retrieval_s": 0.8, "lead_time_s": 1.3,
              "evidence_ready_s": 1.95, "ttft_ms": null, "answer_commit_ms": null},
  "usage": {"llm_calls": 1, "tokens_in": null, "tokens_out": null, "cost_usd": null, "backend": "hosted"},
  "grounding": {"claims": null, "supported": null, "invalid_ids_dropped": 0}
}
```

`uncertainty` is a string, as in the guide; it is null when there is nothing to report. The structured list lives in `ANSWER_COMMITTED.uncertainty[]`. The null metric values show where measurements go; they are not results.

---

## 7. System State Machine

### 7.1 Why not a flat state list

The brief's candidate states (LISTENING, EARLY_RETRIEVAL, DECOMPOSING, RETRIEVING, …) are **not mutually exclusive** in a full-duplex system. At t = 1.6 s in Example 1 the system is simultaneously:

- *listening* (the utterance hasn't ended);
- *decomposing* (new intents);
- *retrieving* (I2 and I3);
- *reranking* (I1).

A flat machine would either lose that concurrency or blow up into a product of states.

**Design:** a **hierarchical statechart** with concurrent (orthogonal) regions.

| Level | Region(s) | Instances |
|---|---|---|
| L0 | Session lifecycle | 1 per session |
| L1 | Turn lifecycle | ≤1 *listening* turn + ≤1 *answering* turn per session at a time |
| L2 | Inside `LISTENING`: Segment region ∥ Intent region ∥ Retrieval-job region | Many per turn |

The brief's states are all represented (mapping in §7.7).

### 7.2 L0: Session states

| State | Entry condition | Exit condition | Allowed transitions | Actions | Data available | Failure behavior |
|---|---|---|---|---|---|---|
| `NEW` | `SESSION_START` or first chunk for an unknown `session_id` | State allocated | → `ACTIVE` | Allocate the `SessionState` actor and queue | session_id | Allocation failure → `ERROR` event; reject the session's events |
| `ACTIVE` | Allocated | `SESSION_END`, TTL expiry, fatal error | → `CLOSING` | Route events to the turn machine | Full session state | Invariant violation → `DEGRADED` flag (§23.9); stays ACTIVE |
| `CLOSING` | End or TTL | In-flight turn committed or aborted (≤ `close_grace_s`) | → `CLOSED` | Finalize the turn, flush telemetry | Full | Grace timeout → abort turn, `ERROR` |
| `CLOSED` | Teardown done | — | (terminal) | Emit `SESSION_CLOSED`, drop all state | Counters only | — |

### 7.3 L1: Turn states

| State | Entry condition | Exit condition | Allowed transitions | Actions | Data available | Failure behavior |
|---|---|---|---|---|---|---|
| `IDLE` | Session active; no utterance open (between turns, or "waiting for late detail") | First chunk of a new `utterance_id` | → `LISTENING` | None. The topic frame stays open for possible refinement. | Prior answers, frame, ledger, evidence store | — |
| `LISTENING` | First chunk | `UTTERANCE_END` (or endpoint timeout) | → `FINALIZING` | Run L2 regions per chunk: segment, classify, decide, decompose, retrieve, rerank-on-stability | Accumulated transcript, segments, intents, provisional evidence | Component failures degrade per §23. Never blocks listening. |
| `FINALIZING` | Utterance ended | Turn plan ready | → `SYNTHESIZING` \| `TRANSFORMING` \| `CLARIFYING` \| `IDLE` (backchannel) | Close all segments; final controller pass; gated LLM decomposition check (await ≤ `llm_check_wait_ms`); reconcile intents; await outstanding retrieval and rerank (≤ `finalize_wait_ms`); fuse evidence; decide turn plan | Final intents, evidence set, turn type | Waits exceeded → proceed with what's available (log `ERROR(recoverable)`) |
| `SYNTHESIZING` | Plan = answer (query or refinement) | Generation done | → `VALIDATING` | Build the prompt (labels; kept claims for refinement); stream through the sentence gate | Evidence set, kept claims | LLM failure → fallback chain (§23.4); last resort extractive |
| `TRANSFORMING` | Plan = presentation, meta or social | Transform done | → `VALIDATING` | LLM transform over prior claims only. **No retrieval.** | Prior version claims and citations | LLM failure → deterministic fallback (claims as a list) |
| `CLARIFYING` | Plan = clarification (blocking ambiguity, §16.5) | Question emitted | → `VALIDATING` | Generate one targeted question (no factual claims) | Ambiguity record | Fallback: a generic clarification prompt from config |
| `VALIDATING` | Generation or transform complete | Validation complete | → `COMMITTED` | Final grounding pass; citation mapping; uncertainty assembly; diff | Claims, verdicts | Validator failure → commit as `unverified` with uncertainty (never as verified) |
| `COMMITTED` | Version stored | Telemetry emitted | → `IDLE` (or → `FINALIZING` of a queued turn) | Emit `ANSWER_COMMITTED` and `TURN_COMPLETED`; apply `pending_constraints` → enqueue a refinement turn | New version | — |

**Concurrency rule.** A new utterance MAY enter `LISTENING` while the previous turn is in `SYNTHESIZING` or `VALIDATING` (full duplex). Its early retrieval runs normally. Its `FINALIZING` waits until the previous turn reaches `COMMITTED`, so commits are serialized and lineage stays linear.

Default barge-in policy: let the current synthesis finish. If the new utterance is classified as a correction ("no wait", "actually") *and* `cancel_on_correction=true`, cancel the synthesis (`LLM_CALL.status=cancelled`) and fold the turns into one refinement.

### 7.4 L2 regions inside `LISTENING`

**Segment region** (per segment). A segment is a clause-like span of the utterance.

| State | Entry | Exit / transitions | Actions |
|---|---|---|---|
| `OPEN` | Segment created at a boundary | → `STABLE` (no new tokens for `Δt_stable` and sufficient); → `CLOSED` (boundary detected or force-close at `max_segment_tokens`); → `IGNORED` (act ∈ suppressed set) | Update features |
| `STABLE` | Quiet and sufficient | → `CLOSED` (boundary); → `OPEN` (new tokens change content materially); → `DISPATCHED` | Controller may RETRIEVE (provisional) |
| `CLOSED` | Boundary | → `DISPATCHED` \| `IGNORED` \| `MERGED` | Controller decides |
| `DISPATCHED` | Retrieval issued or ledger hit | (terminal for this segment) | Linked to intent(s) |
| `IGNORED` | Suppressed act, or empty | (terminal) | Contributes render directives (presentation) or nothing |
| `MERGED` | Folded into a neighboring segment (e.g. a fixed phrase) | (terminal) | — |

**Intent region** (per intent).

| State | Entry | Exit / transitions |
|---|---|---|
| `CANDIDATE` | Created from a segment by the decomposer | → `PROVISIONAL` (retrieval dispatched before the end of the utterance); → `COMMITTED`; → `MERGED`; → `DROPPED` |
| `PROVISIONAL` | Early retrieval done | → `COMMITTED` (survives later decomposition or final reconciliation); → `MERGED`; → `DROPPED` |
| `COMMITTED` | Confirmed in the intent set (clause closed and no conflicting reconciliation) | → `RERANKED` |
| `RERANKED` | Per-intent rerank done | → `FUSED` (at finalize) |
| `FUSED` | Included in the evidence set | → `ANSWERED` \| `UNCERTAIN` (after validation) |
| `MERGED` / `DROPPED` | Over-split merge, or not in the final set | (terminal). Its evidence stays in the store but is ineligible for citation in this turn (REQ-STREAM-004). |

**Rerank-on-stability.** An intent MAY be reranked while still `PROVISIONAL` once its segment is `CLOSED`. The cost [M] is accepted to keep rerank off the critical path (§20). If the intent is later dropped, the rerank work is wasted and is logged as such.

**Retrieval-job region** (per retrieval).

| State | Transitions |
|---|---|
| `QUEUED` | → `RUNNING` |
| `RUNNING` | → `DONE` \| `PARTIAL` (one retriever timed out) \| `FAILED` \| `CANCELLED` |
| `DONE` / `PARTIAL` | Results feed the intent's candidate pool |
| `FAILED` | §23.2 degradation |

### 7.5 Transition table (L1)

| # | From | Event / guard | To | Actions |
|---|---|---|---|---|
| 1 | IDLE | `TRANSCRIPT_CHUNK` (new utterance) | LISTENING | Create the turn and its segment tracker; `CHUNK_RECEIVED` |
| 2 | LISTENING | `TRANSCRIPT_CHUNK` | LISTENING | Segment update → classify → decide per segment → decompose → dispatch → rerank-on-stability |
| 3 | LISTENING | `UTTERANCE_END` \| endpoint timeout \| `SESSION_END` | FINALIZING | Close segments; final decisions |
| 4 | FINALIZING | plan = query \| refinement | SYNTHESIZING | Fuse; `EVIDENCE_FUSED` |
| 5 | FINALIZING | plan = presentation \| meta \| social | TRANSFORMING | `CONTROLLER_DECISION(NO_RETRIEVE)` already logged |
| 6 | FINALIZING | plan = clarification | CLARIFYING | — |
| 7 | FINALIZING | plan = backchannel | IDLE | `TURN_COMPLETED` with no answer |
| 8 | FINALIZING | previous turn not committed | FINALIZING (wait) | Hold until the previous commit (serialization) |
| 9 | SYNTHESIZING | generation finished | VALIDATING | — |
| 10 | SYNTHESIZING | LLM error \| timeout | SYNTHESIZING (next backend) \| VALIDATING (extractive) | Fallback chain; `ERROR` |
| 11 | SYNTHESIZING | correction barge-in ∧ `cancel_on_correction` | FINALIZING (merged turn) | Cancel the LLM; merge the utterances |
| 12 | TRANSFORMING \| CLARIFYING | done | VALIDATING | — |
| 13 | VALIDATING | ok | COMMITTED | Store the version |
| 14 | VALIDATING | validator failure | COMMITTED (`unverified=true`) | Uncertainty; `ERROR` |
| 15 | COMMITTED | `pending_constraints` non-empty | FINALIZING (synthetic refinement turn) | Apply queued constraints |
| 16 | COMMITTED | otherwise | IDLE | `TURN_COMPLETED` |
| 17 | any | `SESSION_END` | (session CLOSING) | Finalize or abort within the grace period |

### 7.6 ASCII state diagram

```
                                   L0  NEW ──► ACTIVE ──────────────────────────────► CLOSING ──► CLOSED
                                                 │
 L1 (turn)  ┌──────────────────────────────────── ▼ ───────────────────────────────────────────────────┐
            │  IDLE ──chunk(new utt)──► LISTENING ──utterance_end / timeout──► FINALIZING              │
            │   ▲                         │  ┌──────────── L2 concurrent regions ─────────────┐ │   │   │
            │   │                         │  │ Segment: OPEN⇄STABLE→CLOSED→DISPATCHED|IGNORED │ │   │   │
            │   │                         │  │ Intent:  CANDIDATE→PROVISIONAL→COMMITTED→      │ │   │   │
            │   │                         │  │          RERANKED→FUSED (or MERGED/DROPPED)    │ │   │   │
            │   │                         │  │ RetrievalJob: QUEUED→RUNNING→DONE|PARTIAL|FAIL │ │   │   │
            │   │                         │  └────────────────────────────────────────────────┘ │   │   │
            │   │        plan=backchannel ◄──────────────────────────────────────────────────────┘   │   │
            │   │                         plan=query/refine ─► SYNTHESIZING ─┐                       │   │
            │   │                         plan=present/meta ─► TRANSFORMING ─┼─► VALIDATING ─► COMMITTED
            │   │                         plan=clarify ──────► CLARIFYING ───┘                    │
            │   └──────────────────────────────── (no pending constraints) ◄─────────────────────┤
            │                       FINALIZING(synthetic refinement) ◄── pending constraints ◄───┘
            └─────────────────────────────────────────────────────────────────────────────────────────┘
   Full duplex: the next utterance may be LISTENING while this turn is SYNTHESIZING/VALIDATING; its FINALIZING
   waits for COMMITTED (serialized commits ⇒ linear version lineage).
```

A Mermaid version is in `docs/architecture/06_state_machine.md`.

### 7.7 Mapping of the brief's suggested states

| Brief state | Where it lives in this design |
|---|---|
| IDLE | L1 `IDLE` |
| LISTENING | L1 `LISTENING` |
| PARTIAL_UTTERANCE | Segment `OPEN` |
| WAITING_FOR_STABILITY | Segment `OPEN` with sufficiency met but not yet `STABLE` (controller WAIT, `reason=awaiting_stability`) |
| EARLY_RETRIEVAL | Intent `PROVISIONAL` + RetrievalJob `RUNNING` during `LISTENING` |
| DECOMPOSING | Intent region transitions (rule) and the LLM check during `FINALIZING` |
| RETRIEVING | RetrievalJob `RUNNING` |
| FUSING | `FINALIZING` (fusion step) |
| SYNTHESIZING / ANSWERING | `SYNTHESIZING` (answering = streaming validated deltas) |
| WAITING_FOR_LATE_DETAIL | `IDLE` with an open topic frame. No special state: any next utterance may be a refinement. |
| REFINING | Turn plan = refinement through `FINALIZING → SYNTHESIZING` (a mode, not a state) |
| COMPLETED | `COMMITTED` → `IDLE` |
| ERROR | Not a state. Errors are events with explicit degradation transitions (§23). Only an unrecoverable session error leads to `CLOSING`. |

---

## 8. Retrieval Controller

### 8.1 Responsibilities and boundaries

The controller answers one question per open segment, per tick (each chunk and the end of the utterance): **should this segment's content be searched now, later, or never?**

It does **not**:

- build queries (that's the Decomposer);
- search (Retriever);
- classify dialog acts (Intent Analyzer).

It consumes their features. It contains **no LLM** in the primary configuration (ADR-004).

### 8.2 Inputs

| Input | Source | Fields used |
|---|---|---|
| Segment view | Chunk Manager | text, state, token counts, last-update stream time, boundary type, dangling flag |
| Act and slots | Intent Analyzer | `act` ∈ {INFO_REQUEST, CONSTRAINT, CONTEXT, PRESENTATION, SOCIAL, BACKCHANNEL, META, CORRECTION}, `act_confidence`, anchors with IDF, slots |
| Session view (read-only snapshot) | Session State Manager | answers non-empty?, frame (topic, slots), ledger (queries + embeddings), budget counters |
| Tick | Orchestrator | `t_stream_s`, kind ∈ {chunk, utterance_end, stability_timer} |

### 8.3 Signals (precise definitions)

| Signal | Definition | Range | Initial threshold (calibrate) |
|---|---|---|---|
| **Retrieval-worthiness** `W` | Derived from `act`: INFO_REQUEST → 1; CONSTRAINT (with an open topic frame or a sibling intent) → 1; CONTEXT with a need cue ("I need", "planning", "looking for") → 1; PRESENTATION / SOCIAL / BACKCHANNEL / META → 0; CORRECTION → inherits the corrected segment | {0, 1} + `act_confidence` ∈ [0,1] | Suppress only if `act_confidence ≥ 0.8` |
| **Content sufficiency** `S` | `content_tokens ≥ 2` (after filler and stopword removal) ∧ ≥1 anchor | bool | `min_content_tokens = 2` |
| **Anchor strength** `A` | Max over segment tokens and bigrams of corpus IDF `log(N/df)` for terms present in the corpus vocabulary (dictionary lookup, no search) | [0, log N] | Anchor if `IDF ≥ idf_floor` (initial 1.0) |
| **Completeness** `C` | `CLOSED` if a boundary was detected: sentence-final punctuation; comma/semicolon + coordinator; coordinator + new request head ("and I need", "also", "plus", "and what"); question word starting a new clause; `UTTERANCE_END`. Dangling = last token ∈ {preposition, article, determiner, conjunction, auxiliary, "I need"-type heads}. | {OPEN, STABLE, CLOSED} | — |
| **Stability** `St` | `STABLE` if no new tokens for `Δt_stable` **or** content-term Jaccard between consecutive ticks ≥ 0.8 for 2 ticks, and not dangling | bool | `Δt_stable = 0.6 s` |
| **Novelty** `N` | `1 − max sim(q_seg, q_ledger)` where sim = max(term Jaccard, cosine of query embeddings) | [0,1] | Ledger hit if `cos ≥ τ_dup` (initial 0.92) **or** Jaccard ≥ 0.8 |
| **Budget** `B` | `provisional_issued < provisional_budget` ∧ `t − t_last_dispatch(segment) ≥ min_interval` | bool | budget 4 per utterance; `min_interval = 0.4 s` |
| **Elapsed / length guard** | Force-close a segment after `max_segment_tokens` content tokens without a boundary (handles unpunctuated ASR) | — | 25 |

Notes:

- ASR transcripts may have **no punctuation**. Boundary detection must not rely on it; lexical coordinators and request heads are primary.
- All lexicons (fillers, request heads, coordinators, presentation verbs) are **generic English**, live in config, and are disclosed (REQ-REPRO-006/007).

### 8.4 Turn-type hypothesis

The controller maintains a running turn-type hypothesis from the segment acts:

- **query**: any INFO_REQUEST, CONTEXT+need, or new-topic CONSTRAINT.
- **refinement**: CONSTRAINT or CORRECTION segments with topic continuity to `frame.topic` (anaphora such as "the trip", "it", "that"; slot overlap; topic similarity ≥ `τ_topic`) and no new-topic INFO_REQUEST.
- **presentation**: all non-filler segments are PRESENTATION and `answers ≠ ∅`.
- **meta / social / backchannel**: all segments are of that act.
- **mixed**: refinement or query segments plus PRESENTATION segments. Handled as query or refinement, with render directives.

### 8.5 Decision procedure

```
procedure DECIDE(segment g, tick, session view V):
    if g.act ∈ {PRESENTATION, SOCIAL, BACKCHANNEL, META} and g.act_confidence ≥ 0.8:
        if g.act == PRESENTATION and V.answers == ∅: note "no_prior_answer"          # still no retrieval
        return NO_RETRIEVE(reason = act_reason(g.act))                              # presentation_restructure | social_ack | backchannel | meta_conversation
    if g.content_tokens == 0:                       return WAIT(reason="no_content")
    if tick.kind != utterance_end:
        if g.state == OPEN and g.dangling:          return WAIT(reason="trailing_function_word")
        if g.state == OPEN and not stable(g):       return WAIT(reason="awaiting_stability")
        if not sufficient(g):                       return WAIT(reason="low_specificity")      # no anchor yet
        if not budget_ok(g):                        return WAIT(reason="budget_or_interval")
    # here: CLOSED/STABLE and sufficient, or end of utterance
    if W(g) == 0 and g.act_confidence < 0.8:        # uncertain act → be conservative toward retrieval at end
        if tick.kind != utterance_end:              return WAIT(reason="act_uncertain")
    q ← DECOMPOSER.query_for(g)                     # may yield ≥1 intent queries
    if novelty(q, V.ledger) < 1 − τ_dup:            return RETRIEVE_SKIPPED(reason="ledger_hit")  # reuse
    trigger ← refinement   if V.turn_type_hypothesis == refinement
              final        if tick.kind == utterance_end and nothing dispatched yet for g
              multi_intent if DECOMPOSER yields >1 intent or sibling intents exist
              provisional  otherwise
    return RETRIEVE(trigger)
```

At `UTTERANCE_END`, every segment with `W=1` (or an uncertain act) that is not yet dispatched is decided with the guards relaxed. This guarantees **no information request goes unretrieved** (K4).

### 8.6 Reason codes (closed vocabulary)

| Decision | Reason codes |
|---|---|
| WAIT | `trailing_function_word`, `awaiting_stability`, `low_specificity`, `no_content`, `budget_or_interval`, `act_uncertain`, `awaiting_previous_commit` |
| RETRIEVE | `clause_closed`, `segment_stable`, `utterance_end`, `anchor_present`, `novel`, `constraint_delta`, `forced_close_length` |
| NO_RETRIEVE | `presentation_restructure`, `social_ack`, `backchannel`, `meta_conversation`, `no_prior_answer` (with presentation) |
| (skip) | `ledger_hit`, `budget_exhausted` (emitted as `RETRIEVAL_SKIPPED`) |

### 8.7 How the controller prevents the known failure modes

| Failure | Mechanism |
|---|---|
| Retrieval storms | Retrieval is dispatched on segment *events* (close or stability), never on chunk arrival; per-utterance provisional budget; minimum interval per segment; force-close only at 25 tokens |
| Noisy partial queries | Dangling detection; sufficiency (≥2 content tokens + a corpus anchor); stability timer; provisional evidence re-scored at finalize (REQ-STREAM-004) |
| Duplicate retrieval | Ledger novelty check (term Jaccard and embedding cosine); `RETRIEVAL_SKIPPED(ledger_hit)` |
| Excessive latency | Pure rules: string operations, one dictionary lookup, at most one ~4–9 ms embedding for novelty; p95 < 20 ms (REQ-PERF-001) |
| Unnecessary LLM calls | No LLM in the controller. The LLM decomposition check happens once, gated, at finalize (§9.6). |
| False triggers on presentation turns | Act classification with high-confidence suppression; requires `answers ≠ ∅` to call it presentation; per-segment decisions handle mixed turns |

### 8.8 Model-based controller (ablation arm, Exp 8)

The interface is the same as the rule controller. Only the worthiness and completeness estimation are swapped:

- **(a) Prototype classifier.** The segment embedding is compared with centroids of generic seed phrases per act (config). Cost: 1 embedding.
- **(b) Small-LLM classifier.** Structured output `{act, complete: bool, confidence}`, called on CLOSED or STABLE segments only.

Timing, budget and ledger logic stay identical, so the ablation isolates the classification quality vs latency/cost trade-off.

---

## 9. Multi-Intent Architecture

### 9.1 Pipeline overview

```
segments (from Chunk Manager) ─► Intent Analyzer (act, slots, anchors, coreference cues)
      ─► Rule Decomposer (coordination split, constraint attachment, dependency, context propagation)
      ─► Over/under-split guards ─► Normalizer (lexical + dense query forms)
      ─► Ledger dedup (reuse | new) ─► parallel dispatch (asyncio.gather, per-intent timeout)
At FINALIZING: [gated] LLM structured check ─► Reconciler ─► final IntentSet (COMMITTED/MERGED/DROPPED)
```

### 9.2 Representation: `IntentSet`

```json
{
  "utterance_id": "u1",
  "intent_set_version": 3,
  "source": "reconciled",
  "original_utterance": "<full utterance text, e.g. organizing a customer workshop in Pune for thirty people plus cancellation and catering questions>",
  "shared_context": {"slots": {"location": "Pune", "headcount": "30", "event_type": "customer workshop"}},
  "intents": [
    {
      "intent_id": "I1",
      "text": "venue for a customer workshop for 30 people in Pune",
      "lexical_query": "customer workshop venue pune 30 people",
      "dense_query": "venue for a customer workshop for 30 people in Pune",
      "constraints": [
        {"constraint_id": "K1", "slot": "headcount", "value": "30", "scope": ["I1"], "source_span": [44, 57], "op": "set"}
      ],
      "uses_shared_slots": ["location", "event_type"],
      "depends_on": [],
      "priority": 0,
      "status": "COMMITTED",
      "provenance": {"source": "rule", "segment_ids": ["u1.s1"], "span": [0, 57]},
      "confidence": 0.82,
      "ledger_refs": ["u1.r1"]
    },
    {
      "intent_id": "I2",
      "text": "cancellation policy",
      "lexical_query": "cancellation policy workshop venue pune",
      "dense_query": "cancellation policy for a workshop venue in Pune",
      "constraints": [],
      "uses_shared_slots": ["location", "event_type"],
      "depends_on": [],
      "priority": 1,
      "status": "COMMITTED",
      "provenance": {"source": "rule", "segment_ids": ["u1.s2"], "span": [70, 93]},
      "confidence": 0.9,
      "ledger_refs": ["u1.r2"]
    }
  ],
  "merged": [],
  "dropped": []
}
```

The values in this example are illustrative. The utterance paraphrases the guide's Example 1.

**Field notes.**

- `text` is the clean intent statement. It populates the guide's `sub_queries`.
- `lexical_query` and `dense_query` are the retrieval forms (§9.4). They populate `retrieval_events[].query`.
- `constraints[].scope` lists the intents the constraint applies to. `op` ∈ {set, update, retract}.
- `depends_on` holds intents whose answer is needed to interpret this one (anaphora: "how much would *that* cost").
- `priority` is answer ordering, by order of mention. It is not retrieval priority: all intents are dispatched in parallel.
- `ledger_refs` are the retrievals serving this intent (reused or new).

### 9.3 Extraction rules (rule decomposer)

| Step | Rule | Example (abstract) |
|---|---|---|
| 1. Act split | Each segment gets an act (§8.2). Only INFO_REQUEST, CONTEXT+need and CONSTRAINT produce intents or constraints. | "hey, quick one" → SOCIAL (no intent) |
| 2. Coordination split | Under one request head, coordinated noun phrases or clauses become sibling intents: `HEAD (NP₁ and NP₂ [and NP₃])` → intents NP₁, NP₂, NP₃, each carrying HEAD's context | "I need the X and the Y" → X, Y |
| 3. Constraint attachment | Subordinate or conditional clauses ("especially if", "if", "when", "in case", "only if", "for N people", "in <place>", "after/before <event>") become **constraints**, attached to the nearest preceding intent. If the clause precedes all intents or uses "for all of these", scope = all. | "X, and Y, especially if Z" → I(X), I(Y) + constraint Z on Y (confidence 0.6; flagged for LLM check if ambiguous) |
| 4. Slot extraction | Numbers with units, locations (corpus-vocabulary capitalized n-grams), dates and durations, roles. Stored in `shared_context` when stated in a CONTEXT segment, or on the intent otherwise. | "for 30 people in Pune" → headcount=30, location=Pune |
| 5. Dependency | A pronoun or demonstrative ("that", "it", "those", "them") as the object of a request, with an antecedent intent in the same utterance → `depends_on` | "…the deposit, and when is it refundable" → I2 depends_on I1 |
| 6. Context propagation | Each intent's queries get the shared slots it uses (default: all shared slots that are not contradicted). Lexical query: append slot values. Dense query: natural phrase "… for <event> in <location>". | I2 "cancellation policy" → "+ workshop venue pune" |

### 9.4 Normalization

| Form | Operations | Rationale |
|---|---|---|
| `lexical_query` | Lowercase; NFKC; remove fillers (um, uh, like, you know, I mean); remove generic stopwords; spoken numbers → digits ("thirty" → 30); keep all anchors; dedup terms | BM25 needs clean discriminative terms |
| `dense_query` | Remove fillers and disfluent repeats; keep function words and natural order; numbers → digits; prefix per the embedder's convention (e.g. a query instruction for bge, `query:` for e5) | Dense models are trained on natural sentences |

### 9.5 Deduplication and split guards

| Guard | Rule | Action |
|---|---|---|
| **Intra-set duplicate** | Two intents with `cos(dense) ≥ τ_dup` or lexical Jaccard ≥ 0.8 | Merge (keep the lower `intent_id`; union constraints) |
| **Over-split (retrieval overlap)** | Two sibling intents from one coordination whose top-5 chunk sets have Jaccard ≥ 0.6 **and** `cos ≥ τ_merge` (initial 0.85) | Merge into one intent ("X and Y" treated as one fixed phrase). Logged in `merged[]`. |
| **Under-split (cluster split)** | One intent whose top-10 results form ≥2 clusters (different doc/section, inter-cluster cos < 0.5) **and** the source span contains a coordinator or ≥2 request heads | Flag for the LLM check (gating condition G-c) |
| **Ledger dedup** | `novelty < 1 − τ_dup` vs any ledger entry in the session | Reuse that retrieval (`RETRIEVAL_SKIPPED(ledger_hit)`); add to `ledger_refs` |

All thresholds are calibrated on the tune split (REQ-MI-003).

### 9.6 LLM structured decomposition check (gated)

**Gating.** The check is invoked at `FINALIZING`, at most once per utterance (REQ-MI-008), if **any** of these hold:

| ID | Condition |
|---|---|
| G-a | ≥2 request cues (question words or request heads) but the rules produced 1 intent |
| G-b | Content tokens ≥ 30 (long utterance) |
| G-c | The under-split flag is set, or a constraint scope is ambiguous |
| G-d | Rule decomposer confidence < 0.6 (e.g. no coordinator but multiple INFO_REQUEST acts) |
| G-e | The turn-type hypothesis is uncertain between refinement and new topic |

To start the check early, it MAY be launched as soon as condition G-b or G-c becomes true during `LISTENING`. It is cancelled if the content changes materially.

**Output schema** (provider structured output; JSON Schema enforced):

```json
{
  "turn_type": "query",
  "intents": [
    {"text": "string", "search_query": "string", "constraints": [{"slot": "string", "value": "string", "applies_to": "this|all"}],
     "depends_on_index": null}
  ],
  "shared_context": {"slot_name": "value"}
}
```

**Reconciliation:**

```
procedure RECONCILE(rule_set R, llm_set L):
    for each l in L:
        r* ← argmax_r sim(l, r) over R
        if sim(l, r*) ≥ τ_match: r*.text ← r*.text (keep rule wording for stable IDs); r*.constraints ∪= l.constraints
                                 r*.provenance.source ← "reconciled"
        else: create new intent from l (status CANDIDATE) → dispatch retrieval (trigger=final)
    for each r in R not matched:
        if r.sufficiency weak and r.confidence < 0.6: r.status ← DROPPED
        else keep r                                   # conservative: rules are not overridden without cause
    emit INTENTS_UPDATED(source="reconciled")
```

Wait policy: `FINALIZING` waits for the check at most `llm_check_wait_ms` (default 300 ms) after `UTTERANCE_END`. If the check is late, synthesis proceeds with the rule intents, and the check result is logged but unused (`LLM_CALL.status=ok, used=false`).

### 9.7 Parallel dispatch

- Intents decided in the same tick are dispatched with `asyncio.gather`. Each retrieval runs in a worker thread (embedding and BLAS release the GIL), with timeout `retrieval_timeout_ms` (default 250).
- Dependent intents (`depends_on`) are dispatched **after** their antecedent's query is formed, not after its results. Their query inherits the antecedent's context, so they stay parallel in practice.

### 9.8 Worked example (abstract)

Input: *"I need information about X, and also tell me Y, especially if Z applies."*

| Element | Result |
|---|---|
| Segments | s1 = "I need information about X" (INFO_REQUEST, CLOSED by ", and also"); s2 = "tell me Y" (INFO_REQUEST); s3 = "especially if Z applies" (CONSTRAINT) |
| Intents | I1 = X; I2 = Y |
| Constraints | K1 = {slot: condition, value: Z, scope: [I2], confidence 0.6}. Ambiguous scope → G-c → LLM check MAY widen the scope to [I1, I2]. |
| Queries | I1: "x" (+ shared slots); I2: "y z" (constraint terms appended to the lexical query; dense: "Y when Z applies") |
| Dispatch | I1 at s1 closure (provisional); I2 at s2/s3 closure (multi_intent); parallel |

---

## 10. Retrieval Architecture

### 10.1 Index-time pipeline (offline, at container start or build)

```
corpus/ ─► Loader (txt|md|pdf|docx|html|json) ─► Document{doc_id, title, metadata}
        ─► Section parser (native §/headings | derived) ─► Section{doc_id, section_id, heading, text, page_span}
        ─► Chunker (section-bounded, ~200–400 tok, 15% overlap within section) ─► Chunk{chunk_id, …}
        ─► [parallel] Embedder (batch)  │  BM25 builder (scipy CSC)  │  Vocabulary/IDF table  │  Near-dup grouping
        ─► CorpusIndex (read-only arrays) + CorpusManifest{corpus_hash, …}  ─► cache/<hash>/
```

**ID rules (REQ-RET-005):**

| ID | Rule |
|---|---|
| `doc_id` | The native ID if the document declares one (filename pattern `Doc_\d+`, front-matter `id`, or a JSON field). Otherwise `Doc_<n>`, where *n* is the 1-based index of the file in **sorted relative path order**. Deterministic. |
| `section_id` | The native marker if present (`§2`, "Section 2", numbered headings `2.`, `2.1`). Otherwise the ordinal of heading-delimited blocks (`1, 2, …`). If there are no headings: page-based `p<k>` (PDF) or paragraph-group ordinals. |
| `chunk_id` | `{doc_id}§{section_id}#{part}` (part is 1-based within the section) |
| Citation key | `"{doc_id} §{section_id}"`. Several chunks of one section share a citation key. |

**Contextual header.** The text that gets embedded is `"{title} > {heading}: {chunk_text}"`. Citations still point to the chunk.

### 10.2 Query-time pipeline: stage contracts

| # | Stage | Input | Output | Notes and defaults |
|---|---|---|---|---|
| 1 | **Query normalization** | `Intent` (text, constraints, shared slots) | `QueryForms{lexical_terms[], dense_text, filters{}}` | §9.4 |
| 2 | **Embedding** | `dense_text` | `q_vec ∈ ℝ^d`, L2-normalized float32 | bge-small (d=384) by default; per-process LRU cache keyed by text (a pure function, so no behavioral state) |
| 3 | **Dense retrieval** | `q_vec`, filter mask | `[(chunk_id, cos, dense_rank)]`, top `N_dense`=50 | Exact `E·q` with numpy and `argpartition`; ties broken by `chunk_id` |
| 4 | **Lexical retrieval** | `lexical_terms`, filter mask | `[(chunk_id, bm25, lex_rank)]`, top `N_lex`=50 | BM25 k1=1.5, b=0.75; same tokenizer as index time |
| 5 | **Metadata filtering** | `filters{doc_ids?, section_ids?, meta?}` | Boolean chunk mask | Applied *before* top-N in stages 3 and 4 (exact search makes pre-filtering free) |
| 6 | **Candidate merging** | Two ranked lists | `Candidate{chunk_id, lex_rank|∞, dense_rank|∞, raw scores}` | Union by `chunk_id` |
| 7 | **Deduplication** | Candidates | Candidates with near-duplicates collapsed | Near-dup groups are precomputed at index time (text hash; cos ≥ 0.97 within group). Keep the best-ranked member and record `alternates[]`. |
| 8 | **Score normalization** | Raw scores | `norm_lex`, `norm_dense` ∈ [0,1] (min-max within the list) | **Not used by RRF.** Computed only for the weighted-fusion ablation (Exp 2) and for logging. |
| 9 | **Fusion** | Candidates | `rrf = Σ_{r∈{lex,dense}} 1/(k + rank_r)`, k=60 | Rank-based. Also fuses *multiple retrievals for the same intent* (e.g. provisional + final). |
| 10 | **Reranking** | Top `M_rerank`=20 by RRF + `dense_text` | `rerank_score` (cross-encoder logit) or RRF passthrough | Cross-encoder for COMMITTED or CLOSED intents only (rerank-on-stability). Disabled → `method=rrf_dedup`. |
| 11 | **Top-k selection** | Ranked candidates | Per-intent ranked list → Evidence Fusion (§12) | k is decided by fusion quotas, not here |

### 10.3 Coverage signal (needed for uncertainty)

RRF scores are rank-based and *not* comparable across queries. An absolute relevance signal is needed to decide `covered`:

| Configuration | `best_score` source | Threshold |
|---|---|---|
| With cross-encoder | Top `rerank_score` | `θ_cov_ce`, calibrated on tune: the value that maximizes F1 of covered vs gold-answerable |
| Without cross-encoder | Top dense cosine | `θ_cov_dense`, calibrated the same way (model-specific; e5-style compressed ranges [M] show why) |

### 10.4 Retrieval scope for refinement

`scope=session_docs` sets `filters.doc_ids` = docs already in the session evidence store. It is used first for delta queries. If that search is not covered, the system falls back to `scope=corpus` (§14).

---

## 11. Evidence Model

### 11.1 Three distinct objects

| Object | Lifetime | Owner | Purpose |
|---|---|---|---|
| **Chunk** | Process (read-only) | CorpusIndex | Immutable corpus unit |
| **Evidence** | Session | Session evidence store | A chunk *as retrieved in this session*, with provenance |
| **Citation** | Answer version | Citation Manager | A rendered reference `Doc_ID §Section`, linked to claims |

### 11.2 `Chunk` (canonical, index-level)

```json
{"chunk_id":"Doc_A§4#1","doc_id":"Doc_A","section_id":"4","part":1,"citation_key":"Doc_A §4",
 "title":"<doc title>","heading":"<section heading>","text":"<chunk text>","token_count":212,
 "char_span":[1830,2911],"page_span":[3,3],"near_dup_group":"g17","metadata":{"source_path":"corpus/xyz.pdf"}}
```

### 11.3 `Evidence` (canonical, session-level)

```json
{
  "evidence_id": "Doc_A§4#1",
  "chunk_id": "Doc_A§4#1",
  "citation_key": "Doc_A §4",
  "doc_id": "Doc_A", "section_id": "4",
  "text": "<chunk text>",
  "hits": [
    {"retrieval_id": "u1.r2", "intent_id": "I2", "trigger": "multi_intent", "t_stream_s": 1.6,
     "lex_rank": 1, "lex_score": 11.4, "dense_rank": 3, "dense_score": 0.71, "rrf_score": 0.0323,
     "rerank_score": 7.2, "final_rank": 1}
  ],
  "intent_ids": ["I2"],
  "best_score_by_intent": {"I2": 7.2},
  "status": "cited",
  "label": "E1",
  "first_seen": {"utterance_id": "u1", "version": 1},
  "cited_in_versions": [1, 2],
  "alternates": ["Doc_D§1#2"],
  "token_count": 212
}
```

| Field | Meaning |
|---|---|
| `evidence_id` | Equals `chunk_id`. Identity across the whole session, so the same chunk retrieved twice is one evidence object. |
| `hits[]` | Full provenance: which retrieval, intent and trigger found it, at what ranks and scores. Supports the evidence-flow telemetry. |
| `best_score_by_intent` | Used for coverage and fusion |
| `status` | `candidate` (retrieved) → `selected` (in a fused set; has a label) → `cited` (a validated claim references it). Or `discarded` (fused out). Status is per session; selection labels are per turn. |
| `label` | `E<n>`, assigned per evidence set (per turn). The LLM only ever sees labels. |
| `alternates` | Near-duplicates collapsed into this one |

### 11.4 `EvidenceSet` (per turn)

```json
{"evidence_set_id":"u1.E","utterance_id":"u1","items":[{"label":"E1","evidence_id":"Doc_A§4#1","intent_ids":["I2"]}],
 "per_intent":{"I2":{"covered":true,"best_score":7.2,"evidence_ids":["Doc_A§4#1"]}},
 "conflicts":[],"carried_from_version":null,"token_count":1420}
```

### 11.5 Evidence flow

```
RetrievalJob ─(candidates)─► Evidence store (upsert by chunk_id; append hit) ─► per-intent candidate pool
   ─► rerank (scores into hits) ─► Fusion selects + labels ─► EvidenceSet ─► Synthesizer (labels + text)
   ─► Grounding maps labels→evidence_ids ─► Citation Manager renders citation keys ─► AnswerVersion
   (refinement: previous version's cited evidence re-enters the next EvidenceSet as "carried" items)
```

---

## 12. Evidence Fusion

### 12.1 Method choice

| Level | Method | Why |
|---|---|---|
| **Within one intent** (lexical ⊕ dense; and several retrievals for the same intent) | **RRF (k=60)** | Rank-based, so no score calibration between BM25 and cosine. Robust on an unseen (held-out) corpus. Cheap. Named by the guide [G§7 p5]. Weighted fusion needs α tuned per corpus, which is fragile under held-out evaluation. Kept as an ablation. |
| **Across intents** | **Quota round-robin** over each intent's reranked list | Intents are *different questions*. RRF across them would let an intent with many strong matches crowd the budget. Quotas guarantee coverage per intent, which is what G3 and per-intent uncertainty need. |

No learned fusion, clustering model or LLM-based selection is used; that would not be justified under C5.

### 12.2 Algorithm

```
procedure FUSE(committed intents I, per-intent ranked lists L_i, carried evidence K (refinement), budget B_tok, q):
    selected ← ordered set; owner ← map evidence → intents
    # 0. refinement: carry evidence cited by kept claims first (they must stay citable)
    for e in K: add(e, intents=e.intent_ids)
    # 1. coverage
    for i in I: covered[i] ← best_score(L_i) ≥ θ_cov
    # 2. quota round-robin (fairness), per-intent cap q (default 3), section cap 2 per intent
    for round in 1..q:
        for i in I ordered by priority:
            e ← next candidate in L_i not violating (section cap) and not near-dup of selected
            if e is None: continue
            if e ∈ selected: owner[e] ∪= {i}; continue       # cross-intent dedup: one item, tagged for both
            if tokens(selected)+tokens(e) > B_tok: stop all
            add(e, intents={i})
    # 3. conflict check (cheap, conservative)
    conflicts ← DETECT_CONFLICTS(selected grouped by intent)
    # 4. order & label: by intent priority, then rank; labels E1..En
    return EvidenceSet(selected, owner, covered, conflicts)
```

### 12.3 Handling of each concern

| Concern | Handling |
|---|---|
| **Duplicates** | Same `chunk_id` across intents → one item tagged with every intent. Near-duplicates (precomputed groups) → best member kept, others recorded as `alternates`. |
| **Overlapping chunks** | Adjacent parts of one section with overlap: the section cap (2 per intent) plus overlap-aware token counting (overlapping characters counted once). Both parts stay citable under the same citation key. |
| **Conflicting evidence** | `DETECT_CONFLICTS`: within one intent, extract typed values (number+unit, currency, duration, percentage, date) from selected items. Two items from **different documents** with the same unit type and a shared nearby keyword but different values → `conflict{intent, items, values}`. The synthesizer is told to present both with citations. Uncertainty `kind=conflict`. No precedence is hardcoded; precedence is used only if the corpus states it. |
| **Source diversity** | Section cap per intent, plus round-robin, prevents one section or document from monopolizing the set |
| **Ranking** | Within an intent: rerank score (or RRF). Across intents: priority (order of mention), then rank. |
| **Evidence density** | Token budget `B_tok` (default 2,000) and quota `q` (default 3). Optional sentence trimming is **off** by default (it keeps chunk-level citation exact). Exp 10 may enable it. |

---

## 13. Session State

### 13.1 Schema

```json
{
  "session_id": "sess-0001",
  "status": "ACTIVE",
  "created_t_wall_ms": 0.0,
  "last_activity_t_wall_ms": 0.0,
  "config_hash": "sha256:…",
  "seq": 0,
  "turns": [
    {"utterance_id": "u1", "turn_type": "query", "transcript": "<accumulated text>", "segments": ["u1.s1"],
     "utterance_end_s": 2.1, "answer_version": 1, "intent_ids": ["I1","I2","I3"]}
  ],
  "frames": [
    {"topic_id": "T1", "topic_text": "<topic statement>", "topic_vec_ref": "emb:…", "slots": {"location": "Pune", "headcount": "30"},
     "active_intent_ids": ["I1","I2","I3"], "opened_in": "u1", "last_version": 1, "status": "active"}
  ],
  "active_topic_id": "T1",
  "intents": {"I1": {"…": "IntentSet item (§9.2)"}},
  "ledger": [
    {"retrieval_id": "u1.r1", "lexical_query": "…", "dense_query": "…", "query_vec_ref": "emb:…",
     "filters": {}, "scope": "corpus", "intent_ids": ["I1"], "trigger": "provisional",
     "t_stream_s": 0.8, "utterance_id": "u1", "result_ids": ["Doc_B§2#1"], "status": "DONE"}
  ],
  "evidence_store": {"Doc_B§2#1": {"…": "Evidence (§11.3)"}},
  "claims": {"C1": {"…": "Claim (§17.2)"}},
  "answers": [{"…": "AnswerVersion (§17.1)"}],
  "pending_constraints": [],
  "counters": {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "retrievals": 0, "ledger_hits": 0},
  "degraded": false
}
```

**Fields deliberately absent:** no user identity, no profile, no cross-session references. There is no `telemetry` field: telemetry is emitted and **never stored or read back** (REQ-SESS-003). The brief's `conversation_context` maps to `turns[].transcript` + `frames`. Its `retrieval_history` maps to `ledger`. Its `citations` are derived from `answers[].citations` and `claims`.

### 13.2 Invariants (checked after every commit; a violation leads to §23.9)

| ID | Invariant |
|---|---|
| I-1 | Every `claims[*].citations` ⊆ keys of `evidence_store` |
| I-2 | Every `answers[k].parent_version` ∈ {null} ∪ {earlier versions in the same `topic_id`} |
| I-3 | Every `intents[*].ledger_refs` ∈ ledger `retrieval_id`s |
| I-4 | `seq` is strictly increasing. Exactly one `ACTIVE` frame. |
| I-5 | No object references another `session_id` |

### 13.3 Isolation mechanisms (cross-session contamination prevention)

1. **Actor per session.** Each session has a dedicated asyncio task and inbound queue. Only that task mutates the `SessionState`, so there are no shared mutable structures and no locks.
2. **Read-only index.** `CorpusIndex` arrays are frozen (`writeable=False`). The index has no per-session fields.
3. **Pure caches only.** The query-embedding LRU and the tokenizer caches are functions of their input text. Their presence cannot change outputs. They can be disabled (`cache.pure=false`) for an isolation audit.
4. **No module-level mutable state** in pipeline code. Enforced by a test that instantiates two sessions with disjoint corpora-of-interest and diffs their state graphs.
5. **IDs are namespaced.** `retrieval_id` and `utterance_id` live inside session-scoped objects. Every event carries a `session_id` that the harness can audit.
6. **Teardown.** On `SESSION_END` or TTL (default 30 min idle), the state is dropped. A GC test checks that no references remain.
7. **LLM context isolation.** Prompts are assembled only from the current session's state. Provider-side prompt caching applies only to the static system-prompt prefix, never to session content `[INF]`.

### 13.4 Topic frames (refinement vs new topic)

- A turn classified **refinement** updates the active frame's slots.
- A **new topic** turn closes the active frame (`status=dormant`) and opens a new one.
- A later turn that matches a dormant frame (topic similarity ≥ `τ_topic` or an explicit back-reference: "back to the venue") re-activates it. The next version's parent is that frame's `last_version`.

---

## 14. Late-Detail Refinement

### 14.1 Definitions

| Term | Definition |
|---|---|
| **Δ (delta)** | The set of constraint operations derived from the new turn: `set` (a slot was unspecified), `update` (a slot changed, e.g. 30 → 45), `retract` (a slot was negated), plus any **new intents** |
| **Affected claim** | A claim of the active frame whose validity may change under Δ |
| **Unaffected claim** | Every other claim of the active frame. It is carried **verbatim**, with its citations, into the next version by deterministic code, not by the LLM. |

### 14.2 Algorithm

```
procedure REFINE(turn u, session S):
    F ← S.active_frame;  v_prev ← S.answers[F.last_version]
    # 1. identify what changed
    Δ ← EXTRACT_DELTA(u.segments, F.slots)          # rules; LLM check if gated (G-e) — §9.6
        for each constraint c in Δ:
            op ← set     if F.slots[c.slot] undefined
                 update  if F.slots[c.slot] ≠ c.value
                 retract if c is negated ("not …", "no longer …")
    NI ← new INFO_REQUEST intents in u               # mixed turn → handled by normal multi-intent path
    # 2. determine affected claims
    for claim k in v_prev.claims:
        k.affected ← (scope(Δ) ∩ {k.intent_id} ≠ ∅)
                     or mentions(k.text ∪ cited_text(k), old values of updated/retracted slots)
                     or any(k.depends_on_claims affected)
    A ← {k | k.affected};  U ← v_prev.claims \ A
    # 3. targeted sub-queries (only the delta)
    DQ ← []
    for c in Δ: for intent i in scope(c) (default: all intents of F):
        DQ += QUERY(i.text ⊕ c.phrase, scope=session_docs)          # e.g. "<topic> international trip"
    DQ += [QUERY(n) for n in NI]
    DQ ← LEDGER_DEDUP(DQ)                                           # never re-issue v_prev queries
    # 4. retrieve delta evidence (streaming: issued during u, as constraints close — §8)
    R ← PARALLEL_RETRIEVE(DQ)
    for q in DQ where not covered(R[q]): R[q] ← RETRIEVE(q, scope=corpus)   # widen only if needed (logged)
    # 5. merge with existing evidence
    EVIDENCE_STORE.upsert(R);  ES ← FUSE(intents(F) ∪ NI, R, carried = cited_evidence(U ∪ A))
    # 6. update only affected claims (+ add new)
    out ← SYNTHESIZE_REFINE(kept = U (read-only, shown for coherence), affected = A, Δ, ES)
          # LLM returns ops: for k∈A: keep|modify(text,labels)|retract ; plus add(text,labels)
          # U is NOT part of the LLM's writable output
    out ← GROUNDING_VALIDATE(out)                   # §16; failed modify → keep old k but flag uncertainty
    # 7. preserve valid previous facts
    claims_v ← U (verbatim, same citations) ∪ apply(out.ops, A) ∪ out.added
    # 8. commit
    v_new ← AnswerVersion(parent=v_prev.version, kind=refinement, claims=claims_v,
                          diff=DIFF(v_prev, claims_v), delta_queries=ids(DQ), full_rerun=false)
    F.slots ← APPLY(F.slots, Δ);  F.last_version ← v_new.version
    emit ANSWER_COMMITTED(v_new)
```

### 14.3 Rendering the refined answer (streaming-friendly)

1. Unaffected claims are already validated. They are emitted **first and immediately** as `ANSWER_DELTA(validated=true)`, in their original order. For refinement turns this makes TTFT almost independent of the LLM `[INF]`, and it matches the guide's v2 shape: the established rule is restated first, then the new specifics [G§4 p4].
2. LLM output for modified and added claims follows, through the sentence gate (§16.6).
3. An optional connective phrase ("However, …") MAY come from the LLM. It is a non-factual connective and is exempt from citation (§16.2).

### 14.4 Edge cases

| Case | Behavior |
|---|---|
| Δ is empty and there are no new intents (the user restates something) | Presentation-like. No retrieval (`NO_RETRIEVE: no_new_information`). Answer: confirm the existing version (no new version), or a presentation version if a format was requested. |
| An `update` contradicts a claim's premise (headcount 30 → 45 makes a venue unsuitable) | The claim is affected. The LLM may `modify` or `retract` it. Retraction is shown in the diff and the uncertainty changes. |
| `retract` ("actually it wasn't international") | Claims added *because of* that constraint become affected (tracked by `introduced_by_constraint`). Typically retracted. The prior slot value is restored. |
| A late detail arrives during synthesis of v(n) | Queued in `pending_constraints` (REQ-SESS-009). After commit, a synthetic refinement turn produces v(n+1). |
| A late detail with no prior answer (first turn) | Treated as a query turn. No refinement. |
| A late detail refers to a dormant topic | Reactivate that frame (§13.4). Its parent = that frame's last version. |
| Multiple late details (Category 7) | The same algorithm runs per turn. Lineage chain v1 → v2 → v3. Constraints accumulate in the frame. |
| Delta evidence not covered even corpus-wide | Uncertainty for that constraint ("whether <constraint> changes the rule could not be verified"). Unaffected claims are kept. |

### 14.5 Why this satisfies G5 [G§5 p5]

| G5 element | Mechanism |
|---|---|
| Narrow or update existing responses | Ops apply only to affected claims |
| Without clearing session state | Frame, ledger, evidence and claims persist |
| Without re-executing full-corpus search | Only Δ queries run; the ledger blocks re-issue; scope is session documents first |

Each element is evidenced in telemetry by `diff`, `delta_queries`, `full_rerun=false` and `RETRIEVAL_SKIPPED`.

---

## 15. Retrieval Suppression

### 15.1 Suppressed act classes

| Act | Definition | Preconditions | Action |
|---|---|---|---|
| `PRESENTATION` | Transform the previous answer's form: format (bullets, table, list), length (shorter, one line), language (translate), repeat or restate, simplify, or summarize **the prior answer** | `answers ≠ ∅` (otherwise `no_prior_answer`) | Transform turn over prior claims. Same citations. No new numbers. |
| `SOCIAL` | Greeting, thanks, acknowledgment, closing | — | Brief acknowledgment (≤1 LLM call, or none in extractive mode) |
| `BACKCHANNEL` | Filler or continuers only ("mm-hm", "okay", "right") | — | No answer. `TURN_COMPLETED` with an empty answer. |
| `META` | A question about the conversation itself ("what did I ask first?", "which sources did you use?") | — | Answer from session state (turns, citations). No retrieval. |

**Not suppressed (K4):** information requests about topics outside the corpus. They are retrieved; the coverage check then produces uncertainty.

### 15.2 Detection

Rules first; the classifier is used for the ablation.

1. **Transformation verb or format noun.** Generic lexicon in config: repeat, rephrase, restate, reword, shorten, condense, simplify, summarize, translate, bullet(s), list, table, one line, "in N words".
2. **Object reference.** The object must refer to *prior system output*: anaphora or deixis ("that", "this", "it", "your answer", "the last answer", "the above", "what you said"), or no object at all.
3. **No new topical content.** The content tokens other than the verb, format nouns and pronouns are ⊆ the vocabulary of the previous answer (and contain no new anchor). Otherwise the turn contains a new request → mixed.
4. **Decision.** PRESENTATION if (1) ∧ (2) ∧ (3). If the verb object is a corpus topic ("summarize the travel reimbursement rule") → INFO_REQUEST.

### 15.3 Examples (Input → Controller → Decision → Reason)

| # | Input (session has a prior answer unless noted) | Controller observations | Decision | Reason |
|---|---|---|---|---|
| 1 | "repeat your last answer in two bullets" | verb repeat + object "your last answer" + format "two bullets" | NO_RETRIEVE | `presentation_restructure` |
| 2 | "make it shorter" | verb + anaphora, no new content | NO_RETRIEVE | `presentation_restructure` |
| 3 | "can you translate that into Hindi" | verb translate + anaphora | NO_RETRIEVE | `presentation_restructure` |
| 4 | "summarize that for me" | verb + anaphora | NO_RETRIEVE | `presentation_restructure` |
| 5 | "summarize the travel reimbursement rule" | verb, but the object is a corpus topic (anchors) | RETRIEVE | `clause_closed, anchor_present` |
| 6 | "thanks, that's helpful" | social | NO_RETRIEVE | `social_ack` |
| 7 | "mm-hm… okay" | backchannel only | NO_RETRIEVE (no answer) | `backchannel` |
| 8 | "which documents did you use for that?" | meta about citations | NO_RETRIEVE | `meta_conversation` |
| 9 | "repeat your last answer" (**empty session**) | presentation, no prior answer | NO_RETRIEVE | `presentation_restructure` + `no_prior_answer`; reply that there is nothing to restate (generic system notice) |
| 10 | "shorter please, and also what's the deposit?" | mixed: presentation segment + INFO_REQUEST segment | segment 1: NO_RETRIEVE; segment 2: RETRIEVE | Mixed turn → query plan with render directive "shorter" |
| 11 | "can you explain that in more detail?" | anaphora, but "more detail" asks for **more information** | RETRIEVE (scope = cited sections of the prior answer; `trigger=refinement`) | Expansion needs evidence beyond the prior claims |
| 12 | "how's the weather in Pune today?" | INFO_REQUEST, but out of domain | RETRIEVE at the end of the utterance (not early unless anchored) | K4: coverage → uncertainty |

Example 11 is the boundary case. Transformations of *existing* content are suppressed; requests for *additional* content are not. That is the operational line between pitfall 4 [G§6 p5] and an under-answered turn.

---

## 16. Grounding & Citation

### 16.1 Pipeline

```
EvidenceSet (labels E1..En) ─► Synthesis prompt (evidence as quoted data; rules; per-intent coverage flags)
  ─► LLM output: sentences with inline labels [E#] or [U] (uncertainty) ─► sentence gate (streaming)
  ─► Claim extraction (1 sentence = 1 claim unit) ─► label resolution (label → evidence_id; unknown → dropped)
  ─► Support verification (L0–L3) ─► policy actions ─► Citation rendering (Doc_ID §Section)
  ─► Uncertainty assembly (coverage-driven + verifier-driven) ─► AnswerVersion
```

### 16.2 Claim taxonomy

| Type | Example (abstract) | Citation required | Detection |
|---|---|---|---|
| FACTUAL | Rule, amount, condition, procedure step | **Yes** | Default for any sentence with content tokens |
| UNCERTAINTY | "X could not be verified from the corpus." | No (tag `[U]`) | `[U]` tag; must reference an intent or aspect |
| CONNECTIVE | "However," / "In addition:" | No | ≤ 4 tokens, no content anchors, no numbers |
| CLARIFICATION | A question back to the user | No | Turn plan = clarification |

Any FACTUAL sentence without a label is UNSUPPORTED by definition.

### 16.3 Synthesis output contract (to the LLM)

- Evidence is shown as `[E1] <citation key> — "<text>"` inside a delimited data block. The system prompt states that the block is data, not instructions (§24).
- **Rules given to the model:**
  - Write one claim per sentence.
  - End every factual sentence with ≥1 `[E#]`.
  - Use only facts stated in the cited evidence.
  - For each intent flagged `NOT COVERED`, write one `[U]` sentence instead of an answer.
  - Never invent labels.
- Initial turns produce prose with inline labels (streamable).
- Refinement turns produce the ops structure (§14.2): `[{op, claim_id?, text, labels}]`.

### 16.4 Verification levels

| Level | Check | Cost | Verdict effect |
|---|---|---|---|
| L0 | Every label ∈ current EvidenceSet labels | µs | Unknown label → dropped; if none remain → `INVALID_CITATION` |
| L1 | Numbers, dates, amounts and percentages in the claim (normalized) ⊆ tokens of the cited evidence | µs–ms | Mismatch → `UNSUPPORTED` |
| L2 | Support score = max over cited items of max(content-word recall of claim vs evidence; cosine(claim, best evidence sentence)) | ms (≤1 embedding per claim) | ≥ `ρ_sup` → SUPPORTED; ≥ `ρ_partial` → PARTIAL; else UNSUPPORTED (thresholds calibrated against human labels) |
| L3 (optional, Exp 9) | NLI cross-encoder entailment | ~tens of ms per pair [E] | Overrides L2 when enabled |
| Repair | If UNSUPPORTED with its own label but SUPPORTED by **another** selected label → swap the label (logged `citation_repaired`) | ms | Fixes miscitation (failure #14) |

### 16.5 Decision table: insufficient, conflicting, unverifiable

| Situation | Detection | Behavior | Output fields |
|---|---|---|---|
| **Evidence insufficient for an intent** | `covered=false` (§10.3) before synthesis | The intent is marked NOT COVERED in the prompt. The answer contains a `[U]` sentence. The uncertainty record is created **deterministically** (not dependent on the LLM). | `uncertainty[{intent_id, kind:no_evidence, aspect}]`, `intent_status=uncertain` |
| **Insufficient for all intents** | All uncovered | No factual claims. Uncertainty plus an optional targeted clarification ("did you mean …" built from the intents, not invented facts). | kind=clarification or uncertainty-only |
| **Evidence conflicts** | `EvidenceSet.conflicts` | The prompt instructs the model to present both values with their own citations. Uncertainty records the conflict. | `uncertainty[{kind:conflict, items}]` |
| **A claim cannot be verified** | L1/L2 → UNSUPPORTED after repair | Policy `on_unsupported` (default `drop_and_flag`): remove the claim and add an uncertainty item for its intent ("details on X could not be verified"). Alternatives: `flag_inline`, `repair_once` (+1 LLM call; ablation only). | `GROUNDING_CHECKED.policy_actions` |
| **Ambiguous request** (unresolved referent, decisive slot missing) | Intent Analyzer: anaphora with no antecedent in the frame; a slot required by ≥2 conflicting evidence branches | Clarification turn: one targeted question. No factual claims. | kind=clarification |
| **Validator itself fails** | Exception | Commit with `unverified=true`. All claims flagged. Never presented as verified. | `ERROR`, uncertainty |

### 16.6 Sentence-gated streaming

- Tokens from the LLM are buffered until a sentence boundary and its trailing labels are complete.
- L0–L2 run on that sentence (≈ms). Then `ANSWER_DELTA(validated=true)` is emitted.
- Two latencies are recorded:
  - `ttft_raw_ms`: the first LLM token;
  - `ttft_ms`: the first validated sentence. **This is the reported TTFT.**
- `stream_mode=raw` (ablation only) emits tokens immediately with `validated=false`. It is never the default.

### 16.7 Citation rendering

- **Inline:** each sentence is followed by its citation keys in brackets, e.g. `[Doc_A §4]`.
- **Record:** `citations[]` = unique citation keys in order of first appearance, matching the guide's list format [G§4 p4].
- **Traceability chain:** citation key → evidence_id(s) → chunk_id → (doc_id, section_id, char_span, source_path). It can be resolved by the eval harness from the manifest alone.

---

## 17. Answer Versioning

### 17.1 `AnswerVersion`

```json
{
  "version": 2,
  "parent_version": 1,
  "topic_id": "T1",
  "utterance_id": "u2",
  "kind": "refinement",
  "created_t_stream_s": 1.5,
  "text": "<rendered answer>",
  "claim_ids": ["C1", "C2", "C4", "C5"],
  "citations": ["Doc_A §1", "Doc_C §3"],
  "evidence_set_id": "u2.E",
  "uncertainty": [],
  "frame_slots": {"trip_type": "international", "booking_timing": "after_travel"},
  "diff": {
    "kept": ["C1", "C2"],
    "modified": [{"claim_id": "C3", "from_text": "…", "to_text": "…", "from_citations": ["Doc_A §1"], "to_citations": ["Doc_A §1", "Doc_C §3"]}],
    "added": ["C4", "C5"],
    "retracted": [],
    "citations_added": ["Doc_C §3"],
    "citations_removed": [],
    "evidence_added": ["Doc_C§3#1"],
    "uncertainty_resolved": [],
    "uncertainty_introduced": []
  },
  "delta_queries": ["u2.r1", "u2.r2"],
  "full_rerun": false,
  "unverified": false
}
```

### 17.2 `Claim`

```json
{"claim_id":"C3","text":"…","type":"FACTUAL","intent_id":"I1","citations":["Doc_A§1#1"],
 "verdict":"SUPPORTED","introduced_in":1,"modified_in":[2],"introduced_by_constraint":null,"status":"active"}
```

Claim IDs are **stable across versions**. A modified claim keeps its ID; the diff records the old and new text. A retracted claim gets `status=retracted` and stays in the store for lineage, never rendered.

### 17.3 Version semantics by kind

| Kind | Claims | Citations | Evidence | Parent |
|---|---|---|---|---|
| `initial` | New | New | New set | null (new topic thread) |
| `refinement` | Kept (verbatim) + modified + added − retracted | Kept ∪ delta | Carried + delta | Previous version of the same topic |
| `presentation` | **Same claim IDs and facts.** Only the rendering changes. | **Same set** (⊆ check) | Unchanged | Previous version |
| `meta`, `social` | None (no factual claims) | None | — | Previous version (does not change topic content). MAY be omitted from the content lineage: `content_parent=parent.content_parent`. |
| `clarification` | None | None | — | Previous version or null |

**Example lineage (abstract):**

```
v1 (initial,  T1)  claims C1,C2,C3 — cites Doc_A §1, Doc_B §2
 └─ v2 (refinement, Δ={trip_type:set}) kept C1,C2; modified C3; added C4,C5 — cites + Doc_C §3
     └─ v3 (presentation: "two bullets") same claims C1..C5 re-rendered — citations ⊆ v2
         └─ v4 (refinement, Δ={headcount:update 30→45}) kept C1,C2,C4; modified C5 — delta cites …
```

---

## 18. Component Contracts

Interfaces are language-neutral signatures. *Latency* is per call: `[M]` measured, `[E]` estimate. All components emit their events through the Telemetry Manager.

### 18.1 Transcript Streamer

| Aspect | Contract |
|---|---|
| INPUT | Replay JSONL (header + input events), or live text/ASR source; clock mode (real, virtual) and speed |
| OUTPUT | Ordered input events to the Orchestrator, timed per `timestamp_s` (real) or immediately (virtual) |
| RESPONSIBILITY | Parse and validate input events; convert cumulative to delta; pace in real time; inject `UTTERANCE_END(reason=eof)` at end of file if missing |
| FAILURE MODE | Malformed line → `ERROR(recoverable)`, skip line. Schema-major mismatch → abort the run (fail fast). Clock drift in real mode → logged as `pacing_lag_ms`. |
| LATENCY | < 1 ms per event [E] (excluding intentional pacing) |
| DEPENDENCIES | Input schema (§5) |

### 18.2 Chunk Manager

| Aspect | Contract |
|---|---|
| INPUT | `TRANSCRIPT_CHUNK`, `UTTERANCE_END` |
| OUTPUT | `CHUNK_RECEIVED`; updated `SegmentView[]` (text, state, boundary, dangling, token counts, timers) |
| RESPONSIBILITY | Idempotency (`utterance_id`, `chunk_index`); reorder window of 1; revisions (`replaces_chunk_index`); accumulation; text normalization; **segmentation** (boundary and dangling detection, stability timers, force-close) |
| FAILURE MODE | Duplicate → ignore. Conflicting duplicate → treat as a revision. Gap (missing index) → `ERROR(recoverable)`, continue. Never blocks. |
| LATENCY | < 2 ms per chunk [E] |
| DEPENDENCIES | Generic lexicons (config) |

### 18.3 Retrieval Controller

| Aspect | Contract |
|---|---|
| INPUT | `SegmentView`, `ActInfo` (from the Intent Analyzer), `SessionView` (read-only), tick |
| OUTPUT | `CONTROLLER_DECISION` per segment; dispatch requests to the Decomposer/dispatcher |
| RESPONSIBILITY | WAIT / RETRIEVE / NO_RETRIEVE per §8; turn-type hypothesis; budgets |
| FAILURE MODE | Internal exception → default to `WAIT` during listening and `RETRIEVE` at the end of the utterance for any non-suppressed segment (fail toward grounding), plus `ERROR` |
| LATENCY | p95 < 20 ms (REQ-PERF-001); typical < 5 ms [E] |
| DEPENDENCIES | Intent Analyzer, Query Ledger (novelty), Embedder (optional novelty cosine) |

### 18.4 Intent Analyzer

| Aspect | Contract |
|---|---|
| INPUT | `SegmentView`, `SessionView` (frame, prior answer vocabulary) |
| OUTPUT | `ActInfo{act, confidence, anchors[(term, idf)], slots{}, anaphora_refs[], topic_continuity}`; turn-type features |
| RESPONSIBILITY | Dialog-act classification (§8.3, §15.2); slot extraction; anchor lookup in the corpus vocabulary; topic continuity vs frame |
| FAILURE MODE | Unknown act → `act=INFO_REQUEST, confidence=0.5` (fail toward retrieval at the end of the utterance) |
| LATENCY | < 3 ms [E] (rules; plus 1 embedding for topic continuity, ~4–9 ms [M/E]) |
| DEPENDENCIES | Vocabulary/IDF table (index), Embedder (optional), lexicons |

### 18.5 Query Decomposer

| Aspect | Contract |
|---|---|
| INPUT | Closed or stable segments + `ActInfo`; `SessionView`; (at finalize) LLM check result |
| OUTPUT | `IntentSet` (§9.2), `INTENTS_UPDATED`; per-intent `QueryForms` |
| RESPONSIBILITY | Coordination split, constraint attachment, dependency, context propagation, normalization, dedup and split guards, gating and reconciliation of the LLM check |
| FAILURE MODE | Rule failure → one intent = the whole segment (no split). LLM check error or timeout → keep the rule intents. |
| LATENCY | Rules < 2 ms [E]; LLM check 0.3–1.5 s hosted [E] (off the critical path, bounded wait 300 ms) |
| DEPENDENCIES | Intent Analyzer, Query Ledger, LLM Gateway (gated), Lexical/Dense Retrievers (over-split overlap test uses cached top-k) |

### 18.6 Dense Retriever

| Aspect | Contract |
|---|---|
| INPUT | `dense_text` (or a precomputed vector), filter mask, N |
| OUTPUT | `[(chunk_id, cos, rank)]` |
| RESPONSIBILITY | Query embedding + exact inner-product search over the frozen matrix |
| FAILURE MODE | Model load or inference error → raise `EmbedderUnavailable`. The dispatcher degrades to lexical-only for the process (`ERROR(action=lexical_only)`). |
| LATENCY | Embed 3–4 ms (MiniLM) [M] / ~6–9 ms (bge-small) [E]; search 1.6 ms @ 100k [M] |
| DEPENDENCIES | CorpusIndex (embeddings), embedder runtime (ONNX or torch) |

### 18.7 Lexical Retriever

| Aspect | Contract |
|---|---|
| INPUT | `lexical_terms`, filter mask, N |
| OUTPUT | `[(chunk_id, bm25, rank)]` |
| RESPONSIBILITY | BM25 scoring over the sparse matrix, same tokenizer as index time |
| FAILURE MODE | Empty term list → empty result (not an error). Exception → dense-only for that query (`ERROR(action=dense_only)`). |
| LATENCY | 0.04–0.4 ms (1k–100k chunks) [M] |
| DEPENDENCIES | CorpusIndex (BM25 matrix, vocabulary) |

### 18.8 Evidence Fusion

| Aspect | Contract |
|---|---|
| INPUT | Committed intents, per-intent ranked candidates, carried evidence, budgets |
| OUTPUT | `EvidenceSet`, `EVIDENCE_FUSED` |
| RESPONSIBILITY | RRF within intent; quota round-robin across intents; dedup and near-dup; section cap; conflict detection; labeling; coverage flags |
| FAILURE MODE | Exception → fall back to the top-q per intent by RRF with no conflict detection (`ERROR`) |
| LATENCY | < 10 ms [E] |
| DEPENDENCIES | Evidence store, near-dup groups |

### 18.9 Reranker

| Aspect | Contract |
|---|---|
| INPUT | `dense_text` (query) + ≤ `M_rerank` candidate chunk texts |
| OUTPUT | `rerank_score` per candidate; `EVIDENCE_RERANKED`; `covered`, `best_score` |
| RESPONSIBILITY | Cross-encoder scoring (if enabled), or RRF passthrough + dedup (`rrf_dedup`) |
| FAILURE MODE | Timeout (600 ms) or error → `rrf_dedup` for that intent; coverage falls back to the dense threshold |
| LATENCY | 105–124 ms @ 20 pairs; 150–185 ms @ 30 pairs [M] |
| DEPENDENCIES | Cross-encoder runtime (optional) |

### 18.10 Session State Manager

| Aspect | Contract |
|---|---|
| INPUT | All state-changing events of one session (actor mailbox) |
| OUTPUT | `SessionView` snapshots (immutable) for other components; `ANSWER_COMMITTED`, `SESSION_CLOSED` |
| RESPONSIBILITY | Own `SessionState` (§13); frames; ledger; evidence store; claims; versions; invariants; pending constraints; TTL; teardown |
| FAILURE MODE | Invariant violation → §23.9 (degraded mode, never silently continue on corrupt state) |
| LATENCY | < 1 ms per update [E] |
| DEPENDENCIES | None, except the read-only index for validation |

### 18.11 Answer Synthesizer

| Aspect | Contract |
|---|---|
| INPUT | Turn plan (mode), `EvidenceSet`, intents with coverage, kept/affected claims (refinement), render directives (presentation), backend |
| OUTPUT | Token stream → sentence gate → claim candidates; `SYNTHESIS_STARTED`, `LLM_CALL` |
| RESPONSIBILITY | Prompt assembly from config templates; backend selection (hosted → local → extractive); streaming; extractive mode |
| FAILURE MODE | LLM error or timeout → next backend; final fallback extractive (cannot fail on valid evidence) |
| LATENCY | TTFT target ≤ 1.5 s after the end of the utterance (hosted) [T]; local CPU to be measured |
| DEPENDENCIES | LLM Gateway, Evidence Fusion, config prompts |

### 18.12 Grounding Validator

| Aspect | Contract |
|---|---|
| INPUT | Sentence or claim candidates with labels; `EvidenceSet`; previous version (for presentation checks) |
| OUTPUT | Verdicts; repaired labels; dropped claims; uncertainty items; `GROUNDING_CHECKED` |
| RESPONSIBILITY | L0–L2 (L3 optional); repair; `on_unsupported` policy; presentation ⊆ checks; deterministic uncertainty from coverage |
| FAILURE MODE | Internal error → mark `unverified` (never "verified") |
| LATENCY | < 5 ms per sentence (L0–L2) [E]; L3 ~tens of ms per pair [E] |
| DEPENDENCIES | Embedder (L2 cosine), optional NLI model |

### 18.13 Citation Manager

| Aspect | Contract |
|---|---|
| INPUT | Validated claims with evidence_ids; CorpusIndex manifest |
| OUTPUT | Inline citation keys, `citations[]`, the traceability map, the citation diff vs the parent |
| RESPONSIBILITY | Label → evidence_id → citation key mapping; **existence check against the manifest** (zero fabricated IDs); ordering; diff |
| FAILURE MODE | Unresolvable key → drop the citation and flag the claim UNSUPPORTED (never emit an unverifiable ID) |
| LATENCY | < 1 ms [E] |
| DEPENDENCIES | CorpusManifest |

### 18.14 Telemetry Manager

| Aspect | Contract |
|---|---|
| INPUT | All events (bus subscriber) |
| OUTPUT | JSONL trace per run; per-turn `TURN_COMPLETED` file; run manifest; optional live SSE feed |
| RESPONSIBILITY | Envelope stamping (seq, clocks); schema validation (debug mode); non-blocking buffered writes; flush on shutdown; text redaction modes (§24) |
| FAILURE MODE | Disk error → stderr fallback + coverage flag. **Never blocks or crashes the pipeline.** |
| LATENCY | < 1 ms per event enqueue [E] |
| DEPENDENCIES | None (sink only). Never read by the pipeline. |

### 18.15 Supporting components (required for completeness)

| Component | INPUT → OUTPUT | Responsibility | Failure mode | Latency | Dependencies |
|---|---|---|---|---|---|
| **Corpus Indexer** | corpus/ → CorpusIndex + manifest | Load, section, chunk, embed, BM25, near-dup, vocabulary | Unparseable file → skip + report. Empty corpus → fail fast. | ≤120 s per 10k chunks [T] | Embedder, parsers |
| **Query Ledger** | Query forms → hit/miss + ledger entry | Normalized and semantic dedup; reuse | Embedding unavailable → lexical-only dedup | < 2 ms [E] | Session state |
| **LLM Gateway** | Prompt + schema → stream + usage | Adapters (Anthropic SDK, Ollama, extractive), timeouts, retries (1), cost from the price table, cancellation | Error → fallback chain | Backend-dependent | Config, network (hosted only) |
| **Turn Orchestrator** | Input events → state transitions | Implements §7; serializes commits; wait bounds | Unexpected exception → abort turn with uncertainty answer + `ERROR` | < 1 ms [E] | All components |
| **Evaluation Harness** | Traces + gold → metrics | §21–§22 metrics; reports | — | Offline | Traces, gold labels |

---

## 19. End-to-End Data Flow

### 19.1 Swimlane view (one query turn, then a refinement turn)

```
LANE            │ DATA IN                         │ PROCESS (sync=│ / async=⇢ / parallel=∥)            │ DATA OUT / STATE UPDATE
────────────────┼─────────────────────────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────
Streamer        │ replay JSONL / live text         │ parse, validate, pace                               │ input events
Chunk Manager   │ TRANSCRIPT_CHUNK                 │ dedup/reorder │ append │ segment (boundary, dangling) │ CHUNK_RECEIVED; SegmentView[]
Intent Analyzer │ SegmentView, SessionView         │ act, slots, anchors (IDF lookup), topic continuity  │ ActInfo
Controller      │ SegmentView, ActInfo, ledger     │ ◆ DECISION per segment (§8.5)                       │ CONTROLLER_DECISION
   ◆ WAIT ──────┼──────────────────────────────────┼► (no-op; stability timer armed)                     │
   ◆ NO_RETRIEVE┼──────────────────────────────────┼► mark segment IGNORED; render directives            │ turn_type hypothesis
   ◆ RETRIEVE ──┼──────────────────────────────────┼► Decomposer                                         │
Decomposer      │ closed/stable segments           │ split, constraints, deps, context, normalize        │ IntentSet; INTENTS_UPDATED
Ledger          │ QueryForms                       │ ◆ hit? → RETRIEVAL_SKIPPED │ miss → dispatch         │ ledger entry (state)
Retrieval       │ QueryForms × n intents           │ ∥ per intent ⇢ thread: embed → (BM25 ∥ dense) → RRF │ RETRIEVAL_STARTED/COMPLETED; evidence upsert
Reranker        │ per-intent candidates (≤20)      │ ⇢ on segment CLOSED (rerank-on-stability)           │ EVIDENCE_RERANKED; coverage flags
────────────────┼────────────── UTTERANCE_END ─────┼─────────────────────────────────────────────────────┼──────────────────────────────────────
Orchestrator    │ UTTERANCE_END                    │ close segments │ final decisions │ ◆ gated LLM check │ final IntentSet (COMMITTED/MERGED/DROPPED)
                │                                  │ ⇢ await outstanding retrieval/rerank ≤ bounds        │
Fusion          │ committed intents, candidates    │ RRF-in-intent │ quota RR │ dedup │ conflicts │ labels │ EvidenceSet; EVIDENCE_FUSED
Orchestrator    │ turn type, EvidenceSet           │ ◆ plan: answer | transform | clarify | none         │ SYNTHESIS_STARTED
Synthesizer     │ prompt(labels, coverage)         │ ⇢ LLM stream (hosted→local→extractive)               │ token stream; LLM_CALL
Grounding       │ sentences                         │ sentence gate: L0→L1→L2(→L3) │ repair │ policy         │ ANSWER_DELTA(validated); GROUNDING_CHECKED
Citation Mgr    │ validated claims                  │ label→evidence→citation key; manifest check; diff   │ citations[], trace map
Session Mgr     │ claims, citations, diff           │ create AnswerVersion; invariants; frame update      │ ANSWER_COMMITTED (state: answers, claims)
Orchestrator    │ —                                 │ assemble guide-format record                        │ TURN_COMPLETED
Telemetry       │ every event                       │ ⇢ stamp, buffer, write JSONL                         │ trace files (never read back)
════════════════╪═════════════ next utterance (late detail) ═════════════════════════════════════════════════════════════════════
Analyzer        │ "the trip was international…"    │ act=CONSTRAINT; continuity=high → turn_type=refinement
Controller      │                                  │ RETRIEVE(trigger=refinement) on constraint closure (early, mid-utterance)
Refiner (§14)   │ Δ, frame, v_prev                 │ affected claims │ delta queries (session_docs scope) │ ledger blocks re-issue
Fusion          │ carried + delta evidence         │ FUSE(carried first)
Synthesizer     │ kept (verbatim, emitted first) + affected/new via LLM ops
Session Mgr     │                                  │ v2(parent=v1, diff, full_rerun=false)               │ ANSWER_COMMITTED(kind=refinement)
```

### 19.2 Stage-by-stage specification

| # | Stage | Enters | Leaves | Decision point | Async / parallel | State update |
|---|---|---|---|---|---|---|
| 1 | Ingest event | Input event | Validated event | Schema valid? | Sync | — |
| 2 | Chunk handling | Chunk | `CHUNK_RECEIVED`, segments | Duplicate, revision or gap? | Sync | `turns[].transcript`, segments |
| 3 | Act analysis | Segments | `ActInfo` | — | Sync (+1 optional embed in a thread) | — |
| 4 | Controller | Segments + ActInfo + SessionView | Decisions | WAIT / RETRIEVE / NO_RETRIEVE | Sync | turn-type hypothesis |
| 5 | Decomposition | Closed or stable segments | IntentSet, QueryForms | Split? merge? constraint scope? | Sync (LLM check async, gated) | `intents` |
| 6 | Ledger | QueryForms | Hit or miss | Novel? | Sync | `ledger` |
| 7 | Retrieval | QueryForms | Candidates | Timeout → partial | **Async; ∥ across intents; ∥ BM25 and dense inside** | `evidence_store` (upsert, hits) |
| 8 | Rerank | Candidates | Scored candidates, coverage | Cross-encoder enabled and within time? | Async (thread), on stability | evidence hits, coverage |
| 9 | Finalize | `UTTERANCE_END` | Final intents | Await bounds exceeded? Use the LLM check? | Bounded awaits | intents status |
| 10 | Fusion | Intents + candidates (+carried) | EvidenceSet | Budget reached? Conflicts? | Sync | evidence status=selected |
| 11 | Plan | Turn type, coverage | Mode | answer / transform / clarify / none | Sync | — |
| 12 | Synthesis | Prompt | Token stream | Backend fallback? | Async stream | counters |
| 13 | Grounding | Sentences | Validated deltas | Supported? repair? drop? | Pipelined with decoding | — |
| 14 | Citation | Claims | Citation keys | Key exists? | Sync | — |
| 15 | Commit | Claims, citations | AnswerVersion | Invariants OK? | Sync (serialized per session) | `answers`, `claims`, frame |
| 16 | Record | All | `TURN_COMPLETED` | — | Sync | — |
| 17 | Telemetry | Events | JSONL | — | Async writer | (external sink) |

---

## 20. Latency Model

### 20.1 Variables

| Symbol | Meaning | Value (RH-dev) | Status |
|---|---|---|---|
| `T_chunk` | Inter-chunk interval (input property) | ~0.8 s in Ex1 | Input |
| `T_gap` | Last chunk → `UTTERANCE_END` (endpoint silence) | 0.5 s in Ex1 | Input |
| `T_controller` | Segment + act + decision per chunk | < 5 ms | [E] |
| `T_decomp_rule` | Rule decomposition | < 2 ms | [E] |
| `T_decomp_llm` | Gated LLM check | 0.3–1.5 s hosted | [E] |
| `T_embed` | Query embedding | 3–4 ms (MiniLM) / ~6–9 ms (bge-small) | [M] / [E] |
| `T_BM25` | Lexical search | 0.04–0.4 ms (1k–100k) | [M] |
| `T_dense` | Exact dense search | 0.01–1.6 ms (1k–100k) | [M] |
| `T_fusion_rrf` | RRF within intent | < 1 ms | [E] |
| `T_rerank(m)` | Cross-encoder, m pairs | 54–63 ms (10), 105–124 ms (20), 149–185 ms (30) | [M] |
| `T_fusion` | Cross-intent fusion | < 10 ms | [E] |
| `T_prompt` | Prompt assembly | < 5 ms | [E] |
| `T_llm_first` | LLM first-token latency (network + prefill) | Backend-dependent | [E]; measure |
| `T_sentence` | Decode time of the first sentence (~20–30 tokens) | Backend-dependent | [E]; measure |
| `T_ground` | L0–L2 per sentence | < 5 ms | [E] |

### 20.2 Metric definitions

| Metric | Formula | Clock |
|---|---|---|
| **Time-to-first-retrieval (TTFR)** | `t(first RETRIEVAL_STARTED) − t(first chunk)` | Stream |
| **Retrieval lead time** | `t(UTTERANCE_END) − t(first RETRIEVAL_STARTED)` | Stream |
| **Time-to-first-token (TTFT)** | `t_wall(first ANSWER_DELTA[validated]) − t_wall(UTTERANCE_END processed)`. Also reported: `ttft_raw` (first LLM token). | Wall |
| **Time-to-final-answer (TTFA)** | `t_wall(ANSWER_COMMITTED) − t_wall(UTTERANCE_END processed)` | Wall |
| **Evidence-ready slack** | `t(UTTERANCE_END) − max over committed intents of t(evidence final)`. Negative = spill onto the critical path. | Stream (real-time replay) |

### 20.3 Critical path

```
TTFT  = spill + T_llm_check_wait* + T_fusion + T_prompt + T_llm_first + T_sentence + T_ground
spill = max(0, max_i [t_rerank_done(i)] − t_end)  bounded by finalize_wait_ms (default 150 ms; then RRF fallback)
* only when the gated check is still in flight (bounded by llm_check_wait_ms = 300 ms)

TTFA  = TTFT + T_decode(remaining tokens) + Σ T_ground(remaining sentences) + T_commit
Refinement turns: TTFT_refine ≈ T_fusion + T_ground(kept)   (kept claims stream first; §14.3)
Presentation turns: TTFT = T_prompt + T_llm_first + T_sentence + T_ground   (no retrieval)
```

**Off the critical path** (runs during speech or concurrently):

- Controller, decomposition, retrieval and rerank (during `LISTENING`).
- The LLM check, if gated early.
- Telemetry writes.
- Grounding of sentence *k* while sentence *k+1* decodes.

**Parallelizable:**

| What | How |
|---|---|
| BM25 ∥ dense | Inside a query |
| Retrieval across intents | ∥ |
| Rerank across intents | Thread pool. CPU-bound, so the gain is limited by cores. Measured scaling with pairs is ~linear, so batching gives little extra. |
| LLM check ∥ listening | Concurrent |
| Grounding ∥ decoding | Pipelined |

**Cannot be parallelized** (sequential by contract):

- Synthesis after `UTTERANCE_END`.
- Commit after validation.
- Commits across turns of one session.

### 20.4 Worked budget for Example 1 on RH-dev

Measured components, estimated composition:

| t (stream) | Work | Done by |
|---|---|---|
| 0.80 | I1 retrieval (≈ 2 + 9 + 1 ms) | ≈ 0.81 |
| 0.81 | I1 rerank on stability (20 pairs, 105–124 ms [M]) | ≈ 0.93 |
| 1.60 | I2, I3 retrieval ∥ (≈ 12 ms) | ≈ 1.61 |
| 1.61 | I2 + I3 rerank, serial (2 × 105–124 ms [M]) | ≈ 1.82–1.86 |
| 2.10 | `UTTERANCE_END`: slack ≈ +0.24 s, so no spill. Fusion + prompt ≈ 15 ms. | ≈ 2.115 |
| 2.115 + | `T_llm_first + T_sentence + T_ground` | TTFT = (backend-dependent) + ~20 ms |

**Judge-hardware sensitivity.** If the judge CPU is *f*× slower (*f* unknown; measure on RH-judge), the I2+I3 rerank takes 0.21–0.25·*f* s. Spill starts at *f* ≈ 2. Mitigations, in order:

1. `finalize_wait_ms` bound, with RRF fallback for the unfinished intents.
2. `M_rerank = 10` (halves the cost [M]).
3. Disable the cross-encoder if Exp 3 shows a small gain.

### 20.5 Budget targets

| Quantity | Target | Basis |
|---|---|---|
| Controller p95 | < 20 ms | REQ-PERF-001 |
| Retrieval p95 per query | < 50 ms | REQ-PERF-002 |
| Rerank p95 per intent (20 pairs, RH-dev) | ≤ 150 ms | REQ-PERF-004 [M] |
| Evidence-ready slack ≥ 0 | ≥ 80% of eligible turns | REQ-STREAM-010 [T] |
| TTFT p50 (hosted) | ≤ 1.5 s | REQ-PERF-003 [T] |
| TTFT (local, extractive) | Reported | — |

---

## 21. Evaluation Mapping

**Corrected G3 definitions (K5):**

- **Lenient:** ≥2 distinct gold intents matched one-to-one by predicted intents.
- **Strict:** all gold intents matched **and** every predicted intent matched (no extras).

| Samsung requirement (source) | Component(s) | Implementation mechanism | Telemetry | Benchmark | Success criterion |
|---|---|---|---|---|---|
| **G1** Reproducibility [G§5 p4] | Packaging, LLM Gateway fallback, Indexer cache | `docker compose up`; keyless fallback; pinned locks and models | Run manifest | Fresh-clone run on amd64 + arm64, empty env | Pass |
| **G2** Early retrieval ≥80% [G§5 p5] | Chunk Manager, Controller, Ledger | Segment-event dispatch; rerank-on-stability | `RETRIEVAL_STARTED.t_stream_s`, `UTTERANCE_END` | Categories 1–4, 6, 10 (eligible) | ≥ 80% of eligible turns |
| **G2** Low false triggers [G§5 p5] | Intent Analyzer, Controller | Suppressed act classes (§15) | `CONTROLLER_DECISION` reasons | Category 5 + suppression subset of 9 | ≤ 5% `[T]` |
| **G3** Multi-intent ≥70% [G§5 p5] | Decomposer (+ gated LLM), Fusion quotas | Coordination split, guards, reconciliation | `INTENTS_UPDATED` | Categories 2, 3, 10 compound cases | Lenient ≥ 70%; strict reported |
| **G4** Citation support ≥85% [G§5 p5] | Synthesizer, Grounding Validator, Citation Manager | Labels, L0–L2(+L3), repair, drop_and_flag | `GROUNDING_CHECKED`, `ANSWER_COMMITTED` | All answered turns; human-labeled sample ≥ 100 claims | ≥ 85% supported; calibration agreement reported |
| **G4** Zero fabricated IDs [G§5 p5] | Citation Manager | Manifest existence check; labels only | `citations[]` | All turns | 100% valid |
| **G5** Session refinement [G§5 p5] | Session Manager, Refiner, Ledger | Δ extraction, affected claims, delta queries, verbatim carry-over | `ANSWER_COMMITTED.diff`, `delta_queries`, `full_rerun`, `RETRIEVAL_SKIPPED` | Categories 6, 7 | Lineage valid 100%; re-issue 0; kept-claim fidelity 100% |
| **G6** Telemetry 100% [G§5 p5] | Telemetry Manager, event schemas | Envelope + 16 types; non-blocking writer | All | All turns | 100% schema-valid with required fields |
| Listens incrementally [G§1 p1] | Streamer, Chunk Manager, Controller | §5, §8 | Chunk and decision events | G2 suite | = G2 |
| Decomposes and parallelizes [G§1 p1] | Decomposer, dispatcher | `asyncio.gather` per intent | Overlapping `RETRIEVAL_STARTED` | Category 2 | Concurrent dispatch observed |
| Refines rather than restarts [G§1 p1] | Refiner | §14 | diff | Categories 6, 7 | = G5 |
| Guarantees corpus grounding [G§1 p1] | Grounding, Citation | §16 | Verdicts | All | = G4 + uncertainty recall ≥ 90% `[T]` |
| Corpus isolation [G§3 p3] | Retrieval, LLM prompt, Grounding | Index-only search; evidence-only prompt; L1/L2 verification | Egress log; verdicts | Egress test; unsupported-claim rate | 0 non-LLM egress; supported ≥ 85% |
| No hardcoding / precomputation [G§3 p3] | Repo hygiene | Eval data only in `eval/`; guard test; index ≠ answers | Manifest (config hash) | Guard test | Pass |
| Rigorous grounding + uncertainty [G§3 p3] | Grounding | Deterministic coverage → uncertainty | `uncertainty[]` | Unanswerable gold intents | Recall ≥ 90% `[T]` |
| Session-bound state [G§3 p3] | Session Manager | Actor isolation; no persistence | `SESSION_CLOSED` | Category 11 isolation test | 0 leakage |
| Architectural parsimony [G§3 p3], [D s7] | Whole system | ≤2 LLM calls per turn; no agent framework; ablation per component | `LLM_CALL` counts | All | Budgets met; each optional component justified by an ablation |
| Pipeline [2] decomposer [G§2 p2] | Decomposer | — | — | Exp 4 | — |
| Pipeline [3] hybrid + rerank + dedup [G§2 p2] | Retrievers, Reranker, Fusion | RRF + dedup always; cross-encoder gated (K3) | `EVIDENCE_RERANKED.method` | Exp 2, 3 | Hybrid ≥ max(single) Recall@5 |
| Pipeline [4] incremental update + uncertainty [G§2 p2] | Refiner, Grounding | §14, §16 | — | G4, G5 | — |
| Pitfall 1: premature retrieval [G§6 p5] | Controller | Dangling, sufficiency, stability, budget | WAIT reasons | Category 3 | 0 retrievals on dangling fragments |
| Pitfall 2: context loss on late constraints [G§6 p5] | Refiner | Frame + carry-over | diff | Categories 6, 7 | Kept-claim fidelity 100% |
| Pitfall 3: citation hallucination [G§6 p5] | Citation Manager | Labels + manifest check | — | All | 0 invalid IDs |
| Pitfall 4: ignoring presentation-only turns [G§6 p5] | Analyzer, Controller | §15 | NO_RETRIEVE reasons | Category 5 | False triggers ≤ 5% |
| Pitfall 5: over-fragmenting [G§6 p5] | Decomposer guards | Overlap merge | `merged[]` | Category 2 | Sub-query precision ≥ 0.8 `[T]` |
| Deliverable: baseline comparison [G§8 p6] | Eval harness | B0 (hybrid static, K1) → B5 ladder (ADR-011) | Run manifests | Full suite | Report produced |
| Deliverable: ≥2 ablations [G§8 p6] | Eval harness | Exp 2 (hybrid vs dense), Exp 8 (rule vs model controller) | — | — | Report produced |
| Deliverable: ≥3 edge-case failures [G§8 p6] | Eval harness | Trace-backed failure analyses | Traces | Categories 3, 5, 7, 8 | ≥ 3 analyzed |
| Deliverable: telemetry schema [G§8 p6] | Schemas | JSON Schema export of §5–§6 | — | — | Published in `docs/` |
| Report recall, groundedness, TTFT, cost per turn [D s7] | Eval harness | §20, §22 metrics | `LLM_CALL`, timings | Full suite | All four reported |
| Full duplex scope [D s7] | Streaming core | §7 concurrency | — | G2 | = G2 |

---

## 22. Benchmark Dataset Specification

### 22.1 Principles

1. **Structure is defined now; content is authored after the corpus arrives.** No expected answers, gold chunk IDs or reference texts exist until they can be read from the actual corpus.
2. **Expected *behavior* can be labeled without the corpus** (turn type, retrieval required, earliest acceptable retrieval chunk, intents, constraints). **Expected *evidence and answers* cannot.**
3. **Gold evidence is at section granularity** (`Doc_ID §Section`), with optional chunk IDs. This makes labels robust to chunker changes (Exp 10).
4. **Test data lives only in `eval/`** (REQ-REPRO-006). The **tune** and **test** splits are separated by session. Thresholds are calibrated only on tune.

### 22.2 Test case schema (session-level, since refinement spans turns)

```json
{
  "case_id": "MI-0007",
  "schema_version": "1.0",
  "category": "multi_intent",
  "split": "tune",
  "description": "three coordinated needs with shared location and headcount",
  "corpus_ref": {"corpus_id": "theme4-official", "corpus_hash": "sha256:<filled at annotation>"},
  "session": {
    "session_id": "case-MI-0007",
    "turns": [
      {
        "utterance_id": "u1",
        "utterance_text": "<full utterance as authored>",
        "word_timing": [{"w": "<word>", "t": 0.00}],
        "chunks": [{"chunk_index": 0, "timestamp_s": 0.0, "text": "<delta>"}],
        "utterance_end_s": 2.1,
        "chunking": {"generator": "word_timing_v1", "seed": 7, "punctuation": "asr_style"},
        "expected": {
          "turn_type": "query",
          "retrieval_required": true,
          "suppression_reason": null,
          "earliest_retrieval_chunk_index": 1,
          "latest_acceptable_first_retrieval_s": 2.1,
          "intents": [
            {"gold_intent_id": "g1", "description": "<need, in annotator words>", "paraphrases": ["<alt>"],
             "shared_slots": ["location", "headcount"], "constraints": [{"slot": "headcount", "value": "<v>"}],
             "answerable": true, "gold_evidence": ["<Doc_ID §Section>"], "min_evidence": 1}
          ],
          "refinement": null,
          "citations": {"must_include_any_of": [["<Doc_ID §Section>"]], "must_not_include": [], "subset_of_previous": false},
          "uncertainty_expected_for": [],
          "clarification_expected": false,
          "reference_answer": null
        }
      }
    ]
  },
  "annotation": {"annotators": ["A1", "A2"], "agreement": null, "notes": ""},
  "provenance": {"authored_by": "team", "derived_from_sections": ["<Doc_ID §Section>"], "created": "YYYY-MM-DD"}
}
```

`reference_answer` stays `null` unless the annotators can write it entirely from cited corpus text, with citations. Refinement turns fill:

```json
"refinement": {"is_refinement": true, "refines_utterance": "u1", "delta": [{"slot": "<slot>", "op": "set|update|retract", "value": "<v>"}],
               "affected_gold_intents": ["g1"], "must_retain_citations": ["<Doc_ID §Section>"],
               "must_not_reissue_prior_queries": true, "expected_new_evidence": ["<Doc_ID §Section>"]}
```

### 22.3 Chunk-stream generation (reduces hand-crafting bias)

Annotators author **full utterances plus word timings**: either recorded speech forced-aligned, or synthetic timings at 2.3–3.0 words/s with jitter (seeded). A deterministic generator splits them into chunks:

- Chunk cut every 0.6–1.0 s (seeded), **not** aligned to phrase boundaries, so mid-phrase cuts happen naturally.
- `UTTERANCE_END` = last word + 0.4–0.8 s.
- Punctuation mode: `asr_style` (none or sparse) or `clean`.
- Optional disfluency injection (fillers, restarts) at a seeded rate.

Each authored utterance yields **≥2 chunkings** (different seeds). Results are reported per chunking, which measures robustness to chunk boundaries.

### 22.4 Categories

| # | Category | Definition | Mandatory expected fields | Primary metrics | Min. cases (tune + test) |
|---|---|---|---|---|---|
| 1 | Single intent | One information need | turn_type, retrieval_required, intents (1), gold_evidence | Recall@k, G2, G4 | 12 |
| 2 | Multi intent | ≥2 explicit coordinated needs | intents (≥2) with shared_slots and constraints | G3 (lenient and strict), sub-query P/R, coverage | 20 |
| 3 | Incremental intent | The need is revealed progressively; early chunks are fragments | earliest_retrieval_chunk_index; expected WAIT ticks | Premature-retrieval count, G2, useful-early rate | 12 |
| 4 | Early retrieval | A specific need stated early, followed by a long tail | earliest and latest acceptable first retrieval | Lead time, G2 | 10 |
| 5 | Retrieval suppression | Presentation, social, backchannel, meta (incl. the empty-session variant) | retrieval_required=false, suppression_reason, subset_of_previous | False-trigger rate, citation ⊆ check | 16 |
| 6 | Late detail | One later turn adds or changes a constraint | refinement{…} | G5 checks, stale-fact rate, delta accuracy | 12 |
| 7 | Multiple late details | ≥2 refinement turns (incl. update and retract) | refinement per turn; expected version chain | Lineage, kept-claim fidelity, re-issue count | 8 |
| 8 | Ambiguous query | Unresolved referent, missing decisive slot, self-correction ("…no wait…") | clarification_expected or uncertainty_expected_for; corrected value | Clarification or uncertainty rate; 0 fabricated resolutions | 10 |
| 9 | Irrelevant continuation | A real request followed (or preceded) by chit-chat or off-topic speech | intents limited to the real request; irrelevant spans marked | No extra intents; no extra retrievals; answer unchanged | 10 |
| 10 | Long streaming query | 15–40 s, 3–5 intents, digressions | intents (3–5), duplicate-query budget | G3, provisional budget adherence, coverage | 8 |
| 11 | Session isolation (extra) | Two concurrent sessions with disjoint topics | per-session expected citations | Cross-session leakage = 0 | 4 pairs |

**Total:** ≥ 120 sessions (≥ 60 tune / 60 test). Compound utterances (categories 2, 3, 10) number ≥ 30, so a G3 estimate has a usable confidence interval (bootstrap CI reported).

### 22.5 Annotation protocol

1. Annotators read corpus sections, then write utterances that a real user might say. They do **not** copy section text (avoids lexical-overlap bias, failure #8).
2. Each gold intent lists acceptable paraphrases and the section(s) that answer it, or `answerable=false` after a documented search.
3. 20% of cases are double-annotated. Agreement is reported for intents (matching) and evidence (section-level Jaccard).
4. Unanswerable intents (needed for uncertainty recall) make up ≥ 15% of intents.

### 22.6 Scoring procedures

| Metric | Procedure |
|---|---|
| **Intent matching (G3)** | Similarity matrix between predicted intent texts and gold descriptions + paraphrases (max cosine using a **separate** embedding model from the system's, to avoid self-preference), plus lexical F1. One-to-one Hungarian assignment with threshold τ_eval fixed before test scoring. A human audit of 20% of matches is reported. |
| **Early retrieval (G2)** | Eligible = `retrieval_required ∧ n_chunks ≥ 2`. Success = first `RETRIEVAL_STARTED.timestamp_s < utterance_end_s`. Also report "useful early" (≥1 cited item from an early retrieval). |
| **Citation correctness** | A citation is correct if its key ∈ the gold evidence of the claim's intent, **or** the verifier and a human judge mark it supporting (the gold set may be incomplete) |
| **G5** | Lineage check, re-issue count, kept-claim fidelity (exact text and citations for gold-unaffected claims), stale-fact rate via the gold `affected_gold_intents` |

### 22.7 File layout (Phase 3)

```
eval/
  schema/test_case.schema.json
  cases/{tune,test}/<category>/<case_id>.json
  chunkings/<case_id>.<seed>.jsonl      # generated input event streams
  gold/                                  # derived label tables
  runs/<run_id>/{manifest.json, trace.jsonl, turns.jsonl, metrics.json}
```

---

## 23. Failure Handling

### 23.1 Principles

1. **Degrade capability, never correctness.** A failure may make the answer less complete, never less grounded.
2. **Every failure is an `ERROR` event** with the action taken (G6).
3. **Bounded retries:** at most 1 per LLM call, 0 for retrieval. **Bounded waits:** every await has a timeout from config.
4. **Fail toward retrieval at the end of the utterance** (an unanswered information need is worse than a harmless extra search). **Fail toward suppression only with high-confidence act detection.**

### 23.2 Failure table

| # | Failure | Detection | Immediate behavior | Degraded mode | User-visible effect | Telemetry |
|---|---|---|---|---|---|---|
| 1 | **Embedding model fails** (load or inference) | Exception or NaN output | Current query → lexical-only | Process-wide `lexical_only` until restart; novelty uses Jaccard only | Possibly lower recall; answers still cited | `ERROR(dense_retriever, action=lexical_only)`; manifest flag |
| 2 | **Retriever timeout** (> `retrieval_timeout_ms` = 250) | Timer | Use whichever list finished (`status=partial`) | — | None or minor | `RETRIEVAL_COMPLETED.status=partial` |
| 3 | Both retrievers fail for a query | Exceptions | Retry once at finalize | If still failing → intent uncovered → uncertainty | "X could not be verified" | `ERROR` + uncertainty |
| 4 | **Reranker fails or times out** (> 600 ms) | Exception or timer | `rrf_dedup` for that intent | Disable the cross-encoder after 3 consecutive failures | None or minor | `EVIDENCE_RERANKED.method=rrf_dedup`; `ERROR` |
| 5 | **LLM fails** (error, timeout, rate limit, refusal) | Status or stop reason | Retry once if budget allows | Fallback chain: hosted → local → **extractive** (always available) | Possibly less fluent answer; still grounded | `LLM_CALL.status`, `ERROR(action=fallback_backend)` |
| 6 | LLM returns invalid labels or malformed structure | L0 / parse failure | Drop invalid labels; parse prose fallback | If no valid claim remains → extractive synthesis for this turn | — | `GROUNDING_CHECKED.dropped_labels` |
| 7 | **No evidence found** for an intent | `covered=false` | Mark NOT COVERED before synthesis | — | Explicit uncertainty sentence | `uncertainty[kind=no_evidence]` |
| 8 | No evidence for any intent | All uncovered | No factual claims | Clarification if the request seems ambiguous | Uncertainty and/or a question | kind=clarification |
| 9 | **Evidence conflicts** | `DETECT_CONFLICTS` | Present both with citations | — | "Sources differ: … [A] … [B]" | `uncertainty[kind=conflict]` |
| 10 | **Stream ends unexpectedly** (no `UTTERANCE_END`; EOF; `SESSION_END` mid-utterance) | Endpoint timer (2.0 s) / EOF / session end | Synthesize `UTTERANCE_END(reason=timeout|eof)`; finalize with the transcript so far | If no retrieval-worthy content → no answer | Answer to what was said | `UTTERANCE_END.reason` |
| 11 | **Duplicate chunks** (same index, same text) | Idempotency key | Ignore | — | None | `CHUNK_RECEIVED.status=duplicate_ignored` |
| 12 | Conflicting duplicate (same index, different text) | Key match + text differs | Treat as a revision; re-segment | Retrieval already issued for the replaced text stays in the pool; re-scored at finalize | None | `status=revision` |
| 13 | Out-of-order chunk | Index < max seen | Accept within the window of 1; re-segment | Beyond the window → accept and log | None | `status=out_of_order_accepted` |
| 14 | Chunk for an unknown or closed session | Lookup | Unknown → auto-create (log). Closed → reject. | — | — | `ERROR(recoverable)` |
| 15 | **Session state inconsistent** (invariant I-1…I-5 violated) | Post-commit invariant check | Freeze the offending version; emit `ERROR(fatal_for_turn)` | Session `degraded=true`: next turns are treated as **new topics** (no refinement from corrupt claims), the ledger is kept, the evidence store is re-validated against the index | Next answer is a fresh grounded answer; a note that refinement context was reset | `ERROR(session, action=reset_frame)` |
| 16 | Grounding validator fails | Exception | Commit `unverified=true` | — | Answer marked unverified, with uncertainty | `ERROR` |
| 17 | Telemetry sink fails | Write error | stderr fallback | — | None | Coverage flag in the manifest |
| 18 | Corpus index missing or corrupt at start-up | Hash mismatch or load error | Rebuild | Empty corpus → **fail fast** with a clear message | Start-up error | Start-up log |
| 19 | Config invalid | Schema validation | **Fail fast** | — | Start-up error | — |
| 20 | Race: a new utterance arrives while synthesizing | State check | Serialize the commit (§7.3). Correction barge-in → cancel per policy. | — | Short delay before the next answer | `LLM_CALL.status=cancelled` (if cancelled) |

---

## 24. Security & Data Isolation

This section is scoped to the hackathon prototype.

| Concern | Control | Verification |
|---|---|---|
| **Session isolation** | Actor per session; immutable `SessionView` snapshots; no module-level mutable state; read-only index; pure caches only (§13.3) | Category 11 concurrent test; state-graph reference audit; GC test after `SESSION_END` |
| **No cross-session persistence** | In-memory only; TTL; no DB; telemetry is write-only and never read by the pipeline | Filesystem diff after the run; import-lint rule (REQ-SESS-003) |
| **Corpus isolation** | Retrieval touches only the local index. The only permitted egress is the configured LLM endpoint (hosted mode). In extractive or local mode the container can run with networking disabled. | Egress test (network-disabled run passes; hosted run logs a single endpoint) |
| **No external knowledge leakage** | Prompt contract (evidence-only, labels); L1 numeric check and L2 support check; deterministic uncertainty for uncovered intents | Unsupported-claim rate; human audit sample |
| **Prompt injection via corpus or transcript** | Evidence is wrapped in delimited data blocks with an instruction that it is data. Transcript text is never concatenated into system instructions. Labels are validated, so an injected "[Doc_999]" cannot become a citation (it is not a label). | Injection fixtures: a corpus chunk with an instruction-like sentence; a transcript asking to ignore rules |
| **Sensitive data in logs** | `telemetry.text_mode` ∈ {full (default for the hackathon replay), hash, none}, applied to transcript and answer text; secrets come only from env and are never logged; HTTP headers are not logged | Secret scan of traces; config test |
| **Secrets** | `.env` (git-ignored), `.env.example` committed; keys read at start-up only | gitleaks-style pre-commit scan |
| **Deterministic reproducibility** | Seeds; temperature 0; pinned models and dependencies; sorted file and ID order; tie-breaks by `chunk_id`; scores rounded to 1e-6 before sorting (stabilizes BLAS thread variance); deterministic event IDs | Two-run trace diff of non-LLM fields (REQ-REPRO-004) |
| **Supply chain** | Model weights from pinned revisions at build time; lockfile hashes | Manifest review |

**Out of scope for the hackathon:** authentication, multi-tenant access control, encryption at rest, PII redaction models.

---

## 25. Final System Specification

### 25.1 Architecture summary

```
                    ┌────────────────────────────── Session actor (per session_id) ──────────────────────────────┐
 Transcript ─► Streamer ─► Chunk Manager ─► Intent Analyzer ─► Retrieval Controller ─► Query Decomposer ─► Ledger │
 (replay/live)            (segments)        (acts, slots,      (WAIT/RETRIEVE/          (intents, constraints,     │
                                             anchors)           NO_RETRIEVE)             context, guards)          │
                                                                                               │ ∥ per intent     │
                                    Lexical (BM25) ∥ Dense (bge-small) ─► RRF ─► Reranker (CE | RRF+dedup)       │
                                                                                               ▼                  │
   UTTERANCE_END ─► Finalize (gated LLM check, bounded waits) ─► Evidence Fusion (quota RR, dedup, conflicts)      │
                                                                                               ▼                  │
                    Answer Synthesizer (hosted → local → extractive; label-constrained; refine ops)              │
                                                                                               ▼                  │
                    Grounding Validator (sentence gate L0–L2[L3], repair, uncertainty) ─► Citation Manager       │
                                                                                               ▼                  │
                    Session State Manager (frames, ledger, evidence, claims, versions, invariants) ──────────────┘
                                                   │ every step emits events
                                                   ▼
                    Telemetry Manager (JSONL traces, TURN_COMPLETED, manifest) ─► Eval Harness / Demo SSE view
 Offline: Corpus Indexer (sections, chunks, embeddings, BM25, IDF vocab, near-dup groups) ─► read-only CorpusIndex
```

### 25.2 Technology stack (selected vs pending)

| Layer | Choice | Status | ADR |
|---|---|---|---|
| Language / runtime | Python 3.11 or 3.12, asyncio | Selected | — |
| Schemas | pydantic v2 models → JSON Schema export | Selected | ADR-012 |
| Lexical | Own BM25 on scipy sparse (as measured in Phase 1) | Selected | ADR-001 |
| Dense | `BAAI/bge-small-en-v1.5` (default); `all-MiniLM-L6-v2` (fallback) | **Pending Exp 1** (needs corpus + download approval) | ADR-002 |
| Vector store | numpy exact search, in-process | Selected | ADR-001 |
| Fusion | RRF (k=60) within intent; quota round-robin across intents | Selected | ADR-009 |
| Reranker | RRF+dedup (always); `cross-encoder/ms-marco-MiniLM-L-6-v2` (gated) | Cross-encoder **pending Exp 3** | ADR-003 |
| Embedder runtime | ONNX Runtime (preferred, small image) or torch-CPU ≥ 2.4 | Pending Phase 3 spike | ADR-002 |
| LLM | Adapters: Anthropic SDK (hosted, e.g. `claude-haiku-4-5`), Ollama (local small model), extractive | Interface selected; **primary backend pending Q2** | ADR-007 |
| Serving / demo | CLI replay + FastAPI SSE minimal view | Selected | — |
| Packaging | Dockerfile + compose (app; optional `ollama` profile) | Selected | ADR-007 |
| Testing | pytest; trace-based eval harness | Selected | — |

### 25.3 Configuration parameters (initial defaults; *calibrate* = set on the tune split)

| Group | Parameter | Default |
|---|---|---|
| Controller | `min_content_tokens` | 2 |
| | `idf_floor` | 1.0 (calibrate) |
| | `Δt_stable_s` | 0.6 |
| | `max_segment_tokens` | 25 |
| | `provisional_budget` | 4 |
| | `min_interval_s` | 0.4 |
| | `act_suppress_confidence` | 0.8 |
| | `endpoint_timeout_s` | 2.0 |
| Ledger / decomposition | `τ_dup` | 0.92 (calibrate) |
| | `jaccard_dup` | 0.8 |
| | `τ_merge` | 0.85 (calibrate) |
| | `overlap_merge_jaccard@5` | 0.6 |
| | `τ_match` (reconcile) | calibrate |
| | `llm_check_wait_ms` | 300 |
| | `τ_topic` | calibrate |
| Retrieval | `N_lex`, `N_dense` | 50 |
| | `k_rrf` | 60 |
| | `M_rerank` | 20 |
| | `retrieval_timeout_ms` | 250 |
| | `rerank_timeout_ms` | 600 |
| | `finalize_wait_ms` | 150 |
| | `θ_cov_ce` / `θ_cov_dense` | calibrate |
| Fusion | `q_per_intent` | 3 |
| | `section_cap_per_intent` | 2 |
| | `B_tok` | 2000 |
| | near-dup cosine | 0.97 |
| Grounding | `ρ_sup`, `ρ_partial` | calibrate vs human labels |
| | `on_unsupported` | drop_and_flag |
| | `stream_mode` | sentence_validated |
| LLM | temperature | 0 |
| | retries | 1 |
| | `ttft_timeout_s` | 5 (hosted) / 30 (local) |
| | `total_timeout_s` | 20 / 120 |
| | `cancel_on_correction` | false |
| Session | `ttl_idle_min` | 30 |
| | `max_evidence` | 500 |
| | `max_versions` | 50 |
| Telemetry | `text_mode` | full |
| | `verbosity` | standard |

### 25.4 Quality check (from the Phase 2 brief)

| Check | Status | Where |
|---|---|---|
| Every official Theme 4 requirement has a corresponding component | ✅ | §21 matrix (gates, capabilities, rules, pipeline, pitfalls, deliverables) |
| Every component has an input/output contract | ✅ | §18.1–§18.15 |
| Every important behavior is measurable | ✅ | Acceptance and measurement columns in §3–§4; §20.2; §22.6 |
| Streaming is treated as a first-class feature | ✅ | §5, §7 (concurrent regions), §8, §20 |
| Multi-intent retrieval is explicitly modeled | ✅ | §9, §12 |
| Retrieval suppression is explicitly modeled | ✅ | §15, §8.6 |
| Late-arriving details do not restart the pipeline | ✅ | §14 (delta-only, ledger, verbatim carry-over), REQ-SESS-005 |
| Session memory is isolated | ✅ | §13.3, §24, REQ-SESS-001…003 |
| Grounding is explicit | ✅ | §16, REQ-GRD-* |
| Citations are traceable | ✅ | §16.7 chain to chunk and source path |
| Latency is measurable | ✅ | §20; every stage emits `latency_ms` |
| No benchmark results have been fabricated | ✅ | Only `[M]` micro-benchmarks (scripts in `research/`); null placeholders in examples |
| No corpus facts have been invented | ✅ | Placeholder IDs (`Doc_A §4`) and `<…>` fields only |
| Architecture is implementable on the available hardware | ✅ | §20.4 measured composition on RH-dev; RH-judge sensitivity and mitigations |
| Design complexity is justified | ✅ | One process; ≤2 LLM calls per turn; each optional component gated by an ablation (Exp 3, 4b, 8, 9) |

---

## 26. Implementation Plan for Phase 3

### 26.1 Entry blockers (from §19 of Phase 1, still open)

| ID | Blocker | Blocks |
|---|---|---|
| Q1 | Corpus (official, or an approved labeled dev corpus) | M2 onward with real data. M0–M1 can proceed. |
| Q2 | Primary LLM backend + key | M4 hosted path only. The extractive path proceeds. |
| D | Download approval: embedder, cross-encoder, optional Ollama model, pip lockfile | M2–M3 |
| Q6 | Deadline | Scope cut line (marked ✂ below) |

### 26.2 Ordered milestones

Each milestone ends with tests passing and a measurable artifact.

| # | Milestone | Builds | Exit criteria |
|---|---|---|---|
| **M0** | Repo bootstrap | `git init`; `pyproject.toml` + lockfile; package skeleton (`src/streamrag/…`); `configs/default.yaml`; `.env.example`; pytest; guard test for no-hardcoding (REQ-REPRO-006); AI-usage log | `pytest` green on an empty skeleton; lockfile committed |
| **M1** | Contracts first | pydantic models for §5 input events, §6 envelope + 16 events, `TurnResult`, `IntentSet`, `Evidence`, `AnswerVersion`, `Claim`, test-case schema (§22.2); JSON Schema export to `docs/schemas/`; event bus + Telemetry Manager (JSONL, non-blocking, flush) | Round-trip and schema tests; a G6 validator script exists |
| **M2** | Corpus indexer | Loaders; section parser; ID rules (§10.1); chunker; BM25; embedder runtime (spike: ONNX vs torch); IDF vocabulary; near-dup groups; cache by hash; corpus audit report | Determinism test (two builds identical); audit report on the real or dev corpus |
| **M3** | Retrieval core + B0 baseline | Lexical, dense, RRF, filters, reranker (gated); Streamer (replay, both clocks); extractive synthesizer + one LLM adapter; label-constrained citations; Citation Manager; `TURN_COMPLETED`. **B0 = static turn-based hybrid** (K1). | First baseline metrics on the dev suite (Recall@k, G4 basic, TTFT); Exp 1–2 runnable |
| **M4** | Streaming core | Chunk Manager (segments, dangling, stability, revisions); Intent Analyzer (acts, slots, anchors); Retrieval Controller (§8); Ledger; Orchestrator state machine (§7) with concurrency and serialized commits | G2 early rate + false-trigger rate measurable; Exp 5, 7 runnable |
| **M5** | Multi-intent + fusion | Rule decomposer; guards; gated LLM check + reconciliation; quota fusion; conflict check; rerank-on-stability | G3 measurable; Exp 3, 4 runnable |
| **M6** | Grounding | Sentence gate; L0–L2 verifier; repair; `drop_and_flag`; deterministic uncertainty; clarification path | G4 measurable; Exp 9 runnable |
| **M7** | Session and refinement | Session actor, frames, claims, versions, diff, invariants; Δ extraction; affected claims; delta queries (scoped-first); verbatim carry-over; presentation and meta transforms; pending constraints | G5 checks pass on the dev suite; Exp 6 runnable |
| ✂ | *Minimum viable submission cut line: M0–M7 cover G2–G6 with real mechanisms* | | |
| **M8** | Packaging (G1) | Dockerfile, compose (+ optional ollama profile), model baking, keyless fallback, amd64 + arm64 builds, clean-machine script | Fresh-clone one-command replay passes with an empty env |
| **M9** | Evaluation and ablations | Full metric suite (§22.6); experiments 1–11 per P1 §15; ≥3 edge-case failure write-ups; report tables | Benchmark report draft with real numbers only |
| **M10** | Demo surface | CLI timeline view; FastAPI SSE page showing chunks → decisions → retrievals → versions → citations | Demo script recorded (≤ 5 min video plan) |

**Ordering rationale.**

- Contracts (M1) before components, so every later piece emits valid telemetry from day one. G6 then holds by construction.
- B0 (M3) before streaming (M4), so every later milestone shows a measurable delta against the official-style hybrid baseline.
- Grounding (M6) before refinement (M7), because refinement relies on validated claims.
- Packaging (M8) is not left to the very end: a smoke Dockerfile SHOULD exist from M3 onward, to surface arm64/amd64 issues early.

**Phase 3 will not begin until instructed.**
