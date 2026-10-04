# Evaluation dataset `streamrag_eval_v1` (Phase 10)

> **TEST FIXTURE ONLY — NOT REPORTABLE.** Every corpus is fictional and small; every label is implementer-written.
> The official Theme 4 corpus is still unavailable (OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE).

| | |
|---|---|
| Files | `experiments/datasets/streamrag_eval_v1/{test,dev}.jsonl`, `manifest.json` (sha256 per split) |
| Builder | `experiments/datasets/build_dataset.py` (deterministic; re-running reproduces the files byte for byte) |
| Schema | `src/streamrag/evaluation/dataset.py` (`EvalSample`) |
| Size | **test: 87 scored turns** in 70 sessions (50 single-turn questions, 15 two- or three-turn conversations, 5 streamed ASR corrections) · **dev: 129 turns** in 112 sessions |
| Corpora | test: `tests/fixtures/corpus_eval_transit` (15 documents, 43 chunks, new in Phase 10); dev: `corpus_adaptive` (Phase 9), `corpus`, `corpus_grounding`, `corpus_conflict`, `corpus_injection` (Phases 3–7) |

## Fields (one record = one scored user turn)
`query` (final transcript of the turn), `stream` (chunks as they arrive; `replaces` = an ASR revision of an earlier
chunk), `conversation_context` (earlier utterances of the session), `expected_intents`, `required_entities`,
`required_constraints`, `ground_truth_evidence` (section citations; `gold_semantics: any` for ambiguous questions,
where any of several sections answers), `expected_claims` (reference statement + citation + **key strings** an answer
stating the claim must contain), `expected_answer` (reference answer), `expected_state` (SUFFICIENT / INSUFFICIENT /
CONTRADICTORY), `conflict_values`, `forbidden` (regexes of stale / unsupported values an answer must not assert),
`source_documents`, `difficulty`, `difficulty_features`, `query_type`.

## Categories (test split, all 14)
SIMPLE 15 · EXACT_TERM 6 · SEMANTIC 5 · MULTI_INTENT 5 · MULTI_CONSTRAINT 8 · TEMPORAL 6 · CONTEXTUAL_FOLLOWUP 5 ·
ENTITY_CORRECTION 5 · MULTI_HOP 7 · AMBIGUOUS 5 · CONTRADICTORY 5 · INSUFFICIENT_EVIDENCE 5 · REPEATED_QUERY 5 ·
STREAMING_CORRECTION 5. (First turns of conversations carry their own type, which is why SIMPLE / MULTI_CONSTRAINT are
larger.) The dev split covers 13 categories (no STREAMING_CORRECTION; few ENTITY_CORRECTION / AMBIGUOUS).

## Difficulty (computed, never assigned)
`points = (needs − 1) + 2·multi-hop + [≥1 constraint] + [≥2 constraints] + temporal + context-dependent + conflicting
evidence + insufficient evidence + streaming revision + lexical gap`, where *lexical gap* = fewer than 34 % of the
query's content terms occur in its gold evidence (index analyzer, computed at build time). EASY = 0, MEDIUM = 1,
HARD = 2–3, VERY_HARD ≥ 4. Test: EASY 27, MEDIUM 42, HARD 18, VERY_HARD 0 — no test item scores ≥ 4, so VERY_HARD
results are reported as "no samples". A test re-derives every stored difficulty from its features.

## Construction and annotation
* **Test split (held out):** a new fictional domain (Northvale transit: fares in two versions with an effective date
  different from the publication date, concessions by applicant type, a station → zone → supplement chain, two
  contradicting night-bus notices, acronym TCO, forms, refunds, penalties, accessibility, bicycles, holiday service).
  Documents and questions were written **before any system was run on them**, and no component was developed or
  tuned on this corpus. Each experiment runs it once with frozen code; system behaviour seen on test outputs was not
  used to change the system (framework bugs were fixed — listed in the report).
* **Dev split:** the Phase 9 dev and held-out sets and the Phase 7 grounded cases, converted (ids prefixed `p9d-`,
  `p9h-`, `p7-`). They were used while building Phases 7–9, so dev numbers are optimistic. Dev labels are
  retrieval labels (gold sections for 112 of 129 turns, 14 with forbidden patterns); 22 turns carry reference claims
  but none carries key strings, so the answer-level label metrics (answer correctness, completeness) are NOT
  MEASURED on dev — dev is used for retrieval-side results only.
* **Annotation method:** one annotator (the implementer) read the corpus text and wrote gold sections, reference
  claims with key strings, reference answers, forbidden patterns and expected states. No second annotator, no
  agreement statistic. Key strings make answer checks deterministic but literal (a paraphrase like "two hours" for
  "120 minutes" counts as missing).

## Splits and leakage
No component is trained, so there is no train split; *dev* = everything used during development, *test* = never
seen. Checks (`tests/evaluation/test_dataset_and_leakage.py`): test questions / answers / claims do not occur in
`src/` or `configs/` (incl. prompt templates); no system module imports the evaluation package or reads dataset files;
test documents share no ids with development documents and no near-duplicate text (highest 5-word-shingle Jaccard
0.125: two holiday notices that list the same three dates; threshold 0.3).

## Limitations and potential bias
Implementer labels (the same person wrote the system), English only, tiny corpora (retrieval is easy; k = 5 covers a
large share of a 43-chunk index), templated question styles, typed (not ASR) text, a single domain per split, no
real users. Category sizes of 5–8 make per-category numbers anecdotal.
