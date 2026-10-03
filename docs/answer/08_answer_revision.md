# 08: Answer Versions, Diffs, Stability, Consistency (Phase 6 integration)

**Code:** `answer_state/engine.py`, `answer_state/models.py` (`GroundedAnswer`, `AnswerClaim`, `AnswerClaimDiff`), `validation/consistency.py`
**Schema:** `docs/schemas/GroundedAnswer.schema.json`

## Versions

**`GroundedAnswer`** is one version of a topic frame's answer. It holds:
- identity: `answer_id` (`GA<n>`), `version` (per frame), `parent_answer_id`, `phase6_answer_id`, `utterance_id`;
- status: `status` (DRAFT / VALIDATED_FINAL / BLOCKED), `partial`, `mode`;
- content: sections, claims, verifications, rejected claims, repairs, the citation map plus its report, coverage, consistency conflicts;
- `diff`, the rendered `text`;
- cost: backend / model / fallback, LLM calls, measured prompt / output tokens, raw time to first token, validation retrievals, revision attempts, stage timings, metrics.

**Claim ids** are stable: `AC-` plus a hash of (section lineage, normalised text). An unchanged sentence keeps its id and citations across versions.

## Incremental refinement (brief §41)

The Phase 6 answer state decides what changed. The grounded engine then:
1. **Reuses whole sections** that Phase 6 marks unchanged (their claims are re-verified from the cache; no generation).
2. **Keeps sentences inside changed sections** whose facts are still planned and whose evidence is still usable (re-verified).
3. **Generates only the missing facts.** The kept sentences are shown to the model as ALREADY WRITTEN, so no LLM call happens when nothing new must be said.
4. **Re-checks consistency** (below), rebuilds citations and computes the diff.

Example (fixture, local LLM, smoke run):
- "Specifically overnight." after the ladder question made **0** LLM calls, kept all three sentences re-ordered by the new importance, and created version 2.
- The full-restart baseline regenerated the whole answer (1 LLM call, about 2.2 s).

## Answer diff (brief §36-37)

`AnswerClaimDiff`:
- `added`, `removed` and `unchanged` claim ids;
- `modified`: (old id, new id) pairs expressing the same planned facts in new words;
- `sections_regenerated` and `sections_reused`.

Section `status` is new / changed / unchanged against the previous final version. Drafts are diffed against the previous version of any status; a final is diffed against the previous final.

## Consistency check (brief §42)

- **Pairing.** New claims are paired with the claims the answer keeps, and with each other, when they share at least half of the shorter claim's content terms (same proposition).
- **Detection.** NLI labels each pair in both directions; a contradiction is a conflict.
- **Two sources disagree:** both claims are supported by *different* evidence. This is an evidence conflict, presented with both sides.
- **Otherwise:** the new claim is withdrawn (`CONSISTENCY_CONFLICT`).
- **Gating.** Without the shared-content gate, NLI flagged two unrelated ladder rules as contradictory in the first integration run.

## Conflict groups

Conflict edges come from three sources:
- the verifier (claims supported by one source and contradicted by another);
- Phase 6 numeric conflicts between planned facts;
- the consistency check.

Groups are their connected components. A side shared by two pairs therefore joins one group ("mandatory" / "optional" / "must provide" from three documents), and a lone side whose partner was removed becomes a plain fact.
