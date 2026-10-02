# 06: Evidence Fusion

**Code:** `fusion/engine.py` (`EvidenceFusionEngine`), `fusion/models.py` (`UnifiedEvidenceSet`)
**Config:** `fusion:` (`strategy`, `top_k`, `min_per_intent`, `section_cap_per_intent`, `rrf_k`, `rerank`, `conflict_check`)
**ADRs:** ADR-009 (quota round-robin), ADR-015

## Input → output

- **Input:** for each active intent, the evidence of its newest *completed* query. A stale earlier version is used while the newest is still running (flagged `stale`). Each intent also carries its status (`ok`, `pending`, `retrieval_failed`, `no_candidates`).
- **Output:** `UnifiedEvidenceSet`. Phase 5 ends here; there is no answer generation.

## Steps

1. **Provenance hits.** Each retrieved chunk becomes an `EvidenceHit`: `intent_id`, `query_id`, `retrieval_method`, rank, score, BM25/dense ranks, RRF/rerank scores, `stale`.
2. **Cross-intent dedup.**
   - The same `chunk_id` from several intents becomes **one** `FusedEvidence` with `supporting_intents` and `supporting_queries`.
   - A chunk recorded by Phase 3 dedup as a near-duplicate *alternate* of a kept chunk collapses into it across intents (`DUPLICATES` relation).
3. **Optional cross-intent rerank** (docs/multi_intent/07).
4. **Selection strategy** under a global budget `top_k`.
5. **Labels** `E1..En`: ordered by the first selecting intent's order of mention, then rank.
6. **Coverage per intent**, **relations** (`DUPLICATES`, same-section `RELATED`, `CONTRADICTS`) and the **conflict check**.

## Strategies compared (brief §23; research/phase5)

| Strategy | Selection | Known weakness |
|---|---|---|
| `concat` | Intent lists one after another, duplicates kept | The first intent takes the budget, so later intents starve; duplicates are visible |
| `global_score` | Union ranked by each chunk's best retrieval score | Scores are not comparable across queries (RRF/rerank scales) |
| `rrf` | Union ranked by Σ over intents of 1/(k + rank) | Rewards chunks shared by many intents over an intent's own best chunk |
| `intent_aware` | Coverage floor, then fill (below) | — |

**`intent_aware` selection:**
1. **Floor:** `min_per_intent` (2) round-robin picks per intent, in priority order. A chunk already chosen for another intent counts for both.
2. **Fill:** up to `top_k` new items, ordered by (rank within its intent, intent priority).
3. **Diversity:** a section cap per intent (2).

**Decision (ADR-015):**
- `intent_aware` is the default. It is the ADR-009 quota round-robin, adapted to a global item budget, and is the only strategy that guarantees each intent a share by construction (unit-tested with a strong intent that would otherwise take the whole budget).
- On the fixture dev suite `global_score` tied with it. That suite cannot separate them, so the choice rests on the construction guarantee, not on a measured win. Details are in Phase 5 report §11.

## Conflicts (conservative, documented limits)

A **potential** conflict is flagged when two selected chunks meet all of these:
- they come from different documents;
- they support the same intent;
- they state a number with the same unit and different values;
- they share a content word near the number. The unit itself does not count; this was a false-positive source found by a unit test and fixed.

Effects:
- `EvidenceConflict(status="potential")` and `CONTRADICTS` relations are recorded. The items are kept, never merged silently.
- **Not detected:** non-numeric contradictions ("permitted" vs "not permitted" in different documents). No NLI model is used (ADR-009). Phase 3 dedup already prevents merging chunks that differ in negation or numbers.

## `UnifiedEvidenceSet`

| Field | Content |
|---|---|
| `unified_set_id`, `session_id`, `utterance_id`, `intent_set_version` | Identity |
| `strategy`, `rerank`, `top_k` | How it was produced |
| `items[]` | `FusedEvidence`: `label`, `evidence_id`, `citation`, text and spans, `supporting_intents`, `supporting_queries`, `selected_for`, `hits[]`, `best_rank_by_intent`, `intent_relevance`, `alternates`, `conflict_ids` |
| `per_intent[]` | `IntentCoverage`: `status`, `candidates`, `evidence_ids`, `labels`, `covered`, `best_rank` |
| `relations[]`, `conflicts[]` | As above |
| `dedup` | `input_hits`, `unique_chunks`, `cross_intent_duplicates`, `near_duplicates_merged`, `duplicate_items` |
| `token_count`, `timings_ms`, `warnings` | Size, timings, warnings |
