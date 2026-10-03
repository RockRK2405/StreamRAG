# 05: Evidence Lifecycle and Validity

**Code:** `delta/evidence.py` (`EvidenceStore`, `EvidenceValidityManager`), `delta/models.py` (`EvidenceRecord`, `EvidenceAssignment`, `EvidenceTransition`, `EvidenceAction`)
**Schema:** `docs/schemas/EvidenceAssignment.schema.json`

## Records vs assignments

- **`EvidenceRecord`**: one per retrieved chunk. Holds document, section, citation, text, and the first query that found it. Records are never deleted.
- **`EvidenceAssignment`**: one per (evidence, need). Holds `status`, `intent_version`, `query_ids`, `best_rank` and the full `history` of `EvidenceTransition`s. Each transition records from / to, the rule name, change id, query id and time.

The same chunk can be ACTIVE for one need and SUPERSEDED for another.

## Statuses

| Status | Meaning | Usable by claims / answer |
|---|---|---|
| `ACTIVE` | retrieved for the need's current version, or confirmed after a change | yes |
| `RETAINED` | carried over a change it is still applicable to (general evidence) | yes |
| `REVALIDATION_REQUIRED` | the change puts its applicability in question; the delta retrieval decides | no |
| `STALE` | not confirmed, or its need was removed | no |
| `SUPERSEDED` | its need was corrected / entity replaced | no |
| `INVALID` | the source chunk is no longer in the index | no |

Plan decisions map onto statuses:
- **RETAIN**: ACTIVE or RETAINED;
- **REVALIDATE**: REVALIDATION_REQUIRED;
- **SUPERSEDE**: STALE or SUPERSEDED;
- **DISCARD**: INVALID.

## Validity rules (analyzed terms; rule names appear in events and history)

| Change | Condition | → status (rule) |
|---|---|---|
| CONSTRAINT_ADDITION (terms T, head h = last term) | evidence has all of T | ACTIVE (`constraint_covered`) |
| | has h but not all of T | REVALIDATION_REQUIRED (`constraint_dimension_other_value`) |
| | none of T | RETAINED (`general_evidence_still_applicable`) |
| CONSTRAINT_REMOVAL (removed terms T) | has all of T | REVALIDATION_REQUIRED (`specific_to_removed_constraint`) |
| | otherwise | RETAINED (`general_evidence_still_applicable`) |
| REFINEMENT (added terms A, topic terms N) | has some of A | ACTIVE (`covers_refinement`) |
| | A adds topic terms and evidence has none of N | REVALIDATION_REQUIRED (`does_not_cover_refined_topic`) |
| | otherwise | RETAINED |
| QUESTION_CHANGE | all | REVALIDATION_REQUIRED (`question_changed`) |
| ENTITY_CHANGE (new topic N) | mentions N | REVALIDATION_REQUIRED (`mentions_new_entity`) |
| | otherwise | SUPERSEDED (`specific_to_replaced_entity`) |
| CORRECTION | old need's evidence | SUPERSEDED (`intent_superseded`) |
| | evidence mentioning the new need's topic | carried to the new need as REVALIDATION_REQUIRED (`carried_to_correction`) |
| INTENT_REMOVAL | all | STALE (`intent_removed`) |
| after a need's delta retrieval | REVALIDATION_REQUIRED retrieved again | ACTIVE (`confirmed_by_delta_retrieval`) |
| | not retrieved again | STALE (`not_confirmed_by_delta_retrieval`) |
| | STALE / RETAINED retrieved again for the need | ACTIVE (`re_retrieved` / `re_cache_hit`) |
| source check | chunk missing from the index | INVALID (`source_unavailable`) |

Brief CASE 6 is tested with fixture strings. Q1 retrieved E1, E2 and E3, and the constraint "for the night shift" touches only E2 ("day shift"):
- E2 → REVALIDATION_REQUIRED;
- E1 and E3 → RETAINED;
- a delta retrieval that does not return E2 → E2 STALE.

## Measured (dev suite, fixture domain, NOT REPORTABLE)

On the 32 follow-up turns that changed something:
- **Incremental:** 93 (evidence, need) assignments stayed usable without retrieval, and 45 new evidence records were added.
- **Full restart:** retains nothing and re-fetched 161 records.

Every expected revalidation in the gold was observed (recall 1.000; 4 expectations). The full restart has no revalidation by construction: it re-retrieves.
