# 04: Delta Planning, Delta Queries, Cache

**Code:** `delta/planner.py` (`DeltaPlanner`, `DeltaQueryGenerator`, `SemanticCache`), `delta/models.py` (`DeltaPlan`, `QueryAction`, `EvidenceAction`)
**Schema:** `docs/schemas/DeltaPlan.schema.json`

The principle: CHANGE → IMPACT ANALYSIS → TARGETED UPDATE. The planner never restarts the pipeline.

## Change taxonomy → impact

| Change type | Detected when | Needs re-queried | Evidence rule (doc 05) | Claims re-checked |
|---|---|---|---|---|
| `NO_CHANGE` | gate closed, or interpretation unchanged | none | none | none |
| `REFINEMENT` | same need, terms only added (or same topic) | that need | covers_refinement / refined topic | that need's |
| `CONSTRAINT_ADDITION` | a constraint id added to a need | that need (delta query) | covered / other value of the dimension / general | that need's |
| `CONSTRAINT_REMOVAL` | a constraint retracted or replaced | that need (often a cache hit) | specific to removed → revalidate | that need's |
| `NEW_INTENT` | a new need (follow-up or independent) | the new need only | none (new evidence) | none |
| `INTENT_REMOVAL` | a need vanished from the re-decomposition | none | STALE | that need's → STALE |
| `CORRECTION` | explicit correction superseding a need ("I meant crates instead of ladders") | the corrected need | old → SUPERSEDED; new-topic evidence carried as REVALIDATION_REQUIRED | old need's → SUPERSEDED |
| `ENTITY_CHANGE` | same need, topic terms replaced | that need | mentions new entity → revalidate, else SUPERSEDED | that need's |
| `QUESTION_CHANGE` | same need, aspect / type changed | that need | all → revalidate | that need's |

## `DeltaPlan`

Fields:
- `plan_id` (`P<n>`), `change_ids`, `affected_intents`, `new_intents`
- `queries_to_create`: action `retrieve`
- `queries_to_reuse`: action `reuse_active` or `cache_hit`
- `queries_to_supersede`
- `evidence_to_retain` / `_revalidate` / `_discard`, as `evidence@intent`
- `claims_to_revalidate`, `claims_unaffected`
- `full_restart`

For each need to update, the planner builds its next query (below) and decides:
1. **`reuse_active`**: the semantic key equals the need's active query key, so nothing retrieval-relevant changed.
2. **`cache_hit`**: an earlier completed query of the session had this key. A `reused` ledger record points to it, and its evidence is applied without retrieval. Example: retracting a late detail returns the need to its first state.
3. **`retrieve`**: a new query whose `parent_query_id` and `supersedes_query_id` are the need's active query and whose `derived_from_change_id` is the change.

## Delta query generation

`DeltaQueryGenerator.generate(need, active constraints, previous query)`:
- start from the previous query's components;
- drop components of retracted constraints;
- append components of new constraints;
- rebuild the need's own words only if they changed.

A test checks that the delta query has the same analyzed terms as a from-scratch build.

```
Q1 "What are the rules for ladders in the orchard"
+ K1 overnight          -> Q2 "What are the rules for ladders in the orchard overnight"   parent Q1, from CH2
- K1 (retracted)        -> semantic key = key(Q1)  -> cache_hit, record Q3 reused_from Q1  (0 retrievals)
```

**Delta scope.** `session.delta_scope: corpus` (default) searches the whole corpus. `session_docs_first` is reserved for a document-scoped search and is not implemented (the retrieval service has no document filter yet).

## Semantic cache and invalidation

Key = `sha1(index content hash | retrieval-options hash | sorted set of analyzed query terms)`. Not the raw string: "Tell me the orchard ladder rules." hits the cache entry of "What are the rules for ladders in the orchard?" (dev S19, 0 retrievals).

| Event | Rule |
|---|---|
| corpus / index snapshot changes | every key changes (hash in key); `invalidate("index_changed")` clears entries |
| retrieval configuration changes | every key changes (options hash in key); `invalidate("config_changed")` |
| a source chunk is no longer available | entries whose evidence includes it are dropped (`invalidate_evidence`) and assignments become INVALID (`check_sources`) |
| entity / constraint / question change | different terms → different key; no rule needed, nothing over-invalidated |

The full-restart baseline disables the cache.

## Measured (dev suite, fixture domain, NOT REPORTABLE)

Over 64 turns:
- 59 retrievals, 3 cache hits;
- retrieval-decision accuracy 1.000 against gold "required / reuse / none";
- the full-restart baseline made 65 retrievals, 3 of them judged unnecessary by the gold.
