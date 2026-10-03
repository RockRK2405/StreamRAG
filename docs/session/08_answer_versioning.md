# 08: Answer Versioning and Incremental Refinement

**Code:** `answers/manager.py` (`AnswerStateManager`), `models/answers.py` (`AnswerVersion`, `AnswerSection`, `AnswerDiff`, `ClaimChange`, `UncertaintyItem`)
**Schema:** `docs/schemas/AnswerVersion.schema.json`

An answer is the **structured state** of one topic frame. Phase 7 renders text from it; `AnswerVersion.text` stays `""` in Phase 6.

## Structure

- **`AnswerVersion`**
  - Identity and lineage: `answer_id` (`A<n>`), `version` (per frame), `parent_version`, `supersedes_answer_id`, `topic_id` (frame), `utterance_id`, `session_version`, `kind` (`initial` | `refinement`).
  - Content: `sections`, `claim_ids`, `evidence_ids`, `citations`, `uncertainty`, `frame_slots` (active constraint id → text).
  - Change record: `diff`, `change_summary`, `delta_queries`, `full_rerun`.
- **`AnswerSection`**: one per active need of the frame, in order of mention. It lists the need's usable selected claims (SUPPORTED / PARTIALLY_SUPPORTED), their evidence, the need's active constraints, and uncertainty items:
  - `no_evidence`;
  - `constraint_not_covered`: every claim only partially supported;
  - `conflict`.

  A section's `status` is `new` / `changed` / `unchanged`. **`needs_regeneration`** is true only for new or changed sections: the exact unit Phase 7 must re-render (minimal regeneration).

## API (brief §25)

| Method | Behaviour |
|---|---|
| `create_answer_state(...)` | first version of the active frame |
| `update_answer_state(uid, session_version, now, changes, delta_queries, full_rerun)` | builds the candidate, diffs it against the frame's current version, and commits **only if the diff is non-empty** (or it is the first version); else `None` |
| `get_current_answer_state(frame_id=None)` | latest version of the (active) frame |
| `compare_answer_versions(a, b)` | `AnswerDiff` |

## Answer diff

- **Claim lists:**
  - `kept`: same claim; same status, text and evidence as of the previous version;
  - `modified`: `ClaimChange` with from / to status and citations;
  - `added`, `retracted`.
- **Citation and evidence changes:** `citations_added` / `_removed`, `evidence_added` / `_removed`.
- **Section changes:** `sections_added` / `_removed` / `_changed` / `_unchanged`.
- **Uncertainty changes:** `uncertainty_introduced` / `_resolved`.

A new version of the need alone does not modify a claim. It is modified only if its validity or support changed.

## Measured: the 7 brief §39 cases (`research/phase6/results/answer_refinement.json`; fixture, NOT REPORTABLE)

| Case (dev session, turn) | Incremental: new retrievals / affected / revalidated / unchanged claims / answer | Full restart: retrievals / revalidated / answer |
|---|---|---|
| 1 no meaningful change (S17 "Okay.") | 0 / 0 / 0 / 2 / no new version | 0 / 0 / no rebuild |
| 2 new constraint (S01 "Specifically overnight.") | 1 / 2 / 2 / 1 / A2: 1 kept, 1 modified (SUPPORTED → PARTIALLY_SUPPORTED) | 1 / 2 / new answer, 2 added |
| 3 new intent (S16) | 1 / 0 / 2 / 0 / new frame answer | 1 / 2 / same |
| 4 entity correction (S10) | 1 / 2 / 4 / 0 / 2 retracted, 4 added | 1 / 4 / 4 added |
| 5 intent refinement (S25 "In the orchard.") | 1 / 2 / 2 / 2 / 2 kept | 1 / 2 / 2 added |
| 6 follow-up (S13 "What about crates?") | 1 / 0 / 4 / 2 / S-I1 unchanged, only S-I2 re-rendered | **2** / 6 / both sections re-rendered |
| 7 evidence contradiction (S28) | 1 / 0 / 3 / 0 / uncertainty `I2:conflict` introduced | 1 / 3 / same |
