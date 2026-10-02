# 01: Intent Model

**Code:** `models/intents.py` (contracts), `intents/tracker.py` (ids, versions)
**Schemas:** `docs/schemas/Intent.schema.json`, `IntentSet`, `IntentSetDelta`, `IntentQuery`
**ADRs:** ADR-015 (Phase 5), ADR-009 (fusion)

## Definition

A **multi-intent utterance** contains two or more retrieval-relevant information requests that may need different queries or evidence.

Neither of the following is an intent:
- **A linguistic component.** "tell", "about" and "information" are not needs.
- **A constraint.** "especially during a storm" narrows a need; it is not a need itself.

Everything else follows from this definition:
- **Request cue required.** A clause yields intents only when it carries a request cue.
- **Splitting.** Coordinated phrases become sibling intents only when every conjunct carries substantive content.
- **Constraints.** Constraint phrases attach to intents; they never become intents.

## `Intent`

| Field | Meaning |
|---|---|
| `intent_id` | Session-scoped, monotonic (`I1, I2, …`). Follow-up utterances can depend on earlier intents. |
| `utterance_id`, `session_id` | Parent utterance and session |
| `version` | Revision of this intent. Text, constraint or context changes → +1 (drives delta retrieval). |
| `text` | The need as spoken, with fillers and request framing removed ("the requirements for ladders") |
| `resolved_text` | `text` with anaphora replaced by the antecedent's words ("how does **lens** apply") |
| `components` | `resolved_text` as verbatim pieces, each with a `SourceSpan` (this or an earlier utterance) |
| `intent_type` | One of 8 types (below) |
| `type_cues` | The words that established the type |
| `entities` | Content phrases, verbatim |
| `topic`, `topic_span` | What the need is about. A follow-up inherits it from `topic_span`. |
| `aspect` | The facet asked for ("requirements", "schedule") |
| `constraint_ids` | Constraints that apply (local and global in scope) |
| `inherited_context[]` | Words borrowed from elsewhere, each with a `reason`, `from_intent` and `source_span` |
| `unresolved_references` | Anaphora that could not be resolved ("there") |
| `order` | Order of mention, used for answer organisation |
| `priority`, `priority_score` | Retrieval priority (docs/multi_intent/05). Not text order. |
| `confidence`, `confidence_signals` | Computed (formula below), never authored |
| `status` | `ACTIVE` · `SUPERSEDED` (a correction replaced it) · `MERGED` (duplicate) · `DROPPED` (budget, or vanished from the decomposition) |
| `provenance` | `source` (`rule` · `rule_fallback` · `llm` · `reconciled`), `split` (`single` · `clause` · `coordination` · `correction` · `fallback`), `clause_index`, `span` |
| `supersedes`, `superseded_by`, `lineage_root` | Correction lineage |
| `created_at_ms`, `updated_at_ms` | Stream time |
| `query_ids` | Ledger queries that served this intent |

**Traceability invariant** (enforced by `intents/validation.py`, tested): every `SourceSpan.text` equals the transcript substring it points to.

## Intent types (8)

| Type | Typical cues | Use |
|---|---|---|
| `REQUIREMENT` | requirements, allowed, must, eligibility, rules | Answer organisation; cue words stay in the query |
| `TIMELINE` | how long, when, how often, schedule, deadline | Same |
| `PROCEDURAL` | how do, steps, process, procedure | Same |
| `EXCEPTION` | exception, except, exempt | Same |
| `COMPARISON` | difference, compare, versus | A comparison is **one** need and is never split |
| `DEFINITION` | define, meaning, what does … mean | — |
| `FACTUAL` | how many, how much, what, which, where | — |
| `OTHER` | "tell me about X" | — |

- The first matching type in `configs/intent_lexicon.yaml` order wins.
- Types do not inject words into queries. Their cue words are already part of the spoken need and stay in the query (the dense embedding benefits from "how long …"; BM25 drops stopwords).

## Confidence (computed)

```
confidence = 0.35*explicit + 0.30*specificity + 0.20*separation + 0.15*resolution
```

| Signal | Values |
|---|---|
| `explicit` | 1.0: own request cue · 0.8: shares a head (coordination) or follows an addition marker · 0.5: fallback |
| `specificity` | Anchor strength: max corpus IDF of the intent's terms ÷ IDF of a term found in one chunk (≤ 1) |
| `separation` | 1.0: clause boundary or single intent · 0.8: coordination split · 0.6: fallback |
| `resolution` | 1.0: self-contained · 0.85: uses inherited context · 0.7: unresolved reference |

`decomposition_confidence = mean(intent confidence) × min(constraint scope_confidence)`. An ambiguous constraint scope lowers it. A test checks the formula against the reported signals.

## `IntentSet`

| Field | Content |
|---|---|
| `session_id`, `utterance_id`, `version` | Identity and version |
| `source` | Decomposition source |
| `original_text` | The transcript decomposed |
| `intents` | Active intents, in order of mention |
| `global_constraints`, `local_constraints` | docs/multi_intent/03 |
| `relationships` | docs/multi_intent/02 |
| `decomposition_confidence` | As above |
| `superseded` | Intent ids superseded in this utterance |
| `merged[]` | `{text, into, reason}` |
| `dropped[]` | `{text, intent_id, reason}`: nothing is discarded silently |
