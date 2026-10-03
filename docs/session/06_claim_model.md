# 06: Claim Model and Claim-Evidence Graph

**Code:** `models/answers.py` (`Claim`, `ClaimSource`, `ClaimEvidenceLink`, `ClaimTransition`), `claims/graph.py` (`ClaimExtractor`, `ClaimGraph`)
**Schema:** `docs/schemas/Claim.schema.json`

Phase 6 tracks answers at claim level. It does **not** generate answer text: that is Phase 7.

## What a claim is (Phase 6)

A claim is an **extractive** unit: a *verbatim sentence* (or list item) of a usable evidence item of one need. Because the claim text is a substring of the evidence text (`ClaimSource`: evidence id + character span), SUPPORTS is justified by construction. No claim is invented, and a test checks `evidence.text[start:end] == claim.text`.

**Extraction** (`ClaimExtractor.select`) uses IDF-weighted relevance: the IDF of the query terms the sentence contains, divided by the IDF of all in-corpus query terms. Query terms here are the need's words plus inherited context and constraints. A sentence qualifies when both hold:
- relevance ≥ `claim_min_relevance` (0.2);
- it shares ≥ min(2, |query terms|) terms with the query, **or** contains an *anchor* term. An anchor is a term of the need's topic, or else the query's max-IDF terms, ties included.

One shared word that is neither (a place name every chunk mentions) does not make a sentence a claim. At most `claims_per_intent` (4) claims are selected per need version.

**Stable ids.** The id is stable per (need lineage, evidence id, span). A constraint change re-uses the same claims and only re-evaluates them; a correction starts a new lineage, so it gets new claims.

## `Claim` fields

| Field | Meaning |
|---|---|
| `claim_id`, `text`, `type` | `C<n>`, verbatim sentence, `extractive` |
| `intent_id`, `intent_version`, `frame_id` | the need (and version) it answers, its topic frame |
| `evidence_ids`, `source` | supporting evidence and the exact span |
| `status`, `status_reason` | lifecycle (doc 07) and the rule that set it |
| `confidence`, `confidence_signals` | the extraction relevance (computed, not authored) |
| `introduced_in`, `modified_in` | answer versions where the claim appeared / changed |
| `introduced_by_constraint` | reserved for constraint-introduced claims |
| `created_at_ms`, `updated_at_ms` | session time |

## Claim-evidence graph

`ClaimEvidenceLink(claim_id, evidence_id, relation, basis)`. Relations are set only when justified:

| Relation | Set when | Basis |
|---|---|---|
| `SUPPORTS` | the claim is verbatim in usable evidence and addresses all active constraints of its need | `verbatim_sentence` |
| `PARTIALLY_SUPPORTS` | verbatim in usable evidence, but lacks the terms of an active constraint | `does_not_address_constraint:K<n>` |
| `CONTRADICTS` | two selected claims of one need, from different documents, state different numbers for the same unit around a shared content word | `potential_numeric_conflict:<unit>` |
| `UNSUPPORTED` | the linked evidence is no longer usable for the need | `evidence_<status>` |

**Contradictions are flagged, not resolved.** Both claims keep their status, and the answer section records uncertainty `conflict`.

Non-numeric contradictions ("permitted" vs "not permitted") are **not** detected. That is a known limitation (report §23).

Example (`tests/fixtures/corpus_conflict`, TEST FIXTURE ONLY):
- "opens its dome only when the wind speed is below 25 kilometres per hour" (older notes);
- "… below 30 kilometres per hour" (newer notes);
- → CONTRADICTS both ways, and the section shows uncertainty `I1:conflict` (dev S27 / S28).
