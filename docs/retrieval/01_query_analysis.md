# 01 · Query analysis (Phase 9)

Code: `src/streamrag/adaptive/analyzer.py` (`QueryComplexityAnalyzer`), contracts `QueryAnalysis`, `TemporalSpec`.

## Inputs
The need's query text (the Phase 5/6 contextual query: anaphora resolved, inherited context, active constraints), the
interpreted `Intent` (entities, inherited context, unresolved references, type), its active `Constraint`s, the number
of needs in the utterance, and **corpus statistics only** (BM25 vocabulary, IDF, term co-occurrence, the metadata
catalog). Retrieved text is never an input to the analysis.

## Signals (all in `QueryAnalysis.signals`, explainable)
| Signal | How it is measured |
|---|---|
| content terms | analyzed terms minus question words / fillers (`QUESTION_WORDS`) and words of an asked-value phrase ("how **high**") |
| exact identifiers | `AB-123`, `XYZ`, `§3` patterns whose analyzed tokens are in the BM25 vocabulary |
| rare terms | content terms with IDF ≥ `rare_idf` |
| vocabulary mismatch | share of content terms unknown to BM25 (digits and temporal cue words excluded) |
| constraints | active Phase 5/6 constraints; metadata values the user *stated as a constraint* (see 03 §security) |
| temporal | ISO date, "Month YYYY", "YYYY" → `as_of` with a period; "current/latest/now…" → reference date; "previous/former…" → `past` |
| value kind | `configs/claim_lexicon.yaml` value questions (amount, duration, time, count, …) |
| multi-hop | **entity_aspect_gap**: an entity-like rare term (capitalised mid-sentence, or an identifier) that shares no chunk with the rest of the question |
| context dependence / ambiguity / comparison | inherited context; ≤ 1 content term or unresolved references; comparison cues |

## Classes (first rule that matches; the rule names are stored in `reasons`)
* **MULTI_HOP** – `entity_aspect_gap:<term>`.
* **COMPLEX** – several needs, a comparison, ≥ 2 constraints / filters, or an explicit date together with a constraint.
* **MODERATE** – one constraint / filter, a temporal cue, context dependence, vocabulary mismatch, ambiguity or > 8
  content terms.
* **SIMPLE** – otherwise (`single_need_no_constraints`).

## Limitations
Capitalisation is the entity cue for MULTI_HOP: lower-case ASR transcripts lose it (the need then routes by its other
signals; ITERATIVE needs can still discover a bridge after the first search). Snowball stems and English cue lists
only. Thresholds are team values, checked on fixture data only.
