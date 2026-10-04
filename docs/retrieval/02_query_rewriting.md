# 02 · Query rewriting (Phase 9)

Code: `src/streamrag/adaptive/rewrite.py` (`QueryRewriteEngine`), contract `RewrittenQuery`.

| Form | Content |
|---|---|
| `original` | what was said |
| `normalized` | NFKC, whitespace collapsed, trailing punctuation removed |
| `contextual` | the Phase 5/6 query of the need (resolved anaphora, inherited context, active constraints - e.g. "What about international applicants?" after an eligibility question becomes the eligibility query + the constraint) |
| `expanded` | contextual + bounded expansions |
| `lexical_query` / `dense_query` | expanded (BM25) / contextual (embedding: expansions add noise) |

**Expansion** (≤ `max_expansions`, additive, never replacing a word):
1. acronym → long form and long form → acronym, mined from corpus definitions "Long Name (LN)" and accepted only
   when the acronym equals the long form's initials (`MetadataCatalog.build`);
2. the configured general-purpose synonym groups (`configs/retrieval_lexicon.yaml`), words unknown to the index first;
   expansions whose words are not in the index are skipped (they cannot match).

Expansions also become *accepted equivalents* of the requirement terms (05).

**Invariants (tested):** every content term of `contextual` is in both retrieval queries (`keeps_need`, checked for
every eval question); entity words removed by a Phase 6 correction (`dropped_entities`) are never re-introduced; the
intent id is carried unchanged. Corrections themselves come from Phase 6: the Phase 6 delta query of the corrected need
is the contextual form, and the adaptive cache entries of the replaced entity are invalidated (08).
