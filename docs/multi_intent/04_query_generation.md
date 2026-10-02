# 04: Query Generation per Intent

**Code:** `intents/query_builder.py` (`IntentQueryBuilder`); contract `IntentQuery`

## Construction

```
query = resolved intent words           (anaphora replaced by the antecedent's words; corrections substituted)
      + inherited context not yet present  (distributed "for X", follow-up topic)
      + constraint text in scope not yet present (local + global)
```

- **"Not yet present"** is decided on analyzed terms (stemmed, stopwords removed), so a constraint is not appended twice.
- **One string per intent.** It feeds both BM25 and dense retrieval through the unchanged Phase 3 `RetrievalService`. BM25 drops stopwords; the dense embedding keeps the natural word order. Spec §9.4 allowed separate lexical and dense forms; one form keeps retrieval identical to Phase 3.

## What is preserved

| Element | How |
|---|---|
| Entities and terminology | Verbatim tokens; nothing inside a component is rewritten |
| Numbers | Verbatim ("after 5 pm" stays) |
| Negation | Never removed ("are ladders **not** allowed …"; tested) |
| Constraints | Appended in scope |
| Context needed for interpretation | Inherited with its source span (pronoun antecedent, follow-up topic, distributed PP) |

## No injected facts

- Every `QueryComponent` carries a `SourceSpan` into the transcript of this or an earlier utterance.
- A test asserts that every query word occurs in the session's transcripts. Another asserts that every component span equals its transcript substring.

Example:

| Input | Query |
|---|---|
| "For visitors, what are the rules and the schedule for the telescope?" | I1: `what are the rules for the telescope For visitors`<br/>I2: `the schedule for the telescope For visitors` |

| Component of I1 | Source |
|---|---|
| "what are the rules" | intent |
| "for the telescope" | inherited from I2's clause (distributed PP) |
| "For visitors" | global constraint K1 |

## Deduplication

`validate_queries` flags identical queries across intents; duplicate intents are merged before queries exist.

Across utterances, the streaming coordinator reuses the evidence of an earlier query instead of searching again (`RETRIEVAL_SKIPPED reason=ledger_hit`). This applies when a new intent's query terms have Jaccard ≥ `duplicate_jaccard` (0.8) with an earlier completed query (REQ-MI-005).
