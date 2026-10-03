# 09: Answer Confidence and Source Priority

## No probability is presented as truth (brief §29)

- **No confidence score in the answer.** The answer carries no single number. Model token probabilities are never read, and the NLI probabilities are not calibrated (they are stored as raw signals only).
- **What the system exposes instead** is categorical and has a stated basis:

| Signal | Basis |
|---|---|
| claim `status` | verification: SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED |
| alignment `strength` | STRONG (one sentence entails it) / MODERATE (window or chunk) / WEAK (similar, not support) / NONE / CONTRADICTORY |
| evidence agreement | conflict groups: more than one source, disagreeing |
| retrieval coverage | intent coverage: facts / explicit uncertainty / failure; `value_not_stated`, `constraint_not_covered` gaps |
| answer `status`, `partial` | DRAFT (unvalidated stream) / VALIDATED_FINAL / BLOCKED; partial = some need answered only with uncertainty |
| grounding metrics | verifier-judged counts (doc 06), with their measured agreement with labels (report §6) |

## Source priority (brief §27-28)

- **No priority policy is applied.** The fixture corpora carry no authority, recency or version metadata. The "2023 / 2025 edition" of the permit notices is part of their *text*, not structured metadata.
- **Newer is not assumed correct.** Conflicts are therefore presented, never resolved: "The sources differ: “… 40 euros” [notice 2023] / “… 55 euros” [notice 2025]".
- **Future hook.** When a corpus provides explicit metadata (`Evidence.metadata`: version, effective date, authority), a priority rule may be added. It must be documented per dimension:
  - authority;
  - recency, only where the documents declare supersession;
  - specificity;
  - document version.

  The conflict must still be shown.
