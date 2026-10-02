# ADR-008: Grounding and Citation

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §16

## Context

- Every factual assertion must cite `[Doc_ID §Section]`. Insufficient evidence must produce explicit uncertainty or a clarification [G§3 p3].
- G4: ≥85% of sampled assertions supported and zero fabricated IDs [G§5 p5].
- Fabricated citations are an explicit pitfall [G§6 p5 #3].
- Streaming output must not show unvalidated facts as final.

## Decision

1. **Label-constrained citations.** The LLM sees evidence as `[E1]…[En]` and cites only labels. The Citation Manager maps labels to evidence to citation keys and checks each key against the corpus manifest. Fabricated document IDs are impossible by construction.
2. **One claim per sentence.** `[U]` tags mark uncertainty sentences. Short connectives are exempt from citation.
3. **Verification ladder:**

   | Level | Check |
   |---|---|
   | L0 | Label validity |
   | L1 | Numbers, dates and amounts ⊆ cited text |
   | L2 | Lexical recall / embedding support |
   | L3 | Optional NLI (Exp 9) |
   | Repair | Swap to a better-supporting selected label |

   Default policy `drop_and_flag` (no extra LLM call).
4. **Deterministic uncertainty.** Uncovered intents are flagged *before* synthesis from coverage scores. Uncertainty does not depend on the LLM admitting ignorance.
5. **Sentence-gated streaming.** Each sentence is released only after L0–L2 pass. The reported TTFT is the first validated sentence; raw first-token latency is also logged.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Trust the prompt only | No guarantee |
| Post-hoc citation attachment (search for a supporting chunk after generation) | Invites rationalized citations |
| Provider-native citations | Incompatible with structured output; locks into one provider |
| Always-on NLI | CPU cost; kept as an ablation |

## Why selected

It gives hard guarantees where possible (IDs) and cheap, measurable checks where not (support). It is also backend-agnostic, so it works identically for hosted, local and extractive modes.

## Trade-offs

- L2 thresholds need human-labeled calibration.
- `drop_and_flag` may remove true but paraphrased claims, trading completeness for groundedness.

## Consequences

- A human-annotated claim sample (≥100) is part of the evaluation.
- TTFT is slightly later than a raw token stream; both are reported.
