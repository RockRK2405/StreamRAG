# 02: Streaming Flow

This is a design artifact (Phase 2). Normative: spec §7, §8, §19, §20.

## Guide Example 1 as a sequence (paraphrased chunks; stream time in seconds)

```mermaid
sequenceDiagram
  autonumber
  participant U as User stream
  participant CM as Chunk Manager
  participant RC as Controller
  participant QD as Decomposer
  participant LG as Ledger
  participant R as Retrieval (BM25 ∥ dense → RRF)
  participant RK as Reranker
  participant FU as Fusion
  participant SY as Synthesizer + Grounding
  participant T as Telemetry

  U->>CM: t=0.0 chunk ("…customer workshop in")
  CM->>RC: segment s1 OPEN, dangling
  RC-->>T: CONTROLLER_DECISION WAIT (trailing_function_word)

  U->>CM: t=0.8 chunk ("…Pune for 30 people, and I need")
  CM->>RC: s1 CLOSED (", and"), s2 OPEN dangling
  RC->>QD: RETRIEVE s1 (provisional)
  QD->>LG: I1 query forms
  LG->>R: miss → dispatch I1
  R-->>T: RETRIEVAL_STARTED t=0.8 trigger=provisional
  R->>RK: I1 candidates (≈10 ms)
  RK-->>T: EVIDENCE_RERANKED I1 (rerank-on-stability, ~105–124 ms measured)

  U->>CM: t=1.6 chunk ("…cancellation policy and the catering options.")
  CM->>RC: s2 CLOSED
  RC->>QD: RETRIEVE s2 (multi_intent)
  QD->>LG: I2, I3 (+ shared slots), I1 unchanged
  LG-->>T: (I1 already served: no new event)
  par I2 and I3 in parallel
    LG->>R: dispatch I2
    LG->>R: dispatch I3
  end
  R-->>T: RETRIEVAL_STARTED ×2 t=1.6
  R->>RK: I2, I3 candidates
  RK-->>T: EVIDENCE_RERANKED I2, I3

  U->>CM: t=2.1 UTTERANCE_END
  CM->>RC: final pass (nothing new)
  RC->>FU: finalize: intents COMMITTED
  FU-->>T: EVIDENCE_FUSED (quota per intent, coverage flags)
  FU->>SY: labeled evidence E1..En
  SY-->>U: ANSWER_DELTA (validated sentences) - TTFT measured from 2.1
  SY-->>T: GROUNDING_CHECKED, ANSWER_COMMITTED v1, TURN_COMPLETED
```

## What runs where in time

| Stream time | Activity | Basis |
|---|---|---|
| 0.0–1.6 s | User speaking | Input |
| 0.80–0.81 s | I1 retrieval (BM25 ∥ dense → RRF) | [M] components |
| 0.81–0.93 s | I1 rerank on stability (20 pairs) | [M] 105–124 ms |
| 1.60–1.61 s | I2, I3 retrieval in parallel | [M] components |
| 1.61–1.86 s | I2 + I3 rerank (serial) | [M] 2 × 105–124 ms |
| 1.6–2.1 s | Endpoint silence (slack ≈ +0.24 s on the dev machine) | Input |
| 2.10–2.115 s | Fusion + prompt assembly | [E] |
| 2.115 s → | LLM first validated sentence | Backend-dependent, to be measured |

Rows marked [M] use measured component latencies; their composition into this timeline is an estimate. Spill risk on slower hardware is covered in spec §20.4.

## Rules that make this work

1. Retrieval is triggered by **segment events** (closure or stability), never by chunk arrival.
2. Provisional work is **reused** through the ledger. I1 is not searched again at 1.6 s.
3. Reranking runs **when a segment closes**, so it fits in the speech gap.
4. Synthesis starts **only** at `UTTERANCE_END`. Tokens are released one validated sentence at a time.
