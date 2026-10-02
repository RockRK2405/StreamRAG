# 05: Session Refinement

This is a design artifact (Phase 2). Normative: spec §13, §14, §15, §17.

## Late detail: v1 → v2 without restart (guide Example 2, abstracted)

```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant IA as Intent Analyzer
  participant RC as Controller
  participant RF as Refiner
  participant LG as Ledger
  participant R as Retrieval
  participant FU as Fusion
  participant SY as Synthesizer
  participant SS as Session State

  Note over SS: v1 committed: claims C1..C3, citations, frame{topic, slots}
  U->>IA: chunk "the trip was international…"
  IA->>RC: act=CONSTRAINT, topic continuity high → turn_type=refinement
  RC->>RF: RETRIEVE (trigger=refinement)
  RF->>RF: Δ = {trip_type: set international}
  RF->>LG: delta query (topic ⊕ constraint), scope=session_docs
  LG->>R: miss → dispatch (v1 queries are never re-issued)
  U->>IA: chunk "…and booked after travel."
  IA->>RF: Δ += {booking_timing: set after_travel}
  RF->>LG: second delta query
  LG->>R: dispatch
  R-->>RF: delta evidence (widen to corpus only if not covered)
  U->>RC: UTTERANCE_END
  RF->>RF: affected claims A (scope ∩ intents, old-value mentions), unaffected U
  RF->>FU: carried evidence of U∪A + delta evidence
  FU->>SY: EvidenceSet
  SY-->>U: emit U verbatim first (already validated), then LLM ops for A + added
  SY->>SS: claims_v2 = U ∪ ops(A) ∪ added
  SS-->>U: ANSWER_COMMITTED v2 (parent=v1, diff, delta_queries, full_rerun=false)
```

## Refinement algorithm (flow)

```mermaid
flowchart TD
  T["New turn"] --> C{"Turn type"}
  C -->|refinement / mixed| D["1. Extract Δ: set / update / retract (+ new intents)"]
  C -->|presentation| P["Transform prior claims only<br/>no retrieval, citations ⊆ prior"]
  C -->|query, new topic| NQ["Normal query path, new frame"]
  D --> AF["2. Affected claims<br/>scope(Δ) ∩ intent, old-value mention, dependency"]
  AF --> DQ["3. Delta queries (topic ⊕ constraint)<br/>ledger dedup"]
  DQ --> RV["4. Retrieve: session_docs first,<br/>corpus if not covered"]
  RV --> MG["5. Merge into evidence store, fuse with carried evidence"]
  MG --> UP["6. LLM ops on affected claims: keep / modify / retract / add"]
  UP --> GV["Grounding validation (failed modify → keep old + uncertainty)"]
  GV --> KP["7. Unaffected claims copied verbatim with citations"]
  KP --> V["8. AnswerVersion v(n+1): parent, diff, delta_queries, full_rerun=false"]
  P --> V2["AnswerVersion kind=presentation (same claim IDs)"]
```

## Version lineage example (abstract)

```mermaid
flowchart LR
  v1["v1 initial<br/>C1 C2 C3"] --> v2["v2 refinement<br/>kept C1 C2 · modified C3 · added C4 C5"]
  v2 --> v3["v3 presentation<br/>same claims, two bullets"]
  v3 --> v4["v4 refinement<br/>Δ update 30→45 · modified C5"]
  v4 -.->|new topic| w1["v5 initial (topic T2)<br/>parent = null"]
  w1 -.->|back to T1| v6["v6 refinement<br/>parent = v4"]
```
