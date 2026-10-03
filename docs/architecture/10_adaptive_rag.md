# 10: Adaptive Streaming RAG (as implemented in Phase 6)

The diagrams match `src/streamrag/{session,context,delta,claims,answers}` and their integration into the Phase 5 `MultiIntentCoordinator` (`session.enabled: true`). Phases 3–5 are reused unchanged. Phase 6 ends at a versioned, claim-level **answer state**; rendering text is Phase 7.

## Primary flow

```mermaid
flowchart TD
  TS["Transcript stream<br/>(Phase 4 chunk manager)"] --> GATE{"Phase 4 controller gate<br/>+ late-detail gate"}
  GATE -- "closed (backchannel, not worthy)" --> NC["NO_CHANGE<br/>nothing re-run"]
  GATE -- open --> CA["Context analyzer<br/>IntentTracker.update:<br/>decomposition + cross-turn ops<br/>(set / update / retract / elliptical / topic return)"]
  CA --> SV["Session State Vn<br/>(SessionMemory, 4 layers)"]
  SV --> FR["FrameManager<br/>same / new / reactivated frame"]
  FR --> CD["ContextChangeDetector<br/>semantic diff -> ContextChange[]"]
  CD --> DP["DeltaPlanner<br/>impact analysis"]
  DP --> AI["Affected needs"]
  DP --> EV["EvidenceValidityManager<br/>retain / revalidate / supersede"]
  DP --> AC["Affected claims<br/>-> PENDING_VALIDATION"]
  AI --> DQ["Delta queries<br/>(previous query + delta)"]
  DQ --> ACT{"action"}
  ACT -- reuse_active --> RU["active query reused"]
  ACT -- cache_hit --> CH["SemanticCache hit<br/>'reused' ledger record"]
  ACT -- retrieve --> TR["Targeted retrieval<br/>Phase 3 service via Phase 4 executor"]
  TR --> NE["New evidence"]
  CH --> NE
  RU --> NE
  NE --> VAL["Evidence validity<br/>confirm / stale"]
  VAL --> CR["Targeted claim revalidation<br/>(extract, select, validate)"]
  EV --> CR
  AC --> CR
  CR --> AV["AnswerStateManager<br/>Answer Vn+1 (diff, sections to re-render)"]
  AV --> SV2["Session State Vn+1"]
  LG[("QueryLedger<br/>parent / supersedes / derived_from_change")] -.-> DQ
  ES[("EvidenceStore<br/>records + per-need assignments")] -.-> VAL
  CG[("ClaimGraph<br/>claims, links, transitions")] -.-> CR
```

## Memory layers

```mermaid
flowchart LR
  subgraph A["A Transcript memory"]
    T1["window of N utterances<br/>(redacted)"]
    T2["older: SHA-1 + length"]
  end
  subgraph B["B Semantic memory"]
    F["Topic frames"]
    N["Needs + versions"]
    K["Constraints<br/>active / retracted"]
    EN["Entities"]
  end
  subgraph C["C Retrieval memory"]
    QL["Query ledger + lineage"]
    SC["Semantic cache"]
    EST["Evidence store<br/>+ lifecycle"]
  end
  subgraph D["D Answer memory"]
    CL["Claims + links"]
    AVS["Answer versions + diffs"]
  end
  A --> B
  B --> C
  C --> D
  SM["SessionMemory façade<br/>versions / snapshot / restore / reset / archive"] -.-> A
  SM -.-> B
  SM -.-> C
  SM -.-> D
```

## Evidence lifecycle

```mermaid
stateDiagram-v2
  [*] --> ACTIVE: retrieved for the need
  ACTIVE --> RETAINED: change keeps it applicable
  ACTIVE --> REVALIDATION_REQUIRED: change puts it in question
  RETAINED --> REVALIDATION_REQUIRED: later change
  REVALIDATION_REQUIRED --> ACTIVE: confirmed by delta retrieval
  REVALIDATION_REQUIRED --> STALE: not confirmed
  ACTIVE --> SUPERSEDED: need corrected / entity replaced
  ACTIVE --> STALE: need removed
  STALE --> ACTIVE: retrieved again
  RETAINED --> ACTIVE: retrieved again
  ACTIVE --> INVALID: source chunk unavailable
  RETAINED --> INVALID: source chunk unavailable
```

## Claim lifecycle

```mermaid
stateDiagram-v2
  [*] --> PENDING_VALIDATION: registered (verbatim sentence)
  PENDING_VALIDATION --> SUPPORTED: usable evidence, covers constraints
  PENDING_VALIDATION --> PARTIALLY_SUPPORTED: usable, lacks a constraint
  SUPPORTED --> PENDING_VALIDATION: affected by a change
  PARTIALLY_SUPPORTED --> PENDING_VALIDATION: affected by a change
  PENDING_VALIDATION --> UNSUPPORTED: evidence no longer usable
  SUPPORTED --> SUPERSEDED: need corrected
  SUPPORTED --> STALE: need removed / not selected
  PARTIALLY_SUPPORTED --> STALE: not selected for the new version
```

## Streaming sequence (dev S01, from the generated trace)

```mermaid
sequenceDiagram
  participant U as User stream
  participant G as Gate
  participant E as Session engine
  participant R as Retrieval
  participant A as Answer state
  U->>G: "What are the rules | for ladders | in the orchard?"
  G->>E: open (provisional ticks)
  E->>R: Q1..Q3 (each a delta of the previous)
  R-->>E: evidence (confirm / stale), claims C1, C2
  E->>A: A1 initial at utterance end
  U->>G: "Specifically | overnight."
  G->>E: late detail (gate opens on content)
  E->>E: CONSTRAINT_ADDITION I1 +K1, evidence RETAINED, C1/C2 pending
  E->>R: Q4 = Q3 + "overnight" (parent Q3)
  R-->>E: C1 PARTIALLY_SUPPORTED, C2 SUPPORTED
  E->>A: A2 refinement: 1 kept, 1 modified, re-render S-I1
```

## Integration points (unchanged components)

| Component | Phase 6 use |
|---|---|
| Phase 4 controller | still the gate. The late-detail gate opens it only for active-need refinements with content (`context/cues.py`) |
| Phase 4 executor / schedulers | same jobs. Virtual traces replay exactly, including all session events |
| Phase 5 decomposer / tracker | extended with cross-turn ops, the constraint registry, elliptical classification by corpus co-occurrence, topic return |
| Phase 5 per-intent guards | budgets now per (need, utterance). A guarded action is **deferred** and re-dispatched while its need version is current |
| Phase 5 fusion | runs on the turn's needs: own needs plus earlier needs the turn changed |
| Phase 3 retrieval | unchanged; delta queries are ordinary requests |

## Baseline for comparison

`FullRestartPipeline` (brief §41). On every turn it builds a fresh session over the whole conversation:
- full intent analysis of every turn;
- full query generation and retrieval of every active need of the current frame;
- no cache and no evidence retention;
- every claim re-extracted and re-validated;
- an `initial` answer every time.

It uses the same gate. Final queries were checked equal to the incremental pipeline's on a 6-turn session (test).
