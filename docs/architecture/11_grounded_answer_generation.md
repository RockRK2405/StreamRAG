# 11: Grounded Answer Generation (as implemented in Phase 7)

The diagrams match `src/streamrag/{claims,generation,citations,validation,answer_state}` and their integration into
the Phase 6 session (sync pipelines and the streaming coordinator, `generation.enabled: true`). The generator is not
the source of truth: the evidence store is.

## Required architecture (brief §58)

```mermaid
flowchart TD
  ES["EvidenceSet<br/>usable evidence per need (Phase 6 store)"] --> CP["Claim Planner<br/>evidence-derived facts F1..Fn,<br/>importance, gaps, conflicts"]
  P6["Phase 6 answer state<br/>(what changed: sections to regenerate)"] --> CP
  CP --> CS["Claim Set (ClaimPlan)"]
  CS --> AP["Answer Planner<br/>sections per need, detail, reuse"]
  AP --> GG["Grounded Generator<br/>local LLM, JSON schema<br/>(extractive fallback)"]
  GG --> CA["Candidate Answer<br/>sentences + facts + labels"]
  CA --> CE["Claim Extraction<br/>1 sentence = 1 candidate claim"]
  CE --> CV["Claim Verification<br/>entailment (NLI) + numbers + rules;<br/>decompose if not supported"]
  CV -- SUPPORTED --> CM["Citation Mapping<br/>from the verification,<br/>sentence span in the chunk"]
  CV -- "UNSUPPORTED / PARTIAL / CONTRADICTED" --> RR["Repair / Retrieval<br/>keep atoms, restore facts,<br/>fallback retrieval, conflict, remove"]
  RR --> CV
  RR --> AV
  CM --> CVAL["Citation Validator<br/>index, text, span, pages, support"]
  CVAL --> AV["Answer Validator<br/>consistency, coverage, strict/relaxed"]
  AV --> GA["Grounded Answer<br/>DRAFT / VALIDATED_FINAL / BLOCKED"]
  GA --> ST["Streamed to user<br/>ANSWER_* events"]
```

## Per answer version (engine stages and events)

```mermaid
sequenceDiagram
  participant S as Session (Phase 6)
  participant E as GroundedAnswerEngine
  participant L as LLM (Ollama, local)
  participant V as Verifier (NLI + rules)
  participant C as Citations
  S->>E: AnswerVersion (sections, needs_regeneration)
  E->>E: CLAIM_PLAN_CREATED (facts, gaps), reuse unchanged sections / kept sentences
  E->>L: facts of changed sections only (ANSWER_GENERATION_STARTED, LLM_CALL)
  L-->>E: JSON sentences {text, facts, evidence}
  E->>V: CLAIMS_EXTRACTED -> CLAIM_VERIFICATION_STARTED
  V-->>E: CLAIM_VERIFIED / CLAIM_REJECTED (status, support, contradiction)
  E->>E: policy: keep / conflict / keep_atoms / restore_facts / retrieve / remove (CLAIM_REPAIRED, VALIDATION_RETRIEVAL)
  opt strict and unsupported content, or critical fact missing
    E->>L: revision with REJECTED + ALREADY WRITTEN (bounded)
  end
  E->>C: map from verification (CITATION_CREATED) -> validate (CITATION_VALIDATED)
  E->>E: consistency, coverage, status (ANSWER_VALIDATED, ANSWER_REVISED)
  E-->>S: ANSWER_STARTED .. ANSWER_CLAIM_READY / ANSWER_CITATION_READY .. ANSWER_COMPLETED (+ ANSWER_FINALIZED)
```

## Claim lifecycle in the answer

```mermaid
stateDiagram-v2
  [*] --> DRAFT: planned / generated
  DRAFT --> SUPPORTED: entailed by evidence, numbers present
  DRAFT --> PARTIALLY_SUPPORTED: some atoms entailed
  DRAFT --> UNSUPPORTED: no entailing premise
  DRAFT --> CONTRADICTED: contradicted (same proposition)
  PARTIALLY_SUPPORTED --> SUPPORTED: keep_atoms (only verified atoms)
  UNSUPPORTED --> SUPPORTED: restore_facts / retrieval fallback found support
  UNSUPPORTED --> [*]: removed (kept in rejected)
  CONTRADICTED --> SUPPORTED: conflict side (supported by its own source, shown with the other side)
  SUPPORTED --> [*]: cited and released
```

## Streaming integration

```mermaid
flowchart LR
  CH["chunks"] --> GATE{"controller gate"}
  GATE --> INT["interpretation + delta plan<br/>(Phase 5/6)"]
  INT --> RET["provisional retrieval"]
  RET --> DR["DRAFT answer<br/>(extractive, verified, no LLM)"]
  END["utterance end"] --> FIN["commit Phase 6 answer"] --> LLM["grounded generation<br/>(only changed sections)"] --> VAL["verified + cited"] --> TC["TURN_COMPLETED.answer<br/>VALIDATED_FINAL"]
  VAL -.-> REP[("trace: LLM_CALL outputs<br/>-> exact replay")]
```

| Component | Phase 7 use |
|---|---|
| Phase 6 answer state | decides which sections changed; Phase 7 renders and validates only those |
| Phase 6 evidence store | the verification pool per need and the citation source; fallback retrieval adds to it |
| Phase 3 retrieval / index | fallback retrieval; the index is the authority for every citation location |
| Phase 4 replay | LLM outputs recorded in `LLM_CALL` and replayed by request hash |
