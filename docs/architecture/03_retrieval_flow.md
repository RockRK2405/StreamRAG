# 03: Retrieval Flow

This is a design artifact (Phase 2). Normative: spec §10, §11, §12.

## Index time

```mermaid
flowchart TD
  A[("corpus/ (txt, md, pdf, docx, html, json)")] --> B["Loader<br/>doc_id: native or sorted-path ordinal"]
  B --> C["Section parser<br/>section_id: native §, numbered heading, ordinal, or page"]
  C --> D["Chunker<br/>section-bounded, ~200–400 tok, 15% overlap"]
  D --> E1["Embedder (batch)<br/>title > heading: text"]
  D --> E2["BM25 builder<br/>scipy CSC, k1=1.5, b=0.75"]
  D --> E3["Vocabulary + IDF table<br/>(controller anchors)"]
  D --> E4["Near-dup groups<br/>hash + cos ≥ 0.97"]
  E1 & E2 & E3 & E4 --> F[("CorpusIndex (read-only)<br/>+ CorpusManifest{corpus_hash, …}")]
```

## Query time (one intent)

```mermaid
flowchart TD
  I["Intent (text, constraints, shared slots)"] --> N["1. Normalize<br/>lexical_terms / dense_text / filters"]
  N --> L{"Ledger hit?"}
  L -- yes --> SK["RETRIEVAL_SKIPPED (reuse)"]
  L -- no --> EM["2. Embed dense_text"]
  N --> MF["5. Metadata pre-filter mask"]
  EM --> DS["3. Dense exact search, top 50"]
  N --> LX["4. BM25, top 50"]
  MF --> DS
  MF --> LX
  DS --> UN["6. Candidate union by chunk_id"]
  LX --> UN
  UN --> DD["7. Near-dup collapse (precomputed groups)"]
  DD --> SN["8. Score normalization (logging + weighted ablation only)"]
  SN --> RRF["9. RRF k=60 (also fuses multiple retrievals of the same intent)"]
  RRF --> RK{"10. Cross-encoder enabled<br/>and segment CLOSED?"}
  RK -- yes --> CE["MiniLM-L6 CE on top 20<br/>(105–124 ms measured)"]
  RK -- no --> PT["RRF + dedup passthrough"]
  CE --> CV["Coverage: best score vs θ_cov"]
  PT --> CV
  CV --> TK["11. Per-intent ranked list → Fusion"]
```

## Across intents (fusion)

```mermaid
flowchart LR
  L1["I1 ranked"] --> RR
  L2["I2 ranked"] --> RR
  L3["I3 ranked"] --> RR
  K["carried evidence<br/>(refinement)"] --> RR
  RR["Quota round-robin<br/>q=3 per intent, section cap 2,<br/>cross-intent dedup, B_tok=2000"] --> CF["Conflict check<br/>typed values across docs"]
  CF --> LB["Label E1..En<br/>by intent priority, then rank"]
  LB --> ES[("EvidenceSet")]
```

**Why two fusion methods.** RRF combines rankings of the *same* question. Quotas share the budget across *different* questions. See ADR-009.
