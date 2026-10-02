# Samsung PRISM Theme 4
# Streaming Live RAG
# Phase 1 — Research & Technical Investigation

| | |
|---|---|
| Date | 2026-10-02 |
| Status | Phase 1 complete. Nothing has been implemented. |
| Scope | Research, system understanding, requirements, design space, measurement methodology |
| Not in scope | Application code, UI, Docker build, benchmark results |

**Source legend.** Every requirement carries a source tag.

| Tag | Source |
|---|---|
| `[G§n pX]` | *Theme 4 Guide_RAG.pdf* ("Streaming Live RAG"), section *n*, page *X* |
| `[D sN]` | *Samsung PRISM_Y2026_GenAI_Hackathon_3rd_Edition.V2(2).pdf*, slide *N* |
| `[T-ppt]` | *CollegeName_TeamName_Submission.pptx* (submission template) |
| `[T-ai]` | *LangAI3.0_AI_Disclosure.docx* (AI usage disclosure form) |
| `[ENV]` | Measured or observed on the development machine during Phase 1 |
| `[INF]` | Our inference. This is not an official statement. |

**Errata (added in Phase 2, 2026-10-02).** `PHASE_2_SYSTEM_SPECIFICATION.md` §1.0 supersedes the following statements in this document. The official documents take priority.

| ID | Correction | ADR |
|---|---|---|
| **K1** | The baseline B0 is *static hybrid*, not dense-only (§14); dense-only is an ablation. | ADR-011 |
| **K2** | The container is the mandatory one-command path (§7.3). | ADR-007 |
| **K3** | The official re-rank minimum is RRF + dedup; the cross-encoder is an optional enhancement (§8.3). | ADR-003 |
| **K4 / K6** | Information requests are never suppressed. Corpus anchors only affect how *early* retrieval fires (§8.5, §17-H). | ADR-004 |
| **K5** | The G3 formula in §10 is corrected. | — |

**Honesty conventions.** Numbers are labeled **measured** (a script exists in `research/phase1/`), **arithmetic** (derived from published prices or measured numbers), or **estimate** (an engineering expectation that Phase 2 must measure). There are no accuracy or quality numbers in this document, because no corpus and no system exist yet.

---

## 1. Executive Summary

**What Theme 4 asks for.** Theme 4 asks for an *event-driven* RAG engine that consumes a conversation as timestamped transcript chunks rather than as finished queries. The engine has to do four things [G§1 p1]:

1. Decide during the utterance whether retrieval is needed, and start it early.
2. Split one natural utterance into several search-ready sub-queries.
3. Refine an existing answer when late details arrive, without restarting.
4. Ground every claim in corpus citations, and flag uncertainty when evidence is missing.

All of it must be observable through telemetry. Six quantitative gates (G1–G6) judge the result [G§5 p4–5]. Separately, the hackathon jury weights *working prototype* at 30% and *technical depth* at 25% [D s11].

**Repository reality.** The project directory `/Users/rudrakhale/Desktop/SamsungHackathon` was **completely empty**: no code, no git repo, no data. The only Theme 4 material on the machine is the 6-page guide PDF, which is image-only "Print to PDF" output, and the overall hackathon deck.

**Critical discovery #1: there is no Theme 4 corpus.** The deck says a corpus is provided [D s7], and the guide's citation examples (`Doc_09 §1`, `Doc_12 §2`, `Doc_31 §4`) imply one exists. But no corpus file is present anywhere I searched, and the guide PDF has no embedded attachments (checked programmatically). Corpus-dependent work cannot start until this is resolved: chunking choices, gold labels, retrieval-quality experiments, and realistic test answers.

**Critical discovery #2: the timeline may already have passed.** The deck lists final submission as 25 Sep 2026 and the top-15 announcement as 9 Oct 2026 [D s9, s12]. Today is 2 Oct 2026. The dates are marked tentative, but the real deadline must be confirmed before Phase 2 scope is fixed.

**Critical discovery #3: retrieval compute is not the bottleneck; LLM calls are.** On the development CPU (measured, [ENV]):

| Operation | Latency |
|---|---|
| Small-embedder query encoding | ~3–4 ms |
| BM25 query at 100k chunks | ~0.4 ms |
| Exact dense search at 100k chunks | ~1.6 ms |

A whole hybrid retrieval therefore costs single-digit milliseconds, which means:

- Early, speculative retrieval is nearly free.
- What costs time and money is every LLM call on the critical path.
- The design should keep streaming decisions (controller, provisional queries) **LLM-free**.
- LLM calls should be spent only where they add the most: synthesis, and at most one decomposition check per utterance.

**Critical discovery #4: the global Python environment is broken for this stack.** transformers 5.9.0 refuses the installed torch 2.2.2, so `sentence-transformers` fails to import [ENV]. Phase 2 needs an isolated environment with pinned versions. The deliverables require a lockfile anyway [G§8 p6].

**Recommended target (detail in §18): Architecture B, a "Streaming Hybrid RAG with a rule-first controller, a query ledger and versioned answer state".**

- **Process model.** One Python process running on an asyncio event loop.
- **Retrieval.** BM25 plus a small CPU embedder, fused with RRF and searched with exact numpy search. No vector database.
- **Reranking.** An optional small cross-encoder, applied only when a sub-query's intent is committed.
- **Decomposition.** Deterministic clause segmentation while the user is speaking, plus at most one structured LLM decomposition check per utterance.
- **Query ledger.** Every query issued in a session is stored, so provisional work is reused and late details trigger only delta queries.
- **Synthesis.** One LLM call per turn, constrained to cite only labeled evidence. Citations are validated afterwards, so a fabricated document ID is impossible by construction. Uncertainty is flagged per sub-intent.
- **Telemetry.** A JSONL event trace that is a strict superset of the guide's example output record.
- **LLM backend.** Pluggable: a hosted model, a local model, or an extractive no-LLM mode. The replay suite can finish on any machine, which protects G1.

---

## 2. Official Problem Interpretation

### 2.1 The problem in our own words

A voice user does not produce a search query. They speak, and while they speak the system receives a growing transcript in timestamped pieces. One spoken request often bundles several needs, for example venue capacity, cancellation terms and catering options [G§1 p1]. Users also add constraints after the fact, such as "actually, the trip was international" [G§1 p1].

A conventional RAG pipeline waits for the full turn, sends it as one query, and answers. That costs seconds of silence, mixes the separate needs into one muddy query, and handles a late constraint by throwing away state and starting over [G§1 p1].

Theme 4 asks for the opposite behavior on every one of those points [G§1 p1, D s7]:

- Retrieve while the user is still talking.
- Split the request into its real questions.
- Merge the evidence into one grounded answer.
- When new detail arrives, improve the answer you already have.

### 2.2 The unit of work

| Level | Definition | Evidence |
|---|---|---|
| **Session** | An ephemeral conversation. All memory is scoped to it and destroyed with it. | [G§3 p3] |
| **Turn / utterance** | One user utterance, delivered as a sequence of timestamped chunks and ended by an end-of-utterance marker. | [G§2 p2], [G§4 p3] |
| **Chunk** | `(timestamp_s, text)`. The guide's example shows chunks at 0.0 s, 0.8 s, 1.6 s and an end marker at 2.1 s. | [G§2 p2], [G§4 p3] |
| **Answer version** | Each answer that synthesis commits is versioned (v1, v2, …), with lineage. | [G§4 p4] |

### 2.3 Pipeline mandated by the guide [G§2 p2]

```
Transcript stream ──► [1] Retrieval Controller (intent stability; WAIT | RETRIEVE | NO-RETRIEVAL)
                          │ retrieve triggered
                          ▼
                      [2] Multi-Intent Decomposer (extract sub-queries; parallel routing)
                          ▼
                      [3] Corpus Retrieval & Fusion (supplied corpus; dense+sparse hybrid; rerank+dedup)
                          ▼
                      [4] Session-Aware Synthesis (incremental update; grounding/citation check; uncertainty)
                          ▼
Output: streamed answer + grounded citations + observability telemetry
```

A fifth cross-cutting component, **Observability Telemetry**, is listed in the components table [G§2 p2].

### 2.4 Details that are easy to miss in the guide

Each of these comes from reading the examples closely.

1. **Provisional retrieval is reused, not repeated.** In Example 1, the output record lists exactly three retrieval events [G§4 p4]:
   - one *provisional* query at 0.8 s;
   - two *multi_intent* queries at 1.6 s (cancellation, catering).

   The venue/capacity intent that was already searched provisionally is **not** searched again at 1.6 s. The system must keep track of what it has already retrieved. [INF]
2. **Sub-queries and retrieval queries are different objects.** The `sub_queries` list holds clean, context-free intent statements. The queries actually sent to retrieval are **enriched with shared context**: the location and event type are attached to the cancellation and catering queries. Decomposition must spread shared context into each sub-query, which matches the guide's warning about losing conversational context [G§2 p2]. [INF]
3. **The `trigger` field is an enumeration.** Observed values are `provisional` and `multi_intent` [G§4 p4]. Late-detail turns need a value of their own; we will use `refinement`.
4. **Uncertainty is a first-class output field.** It is a string naming what could not be verified [G§4 p4]. It must be filled whenever some sub-intent lacks evidence [G§3 p3].
5. **Citation format is `Doc_ID §Section`** [G§3 p3, G§4 p4]. Citations therefore need document-level IDs **and** section markers. Ingestion must preserve or derive both.
6. **Refinement keeps the old answer.** In Example 2, v2 keeps the "standard rule still applies" statement and v1's citations, adds the exception-specific statements and delta citations, and increments the version [G§4 p4]. The guide's term "citation graphs" [G§1 p1] implies a claim→citation structure that can be updated selectively.
7. **Presentation-only turns must not query the corpus at all.** They reuse earlier citations and add none [G§4 p4]. The guide lists *reformat, shorten, translate* as examples [G§6 p5].
8. **The comparison baseline is part of the deliverable.** The guide requires a baseline comparison, at least three analyzed edge-case failures, and at least two ablations; its examples are hybrid vs dense-only and rule-based vs model-based controller [G§8 p6]. The experiment plan must produce exactly these.
9. **The guide penalizes heavy orchestration explicitly.** Multi-agent or complex pipelines are judged on cost-to-performance, and every component must justify its latency and compute [G§3 p3]. The deck makes the same point: simple and cheap beats heavy agent orchestration [D s7].

### 2.5 What Theme 4 is *not*

- It is **not** a chatbot with a vector store. Static request→response RAG misses G2 and G5 by construction.
- It is **not** a speech project. Voice input may be simulated from transcripts [D s7]. ASR is optional and not graded by any gate.
- It is **not** an agent framework demo. Parsimony is a hard rule [G§3 p3].

---

## 3. Repository Audit

Audit date 2026-10-02. Target: `/Users/rudrakhale/Desktop/SamsungHackathon`.

| Item | Finding | Evidence |
|---|---|---|
| Contents | **Empty directory** (0 bytes, no hidden files) | `ls -la`, `find`, `du -sh` → `0B` |
| Git | Not a git repository | Environment report and `ls -a` |
| Programming language / framework | None present | — |
| Project structure, scripts, configs | None | — |
| Dependencies / lockfiles | None | — |
| Tests / test infrastructure | None | — |
| Docker setup | None in the repo. Docker 29.6.1 is installed on the host. | `docker --version` |
| README / docs | None | — |
| Sample datasets / corpus | **None** (see §4) | — |
| Retrieval code, embeddings, indexes, vector DB | None | — |
| LLM integrations | None in the repo | — |
| Benchmarking / streaming utilities | None | — |
| **Baseline system** | **Does not exist.** Everything is greenfield. | — |

Phase 1 added only this dossier and `research/phase1/`, which holds two micro-benchmark scripts and their raw results. No application code was written.

**Related materials found outside the repo**, in `~/Downloads/Samsung PRISM Gen AI Hackathon 3.0/`:

| File | Relevance |
|---|---|
| `Theme 4 Guide_RAG.pdf` | **Primary specification.** 6 pages. Image-only "Microsoft: Print To PDF" render: 5 extractable characters, 2,142 image fragments, 0 embedded files, 0 links. |
| `Samsung PRISM_Y2026_GenAI_Hackathon_3rd_Edition.V2(2).pdf` | Hackathon deck: theme scope, judging weights, submission rules |
| `CollegeName_TeamName_Submission.pptx` | 12-slide submission template. Required sections: theme, existing solutions & gaps, solution & architecture diagram, demo, tech stack, impact, innovation/results/limitations, what's next, differentiation, GitHub checklist. |
| `LangAI3.0_AI_Disclosure.docx` | **AI usage disclosure form.** Per-feature origin (self, AI, or both), tools, prompts, and modifications. This project is being built with AI assistance, so the team must keep records for it from Phase 2 onward. |
| `Theme 2/` (JSON + `schema.py`) | Theme 2's data. Not relevant to Theme 4, but it shows that themes *do* ship data files. Theme 4's data is missing, not nonexistent. |
| Other themes' guides | Checked Theme 5's guide (similar "full-duplex" wording in the deck): no corpus, replay or transcript format is described. |

**Development environment** [ENV]:

| Component | Observed | Implication |
|---|---|---|
| CPU / RAM | Apple M5 Pro, 15 cores, 24 GB, arm64 | Much faster than a typical judge machine. Do not extrapolate latency directly. |
| Python | 3.12.0 (python.org framework build) | Fine. Phase 2 should use an isolated venv. |
| Global packages | torch 2.2.2, transformers 5.9.0, sentence-transformers 5.5.1, chromadb 1.5.9, fastapi 0.135.3, onnxruntime 1.26.0, PyMuPDF 1.27.2, pdfplumber, pypdf, langchain 1.2.x, ollama 0.6.1, openai 2.41.1 | **transformers 5.9 disables torch < 2.4, so `import sentence_transformers` crashes.** The global env is not usable as-is; pin a fresh env. |
| Not installed | rank-bm25, bm25s, faiss, hnswlib, spaCy, nltk, rapidfuzz, ragas, flashrank, fastembed | Phase 2 must choose and pin. |
| HF model cache | `all-MiniLM-L6-v2`, `multilingual-e5-base` (safetensors only, no ONNX) | Two embedders can be benchmarked offline today. Others need a download, with permission. |
| Docker | 29.6.1 | Docker Desktop on macOS has no GPU passthrough. The container is CPU-only by design. |
| Ollama | Installed. Only model: `qwen3.6:latest`, a 36B MoE at Q4_K_M, **23 GB** | Too large for a reproducible container or a 24 GB judge box. A small model must be pulled for local-LLM mode. |
| API keys | No LLM provider key set in the environment | A hosted LLM path needs a key to be supplied (see §19). |
| Tooling | git, gh, docker present. uv, poetry, conda absent. | Phase 2 needs a lockfile strategy (pip-tools `requirements.lock` or install `uv`). |

---

## 4. Corpus/Data Audit

### 4.1 What was searched

- The project directory.
- `~/Downloads`, including the hackathon folder.
- `~/Desktop` at top level. Only folder names were inspected for unrelated personal projects.
- The guide PDF, for embedded files (PyMuPDF `embfile_count() == 0`), annotations and links (none). The PDF's text layer has 5 characters; the content is rendered images.
- The Theme 5 participant guide and PDF, for any shared corpus or replay format (none).

**Result: no Theme 4 corpus is available locally.**

### 4.2 What can legitimately be inferred about the official corpus

All rows below are from the guide's illustrative examples. They are not corpus facts and must not be used as such.

| Property | Inference | Basis | Confidence |
|---|---|---|---|
| Document IDs | Pattern `Doc_<NN>`, e.g. `Doc_09`, `Doc_12`, `Doc_31` | [G§4 p4] | Medium. The examples may be illustrative only. |
| Size | At least ~31 documents if the IDs are sequential | [INF] from `Doc_31` | Low |
| Section structure | Documents have numbered sections cited as `§N` | [G§3 p3, G§4 p4] | Medium-high. Required by the citation rule. |
| Domain | Enterprise operational policy: corporate event and workshop planning (venues, capacity, cancellation/refund, catering) and employee travel reimbursement (international travel, late-booking exceptions, approvals, receipts) | [G§1 p1, G§4 p3–4] | Medium. Domain of the examples only. |
| Language | English | [INF] from the examples | Medium |
| Format | Unknown (PDF, DOCX, Markdown, JSON?) | — | Unknown |
| Metadata | Unknown beyond doc ID + section | — | Unknown |
| Held-out set | The replay evaluation is **private and held out**. It may use the same corpus or another one. | [G§3 p3] | High that prompts are held out |

The guide's example `answer` string in the JSON record is **cut off in the PDF**. Its full text cannot be recovered.

### 4.3 Consequences for design

1. **Ingestion must not depend on any specific corpus.** It takes a folder of documents in common formats (txt, md, pdf, docx, html, json). It should *preserve native document and section IDs* where they exist and *derive* `Doc_ID` and `§N` from file names and headings where they don't. Anything domain-specific would break on a held-out corpus and could count as hardcoding [G§3 p3].
2. **Chunks follow section boundaries.** The citation unit is `Doc_ID §Section`, so chunks should never cross a section boundary. Long sections are split into `§N` parts that still cite as `§N`, keeping `§N.k` internally.
3. **Index build happens at container start or build time from the mounted corpus**, cached by corpus hash. This lets the judges swap corpora and satisfies "no precomputation" for answers. Precomputing the *index* is normal and is not answer precomputation [INF].
4. **Corpus size is unknown, so we designed for 10 to 100,000 chunks.** Measured search latency stays under 2 ms across that range (§8.1). Index build time is dominated by embedding throughput: about 180–210 short passages per second for MiniLM on this machine (measured). That is roughly 1 minute per 10k chunks here and slower on judge hardware (estimate).

### 4.4 Fallback if no official corpus is ever supplied

This is the team's decision (§19, Q1). The options:

- **(a)** Ask the organizers (prism@samsung.com [D s14]) for the Theme 4 corpus or a dev sample.
- **(b)** Build a dev corpus from **public-domain** policy documents in the same domain, for example published government travel-allowance or event-procurement rules. These are real documents, so their facts are genuine.
- **(c)** Write a small **clearly labeled synthetic** corpus that mirrors the guide's scenarios, for controller and refinement tests only.

Never present (b) or (c) as the official corpus. Report all metrics as "dev corpus".

---

## 5. Hard Constraints

### 5.1 Official constraints (I)

| ID | Constraint (paraphrased) | Source | Design consequence |
|---|---|---|---|
| C1 | **Corpus isolation.** Evidence comes only from the supplied corpus. No web scraping, no third-party knowledge bases, and no unindexed parametric model knowledge for factual claims. | [G§3 p3] | No web tools. Synthesis is constrained to labeled evidence. A post-hoc verifier catches claims that came from the model's own knowledge. |
| C2 | **No hardcoding or precomputation.** Benchmark replay is private. Prompts, queries and canned responses must not be embedded in code. | [G§3 p3] | No benchmark-specific strings, lexicons or answer templates. Prompt templates and classifier seed phrases live in config files, stay generic, and are disclosed. Test utterances live only in the eval harness. |
| C3 | **Rigorous factual grounding.** Every factual assertion is attributed to corpus chunk IDs or `[Doc_ID §Section]` markers. If evidence is missing for a sub-intent, emit an uncertainty indicator or ask a targeted clarification. | [G§3 p3] | Citation-label constraint, citation validation, per-intent coverage scoring, an uncertainty field. |
| C4 | **Session-bound state.** No cross-session profiling or persistence across test runs. Memory is ephemeral and session-scoped. | [G§3 p3], [D s7] | In-memory session store, dropped at session end. Telemetry logs are write-only and never read back as memory. |
| C5 | **Architectural parsimony.** Multi-agent or complex orchestration is judged on cost-to-performance. Every component must justify its latency and compute. | [G§3 p3], [D s7] | Single process. LLM-free critical path. Each component gets an ablation. |
| C6 | **Full duplex.** Retrieval begins before the utterance ends. | [D s7], [G§1 p1] | Streaming controller. G2 is measured on stream time. |
| C7 | **Reproducibility.** One command on a clean machine; the automated replay suite completes with no manual steps. | [G§5 p4 G1], [G§8 p6] | `docker compose up` or a single CLI command. The replay suite must finish **even without API keys** (extractive or local fallback). |
| C8 | **Submission packaging.** Public or shared GitHub repo with a README, Docker files and requirements. Release tag `PRISM_GENAI_HACKATHON_Y2026` on the final commit. Everything referenced must be in the tagged commit. | [D s13] | Plan the repo layout and tagging from day one. |
| C9 | **Deliverables.** Pinned lockfiles and env templates. Architecture brief of at most 6 pages. Benchmark report covering the baseline, ≥3 edge-case failures and ≥2 ablations. Demo video of at most 5 minutes. Telemetry schema. | [G§8 p6], [D s11–13] | The experiment plan in §15 is built to produce these. |
| C10 | **Presentation.** PPT/PDF named `CollegeName_TeamName` using the template. One submission per team. Not following the guidelines means disqualification. | [D s12, s13], [T-ppt] | Administrative. Track it in the Phase 5 checklist. |
| C11 | **AI disclosure.** The disclosure form asks for each feature's origin and the AI tools and prompts used. | [T-ai] | Keep an AI-usage log from Phase 2 onward. |
| C12 | **Voice may be simulated** from transcripts. | [D s7] | Transcript replay is the main input path. ASR is optional. |

### 5.2 Team-imposed constraint (not official)

**CPU-first.** Theme 4 does **not** state a CPU requirement. Theme 1 does [D s4]. CPU-first is still the safe reading [INF]:

- G1 requires a clean machine, and judges may have no GPU.
- Docker on macOS cannot reach the GPU.
- "Simple and cheap" is an explicit Theme 4 value [D s7].

We therefore treat "all non-LLM components run on CPU within latency budget" as a hard design constraint. A hosted LLM is allowed as an *option*, not a dependency.

### 5.3 Explicitly or effectively out of scope (J)

| Item | Status | Basis |
|---|---|---|
| Web search / external knowledge | **Forbidden** | [G§3 p3] |
| Cross-session user profiles or persistent memory | **Forbidden** | [G§3 p3], [D s7] |
| Precomputed or canned benchmark answers | **Forbidden** | [G§3 p3] |
| Heavy multi-agent orchestration | Discouraged; must be justified by cost-to-performance | [G§3 p3], [D s7] |
| Real speech recognition | Optional. Transcripts are acceptable. | [D s7] |
| TTS, wake word, UI polish | Not graded by any Theme 4 gate. (Theme 5 lists these as out of scope [D s8]; Theme 4 says nothing.) | [INF] |
| Model fine-tuning | Not required by any gate | [INF] |
| Multilingual support | Not mentioned. Examples are English. | [INF] |

---

## 6. Functional Requirements

Priority: **M** = must (gate-bearing or a hard rule), **S** = should (strongly implied), **C** = could (differentiator).

### 6.1 Core functional requirements (A)

| ID | Requirement | Pri | Source | Gate |
|---|---|---|---|---|
| FR-1 | Ingest a stream of timestamped transcript chunks per session and turn, plus an end-of-utterance signal | M | [G§2 p2], [G§4 p3] | G2 |
| FR-2 | On each chunk, decide one of WAIT, RETRIEVE or NO_RETRIEVE, with a logged reason | M | [G§2 p2] | G2, G6 |
| FR-3 | Detect turns that need no retrieval (presentation restructuring, conversational filler) and answer from session context | M | [G§4 p4 Ex3], [G§6 p5] | G2 (false triggers) |
| FR-4 | Split compound utterances into distinct, orthogonal, search-ready sub-queries | M | [G§1 p1], [G§2 p2] | G3 |
| FR-5 | Retrieve for sub-queries in parallel | M | [G§1 p1], [G§2 p2] | G2/latency |
| FR-6 | Fuse, deduplicate and rerank evidence across sub-queries | M | [G§2 p2], [D s7] | G4 |
| FR-7 | Produce **one** unified streamed answer that covers all sub-intents | M | [G§4 p3 Ex1], [D s7] | G3/G4 |
| FR-8 | Treat late details as refinements: delta retrieval only, answer version increment, earlier citations kept | M | [G§4 p4 Ex2], [D s7] | G5 |
| FR-9 | Emit `[Doc_ID §Section]` citations; never emit IDs that aren't in the corpus | M | [G§3 p3], [G§5 p5 G4] | G4 |
| FR-10 | Emit explicit uncertainty or a clarification request when a sub-intent lacks evidence | M | [G§1 p1], [G§3 p3] | G4 |
| FR-11 | Emit structured telemetry for every turn: timestamps, triggers, citations, version lineage, token cost | M | [G§2 p2], [G§5 p5 G6] | G6 |
| FR-12 | Produce the guide-compatible output record: `retrieval_events[]`, `sub_queries[]`, `answer`, `citations[]`, `uncertainty` | M | [G§4 p4] | G1–G6 |
| FR-13 | Ship a replay runner that feeds recorded chunk streams with real or virtual timing | M | [G§5 p4 G1], [G§7 p5] | G1 |
| FR-14 | Live demo path (CLI and/or minimal web view) showing streaming decisions and telemetry | S | [G§8 p6] (video) | — |
| FR-15 | Optional live microphone input via streaming ASR | C | [D s7] | — |

### 6.2 Retrieval requirements (C)

| ID | Requirement | Pri | Source |
|---|---|---|---|
| RR-1 | Search only the supplied, indexed corpus | M | [G§3 p3] |
| RR-2 | Hybrid dense plus sparse scoring | M | [G§2 p2], [G§7 p5] |
| RR-3 | Rerank and deduplicate chunks | M | [G§2 p2] |
| RR-4 | Reciprocal Rank Fusion (named in the roadmap) | S | [G§7 p5] |
| RR-5 | Chunk IDs that map to `Doc_ID §Section` | M | [G§3 p3] |
| RR-6 | Corpus-agnostic ingestion (held-out evaluation) | M | [G§3 p3] [INF] |
| RR-7 | Retrieval cache or ledger so the same query is never re-run within a session | S | [G§4 p4] [INF], [G§5 p5 G5] |

### 6.3 Streaming requirements (D)

| ID | Requirement | Pri | Source |
|---|---|---|---|
| SR-1 | Process chunks as they arrive, without waiting for the end of the utterance | M | [G§1 p1] |
| SR-2 | Provisional retrieval once the semantic content of a segment is stable | M | [G§4 p3 Ex1] |
| SR-3 | No retrieval on every token or noise (debounce and stability) | M | [G§6 p5 #1] |
| SR-4 | Start synthesis at the end of the utterance with retrieval already done | S | [G§4 p3 Ex1] |
| SR-5 | Measure time-to-first-token and stage latencies | M | [D s7], [G§7 p5] |
| SR-6 | Cancel stale in-flight work when it is superseded | S | [INF] |

### 6.4 Session-memory requirements (E)

| ID | Requirement | Pri | Source |
|---|---|---|---|
| SM-1 | Per-session state: turns, answer versions, evidence, query ledger, claims | M | [G§4 p4 Ex2] |
| SM-2 | Ephemeral; destroyed at session end; never persisted across runs | M | [G§3 p3] |
| SM-3 | Strict isolation between concurrent sessions | M | [G§3 p3] [INF] |
| SM-4 | Version lineage (v1→v2) with a diff of retained, modified and added claims and citations | M | [G§2 p2], [G§4 p4] |

### 6.5 Grounding and citation requirements (F)

| ID | Requirement | Pri | Source |
|---|---|---|---|
| GR-1 | Every factual sentence carries at least one valid citation | M | [G§3 p3] |
| GR-2 | Zero fabricated or nonexistent document IDs | M | [G§5 p5 G4] |
| GR-3 | Cited chunks actually support the claim; at least 85% of sampled claims | M | [G§5 p5 G4] |
| GR-4 | Presentation-only turns reuse prior citations and add none | M | [G§4 p4 Ex3] |
| GR-5 | Uncertainty names the specific unverified aspect | M | [G§4 p4] |

---

## 7. Non-Functional Requirements

### 7.1 Non-functional (B)

| ID | Requirement | Target (ours, provisional) | Source |
|---|---|---|---|
| NFR-1 | Controller decision latency per chunk | < 20 ms p95, so the retrieval timestamp ≈ the chunk timestamp, as in Ex1 | [G§4 p3] [INF] |
| NFR-2 | Retrieval latency per sub-query (hybrid, no rerank) | < 50 ms p95 on judge-class CPU | [INF] |
| NFR-3 | Time-to-first-token after end of utterance | ≤ 1.5 s with a hosted LLM; measure and report for local mode | [D s7] (metric), target ours |
| NFR-4 | LLM calls per retrieval turn | ≤ 2 (1 synthesis + ≤1 decomposition check) | [G§3 p3] parsimony |
| NFR-5 | Telemetry overhead | < 1 ms per event; async JSONL writer | [G§2 p2] |
| NFR-6 | Determinism | Fixed seeds, temperature 0, pinned models, so replay runs are comparable | [G§5 p4 G1] [INF] |
| NFR-7 | Memory footprint | Container RSS < 2 GB excluding a local LLM | [INF] |
| NFR-8 | Explainability | Every decision carries reason codes and features | [G§8 p6] (brief) |

### 7.2 Evaluation requirements (G)

| ID | Requirement | Source |
|---|---|---|
| EV-1 | Automated replay suite over held-out-style streaming prompts | [G§5 p4 G1], [G§7 p5] |
| EV-2 | Gate metrics G2–G6 computed automatically from telemetry | [G§5 p5] |
| EV-3 | Quantitative comparison against a baseline pipeline | [G§8 p6] |
| EV-4 | At least three analyzed edge-case failures | [G§8 p6] |
| EV-5 | At least two ablations (e.g. hybrid vs dense; rule vs model controller) | [G§8 p6] |
| EV-6 | Report retrieval recall, answer groundedness, TTFT and cost per turn | [D s7] |

### 7.3 Deployment requirements (H)

| ID | Requirement | Source |
|---|---|---|
| DP-1 | `docker compose up` or a clean CLI runner; one command | [G§8 p6], [G§5 p4 G1] |
| DP-2 | Pinned dependency lockfiles and an env template (`.env.example`) | [G§8 p6] |
| DP-3 | Runs on a clean machine (assume x86_64 Linux or Windows Docker, no GPU, unknown RAM) | [G§5 p4] [INF] |
| DP-4 | GitHub repo with README; release tag `PRISM_GENAI_HACKATHON_Y2026` | [D s13] |
| DP-5 | Model weights fetched at image build time and pinned by revision | [INF] |
| DP-6 | Multi-arch image, or build-on-host from the Dockerfile (dev machine is arm64; judges likely amd64) | [ENV] [INF] |

---

## 8. Technical Research

### 8.1 Retrieval

#### 8.1.1 Options

| Approach | How it works | Quality profile | Latency / memory (CPU) | Complexity | Fit for this task |
|---|---|---|---|---|---|
| **BM25** (lexical) | Term-frequency × IDF with length normalization | Strong on exact entities, numbers, codes and rare terms (e.g. a city name, "30", a policy name). Weak on paraphrase ("refund terms" vs "cancellation policy"). | Measured: 0.04 ms (1k) → 0.4 ms (100k chunks) per query; sparse matrix ~MBs | Low | **Essential.** Spoken queries carry precise entities, and partial transcripts are keyword-ish. |
| **Dense bi-encoder** | Embed query and chunks; cosine top-k | Strong on paraphrase and intent. Weaker on rare entities and numbers. Sensitive to chunk length and domain. | Measured: query encode 3–4 ms (MiniLM) or 22–26 ms (e5-base); exact search 1.6 ms at 100k×384; 154 MB at 100k×384 float32 | Low–medium | **Essential.** Bridges spoken phrasing to policy wording. |
| **Learned sparse** (SPLADE family) | Transformer-expanded sparse term weights | Often better than BM25 in-domain | Document expansion at index time is heavy on CPU; query encoder ~BERT-base | Medium–high | Not justified. BM25 + dense covers this at lower cost. |
| **Hybrid** (BM25 + dense) | Run both and fuse (RRF or weighted) | Usually more robust than either alone across query styles [INF, widely reported; verify on our dev set] | Sum of both: a few ms | Low | **Recommended.** Mandated by the guide [G§2 p2]. |
| **Vector index** (ANN: HNSW, IVF) | Approximate nearest neighbors | Slight recall loss vs exact | Wins only far beyond 100k chunks | Medium (extra dependency, build params) | **Not needed.** Exact numpy search is 1.6 ms at 100k. |
| **Metadata filtering** | Restrict by doc type or section and similar | Precision boost when metadata exists | Negligible in-memory | Low | **Use if** the corpus has useful metadata (unknown). Always keep `doc_id` and `section` filterable for delta retrieval ("same doc, other sections"). |
| **Multi-vector** (ColBERT/late interaction, BGE-M3 multi-vector) | Token-level embeddings with MaxSim | Strong fine-grained matching | Index ≈ tokens × dim (~100× dense); query-time cost higher; heavy on CPU | High | **Not justified** for a hackathon CPU prototype. |

#### 8.1.2 Retrieval issues specific to speech and streaming

- **Disfluencies and fillers** ("um", "like", "I need to", "you know") pollute BM25 and dilute dense queries. Normalize queries by removing a *generic* English filler and stopword list. This is not benchmark-specific, so it is allowed under C2.
- **Partial queries** are short and ambiguous, so dense retrieval on them is noisy. The controller must not retrieve on fragments (§8.5).
- **Numbers spoken as words** ("thirty people") vs digits in the corpus ("30"). The ASR output form is unknown, so normalize numerals at both index and query time.
- **Context carry-over.** Each sub-query needs the shared context: location, event type, role. In Example 1 the enriched queries carry the location and event type [G§4 p4].

**Verdict.** Use hybrid BM25 + dense with RRF. Use exact in-memory search (numpy for dense, scipy sparse for BM25). Use section-aware chunking. Add metadata filtering only if the corpus has metadata. There is no vector database (decision matrix §12.2).

### 8.2 Embeddings

**Selection criteria for this task, in priority order:**

1. Retrieval quality on short spoken queries against policy text.
2. CPU query latency, which sits on the streaming critical path.
3. Indexing throughput, which affects container start time.
4. Context length vs chunk size.
5. License.
6. Deployment simplicity: no `trust_remote_code`, ONNX available.
7. Multilingual support, only if needed.

**Candidates.** Published facts are from model cards as recalled during research; re-verify at download time. Latency is **measured** only where noted.

| Model | Params | Dim | Max tokens | Prefix needed | License | CPU latency | Notes / verdict |
|---|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 23M | 384 | 256 (truncates) | No | Apache-2.0 | **Measured:** 3.0–4.2 ms per query; 180–210 passages/s (~120 tok) | Cached locally. Very fast. Trained mainly for sentence similarity, not asymmetric query→passage retrieval, so it may underperform on short-query vs long-passage. **Speed floor and fallback.** |
| `BAAI/bge-small-en-v1.5` | 33M | 384 | 512 | Optional query instruction | MIT | Estimate: ~2× MiniLM (12 layers vs 6) → ~6–9 ms per query | Trained for retrieval; same dim as MiniLM; 512-token window fits section chunks. **Primary candidate.** |
| `intfloat/e5-small-v2` | 33M | 384 | 512 | Yes (`query:` / `passage:`) | MIT | Estimate: similar to bge-small | Strong retrieval-trained alternative; prefixes are easy to get wrong. **Secondary candidate.** |
| `Snowflake/snowflake-arctic-embed-s` / `thenlper/gte-small` | ~33M | 384 | 512 | arctic: query prefix | Apache-2.0 / MIT | Estimate: similar to bge-small | Credible alternatives. Include in Exp 1 only if time allows. |
| `intfloat/multilingual-e5-base` | 278M (mostly vocabulary embeddings) | 768 | 512 | Yes | MIT | **Measured:** 22–26 ms per query; 30–35 passages/s | Cached locally. Only worthwhile if the corpus or users are non-English (e.g. Hindi or code-mixed). Note the **compressed similarity range**: an unrelated pair scored 0.675 vs a paraphrase at 0.847 (measured), so thresholds must be calibrated per model. |
| `intfloat/multilingual-e5-small` | 118M | 384 | 512 | Yes | MIT | Estimate: between MiniLM and e5-base | The multilingual option if language support becomes a requirement. |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | 512 | Optional | MIT | Estimate: ~3–4× bge-small | Usually a quality step up; check whether it pays for its latency in Exp 1. |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 (Matryoshka) | 8192 | Yes | Apache-2.0 | Estimate: ~bge-base | Needs `trust_remote_code`. Long context is unnecessary for section chunks. **Not preferred.** |
| `BAAI/bge-m3` | 568M | 1024 + sparse + multi-vector | 8192 | No | MIT | Estimate: >10× bge-small | Very capable, but too heavy for CPU streaming. **Reject.** |
| `Qwen/Qwen3-Embedding-0.6B` | 600M | up to 1024 | 32k | Instruction | Apache-2.0 | Estimate: heavy on CPU | Quality-oriented, CPU-unfriendly. **Reject** for the critical path. |
| `google/embeddinggemma-300m` | ~300M | 768 | 2k | Task prompts | Gemma terms | Estimate: heavy-ish | License terms add review overhead. **Not preferred.** |

**Why the popular choice isn't automatically right.** MiniLM-L6 is the most-downloaded embedder and is already cached. But it truncates at 256 word-pieces and was not trained mainly for asymmetric retrieval. Section-level policy chunks plus short spoken queries is exactly the asymmetric case, so a retrieval-trained small model (bge-small or e5-small) is the better default. Experiment 1 must settle the choice on the actual corpus.

**Verdict.**

- **Default:** `bge-small-en-v1.5` (384-d, MIT).
- **Measured fast fallback:** `all-MiniLM-L6-v2`.
- **Multilingual option:** `multilingual-e5-small`, only if needed.
- **Runtime:** ONNX (onnxruntime is already a dependency family) or torch CPU ≥2.4 in a pinned env. ONNX gives a smaller image and needs no torch.

### 8.3 Reranking

| Approach | Model examples (params, license) | Quality | CPU cost | Verdict |
|---|---|---|---|---|
| **No reranker: RRF of BM25 + dense** | — | Good baseline. Rank-only, so it ignores score magnitudes and is robust without calibration. | ~0 | **Baseline and fallback.** Guide roadmap names RRF [G§7 p5]. |
| **Weighted score fusion** (α·norm(BM25) + (1−α)·cos) | — | Can beat RRF when tuned, but needs score normalization and α tuning. Fragile on a held-out corpus. | ~0 | Ablation only. |
| **Small cross-encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M, Apache-2.0); `ms-marco-MiniLM-L-12-v2` (33M); `ms-marco-TinyBERT-L-2-v2` (~4M, used by FlashRank) | Large precision gains are typical over bi-encoder retrieval [INF; verify on our dev set] | Estimate from measured MiniLM-L6 throughput (~200 seq/s at ~120 tok): **~0.15 s per 30 candidates per sub-query** on this machine; ×3 intents ≈ 0.45 s serial | **Candidate (Exp 3).** Run only on *committed* intents, not on provisional retrieval. |
| **Mid-size cross-encoder** | `BAAI/bge-reranker-base` (278M, MIT); `mixedbread-ai/mxbai-rerank-xsmall-v1` (~70M, Apache-2.0) | Higher quality | Estimate: 3–10× the L-6 cost | Only if Exp 3 shows L-6 is insufficient and the latency budget allows. |
| **Large / LLM reranker** | `bge-reranker-v2-m3` (568M), LLM listwise reranking | Highest | Seconds on CPU, or LLM tokens | **Reject.** Violates parsimony and latency. |

**Licensing note.** MS MARCO cross-encoders are Apache-2.0 on their model cards, but they were trained on MS MARCO, whose data terms are research-oriented. That is acceptable for a hackathon prototype. Flag it in the brief.

**Design rule.** Rerank **per sub-query**, before fusion across sub-queries. Otherwise one well-matched intent crowds the others out of the context window (see §8.8).

**Verdict.**

- **Baseline:** RRF.
- **Target:** RRF, then `ms-marco-MiniLM-L-6-v2` on the top ~20–30 per committed intent, adopted only if Exp 3 shows a recall/precision gain worth its latency.
- **Latency control:** rerank runs during the speech gap (between 1.6 s and 2.1 s in Ex1), so it is mostly hidden from TTFT.

### 8.4 Query Decomposition

**Goal.** Turn one compound spoken utterance into N orthogonal, search-ready sub-queries. Avoid under-splitting (merged intents), over-splitting (near-duplicate fragments) [G§6 p5 #5], and context loss [G§2 p2].

| Method | Mechanism | Strengths | Weaknesses | Latency | Reliability |
|---|---|---|---|---|---|
| **Rule / heuristic** | Clause segmentation on coordinators ("and", "also", "plus", "as well as"), sentence punctuation, and question or request heads ("what", "how", "I need", "tell me"). Noun-phrase extraction for coordinated objects. Shared-context propagation (slots: location, event type, counts, dates, roles). | Deterministic, explainable, **sub-millisecond**, works on partial transcripts | Brittle on implicit intents ("…and what if it rains?"). Over-splits fixed phrases ("terms and conditions", "cancellation and refund policy"). Needs careful generic rules. | < 1 ms | High precision on explicit coordination; low recall on implicit intents |
| **LLM free-form** | Prompt the LLM to list the sub-questions | Handles implicit and nested intents | Unstable format; may invent intents; adds latency | 0.3–2 s hosted (estimate); seconds on a local CPU LLM (estimate) | Medium |
| **LLM structured output** | JSON schema: `{retrieval_required, intents:[{text, search_query, shared_context}]}` (Anthropic structured outputs, Ollama `format` schema) | Validated schema; one call can also classify retrieval-required and extract slots | Same latency; still a model judgment | Same as above | Medium-high |
| **Hybrid (recommended)** | Rules run on every stable segment *during* the stream, which feeds early retrieval. One structured LLM call at the intent-stable point or the end of the utterance validates and repairs the set. The result is reconciled with the query ledger: matched intents reuse retrieval and only new or changed intents trigger retrieval. | Early retrieval comes from rules; the LLM catches implicit intents; at most one LLM call per utterance | Two code paths; needs reconciliation logic | Rules on the critical path; the LLM call overlaps the remaining speech | High |

**Over-decomposition guard, using the corpus and no hardcoding.** If two candidate sub-queries have heavily overlapping top-k results (Jaccard of chunk IDs ≥ θ) **or** near-identical embeddings (cos ≥ τ), merge them. A phrase like "cancellation and refund policy" will hit the same chunks for both halves and collapse into one intent. Policy-specific word lists are not needed.

**Under-decomposition guard.** If a single sub-query's top results split into two distinct clusters (different documents and sections, low inter-cluster similarity) *and* the utterance contains a coordinator, flag it as a split candidate for the LLM check.

**Experiment hook.** Rule vs LLM vs hybrid decomposition, scored for G3 accuracy, precision (over-splitting) and latency (Exp 4b).

### 8.5 Retrieval Controller

**Decision space** [G§2 p2]: `WAIT`, `RETRIEVE` (with trigger type), `NO_RETRIEVE` (with reason). The controller runs on every chunk and once more on end of utterance.

#### 8.5.1 Signals (all cheap, generic and logged)

| Signal | Computation | Purpose |
|---|---|---|
| **Retrieval-worthiness** | Request/question cues vs **presentation cues** (repeat, rephrase, shorten, bullets, translate, simplify) combined with **anaphora to the prior answer** ("your last answer", "that") and a non-empty answer history; vs **social cues** (greetings, thanks, acknowledgments) | Separate Example 3 turns from real queries. "Summarize the *travel reimbursement rule*" needs retrieval; "summarize *your last answer*" does not. The **object** of the verb decides, not the verb. |
| **Content sufficiency** | Count of content tokens after filler and stopword removal; presence of ≥1 *anchor* (noun phrase or entity found in the **corpus vocabulary**, using the index's own IDF table) | Do not search on "I need to plan a…". Corpus vocabulary coverage is a dictionary lookup, not a search, so it costs no retrieval event. |
| **Boundary / completeness** | Does the segment end in a dangling function word (preposition, article, conjunction, "I need…")? Did a clause boundary just appear (comma + coordinator, sentence end, end of utterance)? | Example 1: the first chunk ends on a dangling preposition, so WAIT. The second chunk closes the first clause with a comma and coordinator, so RETRIEVE on the closed clause. |
| **Semantic stability** | Similarity between the segment's normalized query at t and t−1: term-set Jaccard, or embedding cosine for one ~4 ms encode | Retrieve once the query has stopped changing materially. |
| **Novelty vs ledger** | Max similarity of the candidate query to queries already in the session's query ledger | Prevents duplicate retrievals [G§6 p5 #5]. Enables reuse (Ex1). |
| **Corpus affinity** (optional) | Fraction of anchor terms with corpus IDF above a floor | Out-of-corpus chit-chat (e.g. weather) should not trigger retrieval; or it should be answered with "not in the knowledge base". |
| **Budget** | Minimum interval between provisional retrievals; maximum provisional retrievals per utterance | Prevents thrashing [G§6 p5 #1]. |

#### 8.5.2 Policy (rule-first; thresholds are calibrated, not hand-tuned per test)

```
on chunk:
  update segments (OPEN/CLOSED) and slots
  if worthiness == presentation or social with high confidence → NO_RETRIEVE(reason)   # may decide early
  for each CLOSED or STABLE segment not yet covered:
      if sufficiency and anchor present and novel → RETRIEVE(trigger = provisional | multi_intent)
  otherwise → WAIT(reason codes)
on end of utterance:
  close all segments; RETRIEVE any uncovered worthy segment (trigger = final)
  optionally run one LLM decomposition check; reconcile with the ledger
  if nothing was worth retrieving → NO_RETRIEVE(reason)
```

#### 8.5.3 Why rules first, model second

- **Timing.** The guide's Example 1 logs `retrieval_started` at the *same timestamp* as the chunk that made it possible (0.8 s) [G§4 p3–4]. An LLM controller adds hundreds of milliseconds to seconds per chunk, which pushes retrieval later and erodes G2.
- **Cost of a wrong early retrieval is low.** Retrieval costs milliseconds, and provisional results go into a candidate pool that is **re-scored against the final sub-queries before synthesis**. Premature evidence never reaches the answer unless it survives. The controller can therefore lean early on *retrieval-worthy* utterances, while being strict on *no-retrieval* turns, where a false trigger counts against G2 [G§5 p5].
- **The ablation is required anyway.** Rule-based vs model-based controller is explicitly suggested [G§8 p6]. The model-based variant is either:
  - **(a)** a small LLM classifier called on each closed segment, or
  - **(b)** an embedding-prototype classifier: nearest prototype among generic seed phrases for *information request*, *presentation request* and *social*, stored in config.

  Option (b) costs ~4 ms. Its seed phrases must stay generic and must never be benchmark utterances (C2). See §19 Q8.

#### 8.5.4 Real ASR caveat

Real streaming ASR revises earlier partial hypotheses. The guide's chunks are append-only. Design the segment tracker so a chunk can *replace* the trailing partial, but treat this as a stretch goal.

### 8.6 Streaming

**Input simulation.**

- A replay file contains sessions, each session contains turns, and each turn is a list of `{timestamp_s, text}` plus an end marker. JSONL is the format.
- Two clock modes:
  - **Real-time** (sleep to timestamps, with a speed factor): used for official latency numbers.
  - **Virtual-time**: events processed immediately, with stream time and wall time recorded separately. This is fast for CI.
- G2 is defined on the **stream timeline**: retrieval start time vs utterance end time.
- In real-time mode, processing latency is *real*. A retrieval that starts at 0.8 s must also *finish* before 2.1 s to be useful. Report both `retrieval_started` and `retrieval_completed`.

**Event-driven core.** Use one asyncio loop per process and one queue per session. Typed events:

| Group | Events |
|---|---|
| Input | `ChunkReceived`, `UtteranceEnded` |
| Controller | `ControllerDecision` |
| Decomposition | `SegmentClosed`, `SubQueriesUpdated` |
| Retrieval | `RetrievalStarted`, `RetrievalCompleted`, `LedgerHit`, `RerankCompleted`, `EvidenceFused` |
| Synthesis | `SynthesisStarted`, `FirstToken`, `TokenStreamed`, `AnswerVersionCommitted` |
| Grounding | `CitationValidated`, `UnsupportedClaimFlagged`, `UncertaintyEmitted` |
| Turn | `TurnCompleted` |

Every event carries `session_id`, `turn_id`, `seq`, `t_stream_s`, `t_wall_ms`, `component` and `payload`. The telemetry sink is a subscriber, so tracing is complete by construction (G6).

**Concurrency.**

- Retrieval is CPU-bound and ms-scale. Run embedding and rerank through `asyncio.to_thread` or a small thread pool so the loop stays responsive. numpy BLAS releases the GIL.
- Parallel sub-query retrieval is `asyncio.gather`.
- Real parallelism matters most for **I/O-bound LLM calls**: a decomposition check in parallel with ongoing speech.

**Cancellation.**

- Provisional retrieval is too fast to need cancelling.
- What must be cancellable:
  - **(a)** an in-flight LLM decomposition check when the utterance changes materially;
  - **(b)** an in-flight synthesis when a new chunk arrives for the same turn (barge-in), or when a refinement turn starts.
- Use `asyncio.Task.cancel()` with cleanup that logs `cancelled` events and counts wasted tokens.

**Output streaming.**

- **CLI:** tokens to stdout plus a JSONL trace.
- **Demo:** FastAPI with SSE or WebSocket pushing tokens and controller events to a minimal page.
- **TTFT:** the first token's wall time minus the wall time of the end of the utterance. Also report from the first chunk.

### 8.7 Session Refinement

**Session state.** In memory, per `session_id`, destroyed at session end.

```
Session
├── turns[]                 transcript, decisions, timings
├── frame                   active topic + slots/constraints (e.g., topic=travel reimbursement; trip_type=?; booking_timing=?)
├── query_ledger[]          {query_norm, embedding, intent_ids, trigger, t_stream, result_ids, scores}
├── evidence_store{}        chunk_id → {text, doc_id, section, scores, retrieved_by[]}
├── answers[]               AnswerVersion {version, parent, kind: full|refinement|presentation,
│                                          claims[], citations[], uncertainty, text, diff}
└── claims[]                {claim_id, text, citation_ids[], status: active|modified|retracted, intent_id}
```

**Late-detail flow** (Example 2):

1. **Classify the new turn** using retrieval-worthiness plus *topic continuity*. Topic continuity means similarity to `frame.topic` plus anaphora such as "the trip" or "that", or constraint markers such as "actually", "also", "it was", "applies when". The four outcomes are REFINE, NEW_TOPIC, PRESENTATION and SOCIAL.
2. **Extract delta constraints.** Use rules for slot-like constraints and the LLM structured call when needed. Example: `trip_type=international` and `booking_timing=after_travel`.
3. **Delta queries** = base topic + each constraint, e.g. "travel reimbursement international travel". The ledger blocks re-running v1's queries. Use `trigger: refinement` and optionally restrict to the documents already in play.
4. **Merge** new evidence into `evidence_store`.
5. **Affected-claim tracking.** Mark v1 claims whose intent or slots touch the constraint as *candidates for modification*. Others are *retained*.
6. **Incremental synthesis.** The prompt contains the v1 claims with their citation labels, the new evidence, and the constraint. It instructs the model to keep claims that remain valid, revise affected ones, add new ones, and cite only from the labels.
7. **Commit v2** with `parent=v1` and a diff of retained, modified, added and retracted claims and citations. Log `full_corpus_rerun=false` plus the delta query count.

**Why not just re-run everything with the merged utterance?** That is the "restart" baseline. It doubles latency and tokens, and it can silently change v1 statements and citations that were correct (pitfall #2 [G§6 p5]). Experiment 6 measures this.

**Answer delta vs full regeneration.** True in-place claim mutation (editing only affected sentences) is the guide's ideal [G§2 p2]. A pragmatic middle ground is to regenerate the *text* while constraining *claims*: retained claims must be preserved, possibly reworded, with their citations. The verifier then checks that every retained claim still appears and is still cited. Claim-level structured output makes this checkable.

### 8.8 Evidence Fusion

| Technique | Use |
|---|---|
| **RRF within a sub-query** (BM25 ⊕ dense, k≈60) | Rank fusion that needs no score calibration [G§7 p5] |
| **Exact dedup by chunk_id across sub-queries** | One chunk retrieved by two intents is kept once and tagged with both |
| **Near-duplicate dedup** (cos ≥ 0.95, or shingle Jaccard) | Corpora often repeat boilerplate across documents |
| **Per-intent quota** (e.g. top-3 per intent after rerank) instead of a global top-k | Guarantees every intent is represented; prevents a strong intent from starving a weak one. This directly supports G3 and the uncertainty logic. |
| **MMR / source diversity** | Avoid three chunks from the same section saying the same thing |
| **Evidence density** | Prefer chunks with high query-term coverage per token; optionally trim to the best sentences within long chunks while keeping the chunk ID for citation |
| **Context budget** | E.g. ≤3 intents × 3 chunks × ~200 tokens ≈ 1.8k tokens. This matters most for **local CPU LLM** time-to-first-token, because prefill cost scales with prompt length. |
| **Contradiction detection** | Cheap: same slot type (amount, days, approver) with different values across cited chunks, so flag it. Expensive: NLI cross-encoder between top chunks. Or instruct the LLM to surface conflicts as uncertainty. Start cheap. |
| **Coverage scoring per intent** | Best reranked score vs a calibrated threshold. Below threshold means the intent is *unsupported*, which feeds `uncertainty` (C3). |

### 8.9 Grounding & Citations

The goal is **zero fabricated IDs** (hard) and **≥85% of sampled claims supported** [G§5 p5 G4].

| Mechanism | How | Guarantees / catches |
|---|---|---|
| **Label-constrained citation** | The evidence block lists `[E1]…[En]`, each mapped server-side to `Doc_ID §Section`. The LLM cites only labels. A post-processor maps labels to real IDs and **drops and flags any unknown label**. | **Fabricated IDs are impossible by construction**: the model never writes a real doc ID itself. |
| **Sentence-level claim extraction** | Split the answer into sentences. Each factual sentence is a claim that must carry ≥1 label. (Or use a structured output of `[{claim, labels}]`, rendered to prose.) | Finds uncited claims |
| **Support verification** | For each (claim, cited chunk): **(1)** numbers, dates and amounts must appear in the cited chunk (cheap, high precision); **(2)** content-word overlap or embedding similarity to the chunk's best sentence; **(3)** optionally a small NLI cross-encoder for entailment (model to be selected; CPU cost must be measured) | Catches parametric leakage and miscitation |
| **Action on failure** | Attach an uncertainty marker to that claim. Policy options: drop the claim, re-ask once, or keep it flagged. **Never** silently pass it. | Keeps G4 honest |
| **Per-intent uncertainty** | Uncovered intents are listed explicitly ("X could not be verified from the corpus"), mirroring the guide's `uncertainty` field [G§4 p4] | C3 |
| **Clarification** | If an utterance is ambiguous (unresolved referent, missing slot that decides the answer), ask one targeted question instead of guessing [G§3 p3] | C3 |
| **Presentation turns** | The transformation prompt receives only prior claims and labels. The verifier asserts *new citations ⊆ prior citations* and *no new numbers* [G§4 p4 Ex3]. | GR-4 |
| **Extractive mode (no LLM)** | The answer is assembled from top evidence sentences per intent, with citations | 100% citation validity by construction. Baseline floor and G1 fallback. |

**Prompt-level grounding.** The system prompt (kept in config, not code) instructs the model to:

- answer **only** from the evidence;
- cite after every factual sentence;
- state explicitly when something is not in the evidence.

The verifier still matters, because instructions alone don't guarantee compliance.

---

## 9. Benchmark Behavior Analysis

**Reading the timelines.**

- `t` is **stream time** in seconds, as in the guide.
- Chunk text is **paraphrased** from the guide's examples. Key entities are kept.
- The actions describe how the **recommended architecture (§18)** would behave. Nothing here is a measured result.
- Retrieval latencies of "~ms" come from the micro-benchmarks in §8.1.

### 9.1 Scenario 1: Incremental multi-intent utterance [G§4 p3–4]

**Input:** one turn, three chunks, and an end marker at 2.1 s. The user is planning a customer workshop in Pune for 30 people and needs the cancellation policy and catering options.

```
t=0.0s  CHUNK  [user begins: needs to plan a customer workshop; chunk ends on a dangling preposition "in…"]
        CONTROLLER sees: content tokens {plan, customer, workshop}; segment S1 OPEN (trailing function word);
                         no location/count slot; anchor "workshop" (if present in corpus vocabulary)
        DECISION: WAIT
        REASON:   trailing_function_word, low_specificity
        LOG:      controller_decision{t=0.0, decision=WAIT, reasons, features}

t=0.8s  CHUNK  [location "Pune" + headcount "30 people" + ", and I need…"]
        CONTROLLER sees: ", and" closes S1 ("plan a customer workshop in Pune for 30 people");
                         slots: location=Pune, count=30, event=customer workshop;
                         S1 sufficient + anchored + novel (ledger empty); new segment S2 OPEN ("I need…", dangling)
        DECISION: RETRIEVE (trigger=provisional) for S1;  WAIT for S2
        ACTION:   Q1 = normalized S1 + slots → e.g. "customer workshop Pune 30 people"
                  hybrid search (BM25 ⊕ dense, RRF) — few ms; results → candidate pool; ledger += Q1 (intent I1)
        NOTE:     the guide's provisional query mentions "venue capacity", words the user never said.
                  A rule-built query will not contain them; dense retrieval must bridge "plan a workshop for 30" → venue/capacity
                  sections. Exp 4b checks whether an LLM rewrite is worth its latency.
        LOG:      retrieval_started{t=0.8, trigger=provisional, query=Q1}; retrieval_completed{latency_ms, top_ids}

t=1.6s  CHUNK  [the cancellation policy and the catering options. (sentence end)]
        CONTROLLER sees: S2 CLOSED; head "I need" + two coordinated noun phrases → intents I2 (cancellation policy),
                         I3 (catering options); I1 already in ledger
        DECOMPOSER: rules → {I1 venue for 30 in Pune, I2 cancellation terms, I3 catering options}
                    context propagation: frame slots {Pune, workshop} appended to I2/I3 retrieval queries
                    over-split guard: I2 vs I3 top-k overlap is low → keep separate
        DECISION: RETRIEVE (trigger=multi_intent) for I2, I3 — in parallel;  LedgerHit for I1 (no new search)
        ACTION:   intents now committed → per-intent cross-encoder rerank (I1..I3), hidden in the 0.5 s before utterance end
                  OPTIONAL (confidence-gated): one structured LLM decomposition check, async
        LOG:      sub_queries_updated{3}; retrieval_started×2{t=1.6, trigger=multi_intent}; ledger_hit{I1→Q1};
                  rerank_completed{per intent, latency}; [llm_call{purpose=decompose_check, tokens}]

t=2.1s  [UTTERANCE END]
        CONTROLLER: final pass — nothing new; all three intents covered → SYNTHESIZE
                    (if an LLM check is in flight: wait ≤ a short cap, e.g. 300 ms, else proceed with rule intents)
        FUSION:   per-intent quota (e.g. top-3 each), dedup by chunk_id, coverage score per intent
        SYNTHESIS: one LLM call, evidence labelled [E1..En], streamed tokens
        GROUNDING: map labels → Doc_ID §Section; verify numbers/overlap; intents/aspects lacking support → uncertainty
                   (guide's example flags catering accommodation for one venue as unverified)
        COMMIT:   Answer v1 {claims, citations, uncertainty}
        LOG:      synthesis_started{t=2.1}; first_token{ttft_ms}; answer_version_committed{v1};
                  turn_completed{retrieval_calls=3, llm_calls=1–2, tokens_in/out, cost}
```

**Why this matches the guide's output record.** There are three `retrieval_events`: one provisional and two multi-intent. There are three `sub_queries`, one unified answer, citations, and an `uncertainty` string [G§4 p4]. The early-retrieval **lead time** is 2.1 − 0.8 = **1.3 s** for I1 and **0.5 s** for I2 and I3.

### 9.2 Scenario 2: Late-arriving detail, refine without restarting [G§4 p4]

The guide gives no timestamps for this example, so the times below are illustrative.

```
TURN 1 — user asks for a summary of the travel reimbursement rule for an employee trip
t=0.0s  CHUNK  [summarize the travel reimbursement…]
        CONTROLLER: verb "summarize" + object = a corpus topic (not "your answer"; session has no prior answer)
                    → retrieval-worthy; anchor "travel reimbursement" in corpus vocabulary; segment still OPEN
        DECISION: WAIT (or RETRIEVE provisional if the noun phrase is already closed and stable)
t=0.9s  CHUNK  […rule for an employee trip.]
        DECISION: RETRIEVE (provisional/final) single intent I1 = travel reimbursement rule (employee trip)
t=1.3s  [UTTERANCE END] → SYNTHESIZE v1
        STATE:    frame.topic = travel reimbursement; slots {trip_type: unspecified, booking_timing: unspecified}
                  claims c1..cn with citations; ledger {Q1}
        LOG:      answer_version_committed{v1, citations, claims}

TURN 2 — user adds: the trip was international and the booking was made after travel
t=0.0s  CHUNK  [the trip was international…]
        CONTROLLER: declarative, not a question; anaphora "the trip" → refers to frame.topic; constraint pattern
                    → turn type REFINE (not a new topic, not presentation)
        DELTA:    constraint k1: trip_type=international
        DECISION: RETRIEVE (trigger=refinement) D1 = "travel reimbursement international travel"   ← early, mid-utterance
t=1.0s  CHUNK  […and the booking was made after travel.]
        DELTA:    constraint k2: booking_timing=after_travel
        DECISION: RETRIEVE (trigger=refinement) D2 = "travel reimbursement booking made after travel exception"
        LEDGER:   Q1 is NOT re-issued (no full-corpus re-run)
t=1.5s  [UTTERANCE END] → INCREMENTAL SYNTHESIS
        AFFECTED CLAIMS: v1 claims about the standard rule → retained (still valid);
                         new claims from D1/D2 evidence → added (the guide's example: an approval requirement for late
                         booking and a receipt-verification requirement for foreign currency)
        COMMIT:   Answer v2 {parent=v1, retained citations from v1 + delta citations}
        LOG:      answer_version_committed{v2, parent=v1, diff{retained, modified, added}, delta_queries=[D1,D2],
                  full_corpus_rerun=false}; retrieval_events only with trigger=refinement
```

**G5 evidence produced:**

- Version lineage v1→v2.
- v1 citations are retained.
- Retrieval in turn 2 is delta-only: the ledger shows no re-issue of Q1.
- Session state persisted across turns.

### 9.3 Scenario 3: Query suppression, no retrieval required [G§4 p4]

```
TURN 3 — user asks to repeat the last answer as two bullets
t=0.0s  CHUNK  [please repeat your last answer…]
        CONTROLLER: presentation verb "repeat" + object "your last answer" (anaphora to assistant output)
                    + session.answers non-empty → class PRESENTATION with high confidence
        DECISION: NO_RETRIEVE  (retrieval_required=false, reason=presentation_restructure) — decided before the turn ends
t=0.7s  CHUNK  […in two bullets.]
        CONTROLLER: format constraint {bullets: 2}; decision unchanged
t=1.0s  [UTTERANCE END]
        ACTION:   one LLM transform call over the latest answer version's claims + labels only (no new evidence);
                  verifier asserts new_citations ⊆ prior_citations, no new numbers, exactly 2 bullets
        COMMIT:   Answer v3 {kind=presentation, parent=v2, evidence unchanged}
        LOG:      controller_decision{NO_RETRIEVE, presentation_restructure}; retrieval_events=[] for this turn;
                  llm_calls=1; answer_version_committed{v3, kind=presentation}
EDGE:   if the session has no prior answer → reply that there is nothing to restate; still no retrieval.
```

---

## 10. Evaluation Gates

These gates come from [G§5 p4–5]. The guide does not say how the private harness computes them (§19 Q3–Q5). Below is how **we** will measure them so that our numbers can be defended.

### G1: Reproducibility (pass/fail)

| Aspect | Plan |
|---|---|
| Measures | Whether a clean machine can launch everything with one command, and whether the replay suite finishes with no manual intervention |
| Why it matters | It is pass/fail. Failing it likely voids every other score. |
| Implementation | `docker compose up` (app + optional Ollama sidecar) or `make replay`. Models baked in at build time. Corpus mounted from `./corpus`. `.env.example`. Replay finishes **without API keys** via local or extractive fallback. |
| Benchmark | CI-style test in a fresh VM or container: `git clone` → one command → assert the replay report exists. Test on amd64 as well as the arm64 dev machine. |
| Telemetry | Run manifest: git SHA, image digest, model revisions, corpus hash, backend used, start and end times |
| Failure risks | Missing API key; model download fails at runtime; arm64-only image; port conflicts; slow first-start index build that looks hung; unpinned dependencies drifting (already observed in the global env) |

### G2: Early retrieval (≥80% of eligible queries, low false-trigger rate)

| Aspect | Plan |
|---|---|
| Measures | Share of retrieval-needing utterances where the first retrieval starts **before** the utterance ends. Also the false-trigger rate on no-retrieval utterances. |
| Why it matters | This is the core "full-duplex" value [D s7] |
| Implementation | Rule-first controller (§8.5); retrieval within the same chunk-processing tick |
| Benchmark | Replay a labeled set: `needs_retrieval ∈ {true, false}`, multi-chunk turns. **Eligible** = needs retrieval and has ≥2 chunks (our definition, §19 Q4). `early_rate = #eligible with t_first_retrieval_start < t_utterance_end / #eligible`. `false_trigger_rate = #no-retrieval turns with ≥1 retrieval event / #no-retrieval turns`. Also report *useful early rate*: early retrievals that contributed ≥1 cited chunk. |
| Telemetry | `ChunkReceived.t_stream`, `RetrievalStarted.t_stream`, `UtteranceEnded.t_stream`, controller reasons |
| Failure risks | Single-chunk utterances (structurally impossible to retrieve early); controller too conservative; LLM on the critical path; early retrievals on noise that never contribute (inflates the rate without value) |

### G3: Multi-intent identification (≥70% of compound queries)

| Aspect | Plan |
|---|---|
| Measures | Whether ≥2 distinct sub-intents are correctly identified and isolated in compound utterances |
| Implementation | Hybrid decomposition (§8.4) with over- and under-split guards |
| Benchmark | Gold intents per compound utterance, annotated by us. Match predicted to gold by semantic similarity ≥ τ with one-to-one Hungarian assignment, spot-checked by a human. `G3_pass(u) = (#matched ≥ 2) ∧ (#matched == #gold or ≥ 2)`; report both strict and lenient variants. Also report sub-query precision to penalize over-splitting. |
| Telemetry | `SubQueriesUpdated` (final list + provenance: rule or LLM), retrieval queries per intent |
| Failure risks | Merged intents; split fixed phrases; lost context (sub-query lacks the location); implicit intents missed by rules; the harness may judge by exact strings (unknown) |

### G4: Factual grounding (≥85% citation support, zero fabricated IDs)

| Aspect | Plan |
|---|---|
| Measures | Share of sampled factual assertions supported by the cited chunks; whether any citation refers to a nonexistent document |
| Implementation | Label-constrained citations, post-hoc ID validation, claim support verification, uncertainty (§8.9) |
| Benchmark | **(1)** ID validity: 100% automatic. **(2)** Support rate: an automatic verifier (numbers + overlap + optional NLI) on every claim, calibrated against a **human-annotated sample** (≥100 claims) so the automatic judge's agreement is reported. **(3)** Unanswerable intents: share flagged as uncertain. |
| Telemetry | Claims, citation labels → IDs, verifier verdicts, dropped labels |
| Failure risks | The LLM merges evidence across chunks so the claim is supported only jointly; partial support; paraphrase the verifier can't match; a stricter external judge; parametric leakage on familiar topics |

### G5: Session refinement (verified state continuity)

| Aspect | Plan |
|---|---|
| Measures | Whether late constraints update the existing answer without clearing state or re-running full-corpus search |
| Implementation | Query ledger, frame and slots, versioned answers, delta retrieval, affected-claim tracking (§8.7) |
| Benchmark | Multi-turn sessions with gold deltas. Assert `v_n.parent == v_{n−1}`, `full_corpus_rerun == false`, `reissued_prior_queries == 0`, and retained-citation ratio ≥ threshold. Report the stale-fact rate (claims invalidated by the constraint but still present). |
| Telemetry | `AnswerVersionCommitted{version, parent, kind, diff}`, refinement retrieval events, ledger hits |
| Failure risks | A refinement misclassified as a new topic, which restarts; a new topic misclassified as a refinement, which contaminates the answer; a delta query missing the topic context; the LLM rewriting retained claims with different facts |

### G6: Telemetry and observability (100% trace coverage)

| Aspect | Plan |
|---|---|
| Measures | Whether every execution has structured traces with timestamps, retrieval triggers, citations, version lineage and token cost |
| Implementation | Event bus with a telemetry subscriber (§8.6); JSON-Schema-validated records; a per-turn summary record in the guide's format |
| Benchmark | `coverage = #turns whose trace validates against the schema and contains all required fields / #turns`. Required to be 100%. Fails CI if lower. |
| Telemetry | The telemetry is itself the thing measured. Token cost needs the provider's `usage` (hosted) or tokenizer counts (local), plus a price table in config. |
| Failure risks | Exceptions bypass logging; cancelled tasks log nothing; token usage is missing for streamed responses; async writer loses the tail on shutdown |

**Jury criteria beyond the gates** [D s11]: working prototype 30%, technical depth 25%, innovation 20%, relevance 15%, presentation 10%. The live demo (§18) must make the gates *visible*. A timeline view of the controller and retrieval events directly serves the "working prototype" and "technical depth" criteria.

---

## 11. Failure Mode Analysis

| # | Failure | Cause | Observable symptom | Impact | Detection | Mitigation |
|---|---|---|---|---|---|---|
| 1 | Premature retrieval | Retrieving on OPEN or dangling segments; thresholds too low | Retrieval events with fragment queries; evidence irrelevant to the final intents | Noise in candidate pool; wasted compute; G2 inflated without value | `useful_early_rate` low; share of provisional results never cited | Boundary and sufficiency signals; candidate pool re-scored against final sub-queries before synthesis |
| 2 | Retrieval too late | Controller waits for the end of the utterance; LLM controller on the critical path | `t_retrieval_start ≥ t_utterance_end` | G2 miss; higher TTFT | G2 early rate; lead-time distribution | Rule-first controller; retrieve on clause closure |
| 3 | Retrieval on non-retrieval turns | Presentation or social turns misclassified | Retrieval events on presentation turns | G2 false triggers; new or fabricated citations on restated answers (GR-4) | False-trigger rate on Category E | Presentation/anaphora/history signals; strict threshold for NO_RETRIEVE vs RETRIEVE |
| 4 | Wrong decomposition | Rule mis-parse; context dropped | Sub-queries that don't match the user's needs | Wrong evidence; G3 miss | Sub-query recall vs gold | Hybrid with an LLM check when confidence is low; context propagation |
| 5 | Over-decomposition | Splitting fixed phrases or repeated asks | Near-duplicate sub-queries; reranker pollution [G§6 p5 #5] | Token waste; diluted context | Sub-query precision; duplicate rate | Top-k overlap merge; embedding near-dup merge |
| 6 | Under-decomposition | No explicit coordinator; implicit intents | One sub-query covers two needs; one need unanswered | G3 miss; incomplete answer | Intent recall; coverage per gold intent | LLM check on long or multi-cue utterances; cluster-split detector |
| 7 | Duplicate queries | No memory of issued queries; repeated phrasing | The same query retrieved twice | Cost; G5 "re-execution" | Ledger duplicate counter | Query ledger with normalized and semantic matching |
| 8 | Poor hybrid retrieval | Bad tokenization (numbers, fillers); embedder truncation; chunk granularity | Gold chunks missing from top-k | Ungrounded or uncertain answers | Recall@k per intent on the dev set | Query normalization; section-aware chunking; embedder choice (Exp 1–2) |
| 9 | Bad reranking | Cross-encoder domain mismatch (web QA vs policy text); truncation | Rerank lowers Recall@k vs RRF | Wrong evidence promoted | Exp 3 A/B; rank-change analysis | Adopt rerank only if Exp 3 shows gains; per-intent rerank; fall back to RRF |
| 10 | Conflicting evidence | Several documents or versions disagree; general rule vs exception | Contradictory claims, each cited | Misleading answer | Slot-value conflict check; NLI (optional) | Surface the conflict as uncertainty; prefer the more specific section when the corpus states precedence (no hardcoded precedence) |
| 11 | Stale evidence after late details | v2 reuses v1 evidence that the constraint makes irrelevant | v2 keeps claims the constraint invalidates | Wrong refined answer | Stale-fact rate on Category F/J | Affected-claim marking; constraint-aware re-scoring of the evidence store; explicit "modified" claims |
| 12 | Restart instead of refinement | Refinement misclassified as a new topic; naive implementation | Turn 2 re-issues v1 queries; v1 citations lost | G5 fail; latency and tokens doubled | Ledger re-issue count; lineage check | Topic-continuity classifier; ledger blocks re-issue; tests in Category F |
| 13 | Unsupported generated claims | Parametric knowledge; over-generalization | Claims without support in the cited chunks | G4 fail; C1 violation | Claim verifier; human sample | Label-constrained prompt; verifier; flag or drop |
| 14 | Incorrect citations | Label mix-up; citing a neighbor chunk | Valid ID, but the chunk doesn't support the claim | G4 fail | Per-citation support check | Sentence-level citation; verifier re-maps to the best-supporting label among the cited ones |
| 15 | Session contamination | Shared global state; caches keyed without session_id; reading telemetry back | Evidence or claims from session A appear in B | C4 violation; wrong answers | Concurrent-session isolation test (two sessions, disjoint topics) | All state keyed by session; no global mutable caches except the read-only corpus index |
| 16 | Excessive LLM calls | LLM controller per chunk; LLM decomposition always; retries | >2 LLM calls per turn | Cost, latency; parsimony penalty | `llm_calls` per turn in telemetry | Rules on the critical path; confidence-gated LLM check |
| 17 | Excessive latency | Local CPU LLM prefill; big contexts; serial rerank | High TTFT | UX; jury perception | Stage latency breakdown | Small context budget; rerank during speech gaps; hosted LLM option; streaming output |
| 18 | Excessive token usage | Large evidence blocks; whole sections as context; verbose prompts | High tokens per turn | Cost per turn | Token telemetry | Per-intent quota; sentence trimming; compact prompts |
| 19 | CPU bottlenecks | Embedding big batches at start-up; cross-encoder on many candidates; a local LLM competing for cores | Index build looks hung; latency spikes | G1 risk; latency | Start-up timing; CPU time per stage | Index cache by corpus hash; cap rerank candidates; thread limits; separate container for the LLM |
| 20 | Streaming race conditions | Chunk arrives during synthesis; overlapping retrievals finish out of order; cancelled tasks partially update state | Answer built on stale intent set; duplicate versions; missing events | Wrong answers; trace gaps (G6) | Sequence numbers per event; invariants checked in tests | One queue per session (serialized state mutation); immutable snapshots for synthesis; cancel and restart on new content; version commits are atomic |

**Edge-case failures to analyze in the report (≥3 required [G§8 p6]).** Pick from #1/#3 (controller), #5/#6 (decomposition), #11/#12 (refinement) and #13/#14 (grounding). Each will have a replay trace.

---

## 12. Technology Decision Matrices

**Scoring.** 1 (poor) to 5 (best) per criterion, from the evidence in §8. Scores are *qualitative judgments*, except where they rely on measured latency [ENV]. **Bold** marks the chosen option. "Bench" = benchmark suitability (supports the gates and ablations).

### 12.1 Embedding model

| Option | Quality (expected) | Latency | CPU fit | Memory | Complexity | License | Bench | Notes |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 3 | 5 (measured 3–4 ms) | 5 | 5 | 5 (cached) | 5 | 4 | Symmetric-similarity training; 256-token cap |
| **bge-small-en-v1.5** | 4 | 4 (estimate) | 5 | 5 | 4 | 5 | 5 | Retrieval-trained; 512 tokens |
| e5-small-v2 | 4 | 4 (estimate) | 5 | 5 | 3 (prefixes) | 5 | 5 | Close alternative |
| bge-base-en-v1.5 | 4–5 | 3 (estimate) | 4 | 4 | 4 | 5 | 4 | Check in Exp 1 if small underperforms |
| multilingual-e5-small | 3–4 | 4 | 5 | 4 | 3 | 5 | 3 | Only if multilingual is needed |
| multilingual-e5-base | 4 | 3 (measured 22–26 ms) | 4 | 3 | 3 | 5 | 3 | Cached; compressed cosine range |
| bge-m3 / Qwen3-Emb-0.6B | 5 | 1–2 | 2 | 2 | 2 | 5 | 2 | Over budget |

**Decision.** bge-small-en-v1.5 by default; MiniLM as the measured fallback. Final choice by Exp 1 on the corpus.

### 12.2 Vector store

| Option | Latency @≤100k | Memory | Complexity | Reproducibility | Explainability | Notes |
|---|---|---|---|---|---|---|
| **numpy exact (in-process)** | 5 (measured 1.6 ms @100k) | 4 | 5 | 5 | 5 | Exact; zero dependencies |
| FAISS flat/HNSW | 5 | 4 | 3 | 4 | 4 | Useful only at much larger scale |
| Chroma (embedded) | 4 | 3 | 3 | 3 | 3 | Extra persistence layer; already in global env, but conflicts are likely |
| Qdrant / Weaviate (server) | 4 | 3 | 2 | 3 | 3 | Extra container; overkill |
| LanceDB / sqlite-vec | 4 | 4 | 3 | 4 | 3 | Fine, but unnecessary |

**Decision.** numpy exact search plus a scipy-sparse BM25 matrix, persisted as `.npy` + JSON and keyed by corpus hash.

### 12.3 Retrieval strategy

| Option | Quality | Latency | Complexity | Robustness on held-out corpus | Guide alignment | Notes |
|---|---|---|---|---|---|---|
| BM25 only | 3 | 5 | 5 | 3 | 2 | Misses paraphrase |
| Dense only | 3 | 5 | 5 | 3 | 2 | Misses entities and numbers; **baseline B0** |
| **Hybrid RRF** | 4 | 5 | 4 | 5 | 5 | No calibration needed [G§7 p5] |
| Hybrid weighted | 4 | 5 | 3 | 3 | 4 | Needs α tuning; ablation only |
| + SPLADE | 4–5 | 2 | 2 | 4 | 3 | Heavy |
| ColBERT / multi-vector | 5 | 2 | 1 | 4 | 3 | Heavy |

### 12.4 Reranker

| Option | Quality gain (expected) | Latency | CPU fit | Complexity | Notes |
|---|---|---|---|---|---|
| None (RRF only) | — | 5 | 5 | 5 | Baseline and fallback |
| **ms-marco-MiniLM-L-6-v2** | 4 | 3 (~0.15 s / 30 candidates, estimate) | 4 | 4 | Committed intents only |
| ms-marco-TinyBERT-L-2-v2 | 2–3 | 5 | 5 | 4 | Fastest; lower quality |
| bge-reranker-base / mxbai-xsmall | 4–5 | 2 | 3 | 3 | Only if L-6 is insufficient |
| LLM rerank | 5 | 1 | 1 | 2 | Rejected (parsimony) |

**Decision.** RRF in the baseline. MiniLM-L-6 cross-encoder in the target, conditional on Exp 3.

### 12.5 LLM approach (synthesis and the optional decomposition check)

| Option | Quality / instruction following | TTFT | Cost per turn | CPU / G1 reproducibility | Complexity | Notes |
|---|---|---|---|---|---|---|
| Hosted small: Claude Haiku 4.5 (`claude-haiku-4-5`, $1 / $5 per MTok in/out, 200K ctx) | 4 | 4 (to measure) | ~$0.003–0.004 (arithmetic, §18.6) | 2 alone (needs key + network) | 4 | Strong structured output and citation discipline; no thinking overhead by default |
| Hosted mid: Claude Sonnet 5.5 (`claude-sonnet-5-5`, $2 / $10 per MTok) | 5 | 3–4 (to measure; thinking must be set off or low for latency) | ~2× Haiku | 2 alone | 4 | Quality headroom if Haiku fails the grounding checks |
| Local small via Ollama (1.5–4B instruct, Q4; e.g. Qwen3 1.7B/4B (Apache-2.0, disable thinking), Llama 3.2 3B, Phi-3.5-mini (MIT)) | 2–3 | 1–3 (CPU prefill-bound; to measure) | $0 | 4 (sidecar pulls ~1–3 GB) | 3 | Note: Qwen2.5-**3B** has a non-Apache research license; check every pick |
| Local large (installed qwen3.6 36B, 23 GB) | 5 | 1 on CPU | $0 | 1 | 2 | Unsuitable for a container or judge box |
| **Extractive (no LLM)** | 2 (fluency) / 5 (groundedness) | 5 | $0 | 5 | 5 | Baseline floor and guaranteed G1 path |

**Decision.** One provider interface with three adapters: **hosted (Anthropic SDK), local (Ollama), extractive**.

- **Selection:** auto-select by available configuration (key present → hosted; Ollama reachable → local; otherwise extractive). An explicit `--backend` overrides.
- **Reporting:** every result records which backend produced it.
- **Team decision needed:** the primary demo backend (§19 Q2).

### 12.6 Query decomposition

| Option | Accuracy (expected) | Latency | Reliability | Explainability | LLM calls | Notes |
|---|---|---|---|---|---|---|
| Rules only | 3 (explicit) / 2 (implicit) | 5 | 4 | 5 | 0 | Feeds early retrieval |
| LLM structured only | 4 | 2 | 3–4 | 3 | 1 per utterance (on the critical path) | Late retrieval for intents |
| **Hybrid, confidence-gated** | 4 | 5 (critical path) | 4 | 4 | 0–1 | Rules during the stream; LLM check only when needed |

### 12.7 Streaming mechanism

| Option | Latency | Complexity | Reproducibility | Observability | Notes |
|---|---|---|---|---|---|
| **asyncio in-process event bus + JSONL replay** | 5 | 4 | 5 | 5 | Single process; deterministic replay |
| Threads + queues | 4 | 3 | 4 | 3 | Harder cancellation |
| Broker (Redis Streams / Kafka) | 3 | 1 | 2 | 4 | Infrastructure overhead; violates parsimony |
| Agent / graph framework (LangGraph etc.) | 3 | 2 | 3 | 3 | Abstraction cost; little benefit for a fixed pipeline |

Transport for the demo: **FastAPI + SSE/WebSocket**. Core pipeline: **CLI replay**.

### 12.8 Session-state design

| Option | Isolation | Ephemerality (C4) | Complexity | Supports versioning | Notes |
|---|---|---|---|---|---|
| **In-memory structured store (dict by session_id, TTL, explicit close)** | 5 | 5 | 5 | 5 | Fits the constraint |
| Redis | 4 | 3 (persistence risk) | 3 | 4 | Unneeded infrastructure |
| SQLite | 4 | 2 (persists across runs) | 3 | 4 | Violates the spirit of C4 |
| Chat history only (unstructured) | 3 | 5 | 5 | 1 | Cannot track claims or ledger; G5 is weak |

---

## 13. Candidate Architectures

### Architecture A: Simple hybrid RAG (turn-based)

```
utterance end ─► normalize full utterance ─► hybrid retrieve (BM25⊕dense, RRF) ─► top-k ─► LLM synthesis (+citations) ─► answer
                                                    (late detail: concatenate with previous turn and rerun everything)
```

| Aspect | Assessment |
|---|---|
| Components | Ingestion and index; hybrid retriever; one LLM call; citation label mapping |
| Latency behavior | Nothing happens until the utterance ends, then retrieval (ms) and LLM. TTFT is acceptable, but there is no lead time. |
| Strengths | Simplest; robust; easy to reproduce; good baseline |
| Weaknesses | G2 = 0% by construction. Weak G3: one blended query for N intents. G5 restarts. No suppression. |
| Risks | Fails three of six gates by design |
| Complexity | Low (~1–2 days) |
| Gate coverage | G1 ✓, G4 partial, G6 partial; G2 ✗, G3 ✗, G5 ✗ |

### Architecture B: Streaming hybrid RAG with rule-first controller, query ledger and versioned session state (recommended)

```
chunks ─► Segment tracker ─► Controller (rules; WAIT/RETRIEVE/NO_RETRIEVE) ─► Decomposer (rules; LLM check gated)
              │                     │                                              │
              │                     └──────────── Query Ledger (dedup / reuse / delta) ◄┘
              ▼                                           │
         Session frame/slots                              ▼
                                         Hybrid retrieval (BM25⊕dense RRF; exact numpy)
                                                          ▼
                                    per-intent rerank (committed intents) ─► fusion (quota, dedup, coverage)
                                                          ▼
   utterance end ─► Synthesis (1 LLM call; labeled evidence) ─► citation map + verifier ─► AnswerVersion (v_n, parent, diff)
                                                          ▼
                                    streamed tokens + guide-format record + JSONL telemetry
```

| Aspect | Assessment |
|---|---|
| Components | Everything in A, plus the segment tracker, rule controller, ledger, decomposer with gated LLM check, reranker, fusion, verifier, version store, event bus and telemetry |
| Latency behavior | Retrieval done before the utterance ends in the typical case. The critical path after the end is fusion (ms) plus LLM TTFT. Rerank hidden in speech gaps. |
| Strengths | Covers every gate. Measurable per component (ablations fall out naturally). Cheap: ≤2 LLM calls per turn. Explainable reason codes. |
| Weaknesses | Rules need careful generic design; implicit intents depend on the LLM check; affected-claim tracking is approximate |
| Risks | Rule brittleness on held-out phrasing; G3 judge mismatch; verifier calibration |
| Complexity | Medium (~1.5–3 weeks for a small team, estimate) |
| Gate coverage | G1 ✓ (fallback backends), G2 ✓, G3 ✓, G4 ✓, G5 ✓, G6 ✓ |

### Architecture C: Full streaming agentic RAG

```
chunks ─► LLM controller (per chunk) ─► LLM planner/decomposer ─► tool-calling retrieval agent(s) per intent
          ─► LLM reranker ─► LLM synthesizer ─► LLM critic / self-reflection loop ─► LLM refinement agent (session)
```

| Aspect | Assessment |
|---|---|
| Components | Multiple LLM roles, possibly multi-agent orchestration |
| Latency behavior | An LLM call on every chunk makes retrieval start ≥0.3–2 s after each chunk (estimate). Reflection loops add seconds. TTFT is high. |
| Strengths | Best handling of implicit intents and nuanced refinement; flexible |
| Weaknesses | 5–15 LLM calls per turn; nondeterministic; hard to reproduce offline; cost; explicitly discouraged [G§3 p3, D s7] |
| Risks | G1 (keys, nondeterminism); G2 (timing); parsimony penalty from the jury |
| Complexity | High, with high variance |
| Gate coverage | G3 and G5 potentially strongest; G2 weaker; G1 risky; parsimony criterion ✗ |

### Comparison

| | A | **B** | C |
|---|---|---|---|
| LLM calls per turn | 1 | **1–2** | 5–15 |
| Early retrieval (G2) | ✗ | **✓** | partial |
| Multi-intent (G3) | ✗ | **✓** | ✓ |
| Grounding (G4) | partial | **✓ (by construction + verifier)** | ✓ (if verifier added) |
| Refinement (G5) | ✗ | **✓** | ✓ |
| Telemetry (G6) | partial | **✓** | ✓ (complex) |
| Reproducibility (G1) | ✓ | **✓** | risky |
| Parsimony | ✓ | **✓** | ✗ |
| Role in project | **Baseline B0/B1** | **Target** | Rejected; a single component (the LLM controller) survives as the "model-based controller" ablation |

---

## 14. Proposed Baseline

### 14.1 Baseline B0, the comparison anchor

B0 is a *static, turn-based, dense-only* pipeline:

- Wait for the end of the utterance.
- Use the full normalized utterance as **one** query.
- Run **dense-only** retrieval and keep the top-k (k=5).
- Make one LLM synthesis call with the same label-constrained citation prompt as the target, so the comparison isolates retrieval and streaming effects.
- No controller: always retrieve.
- On late details: concatenate the turns and **re-run everything** (restart).
- No verifier: citations are still mapped from labels so B0 can be scored.

Why dense-only? The guide's own ablation example is "hybrid vs dense-only retrieval" [G§8 p6]. A dense-only B0 makes the first rung of the ladder measure exactly that.

**B0 must be measurable on every metric in §16.** Its G2 is 0% by construction, which is the point of the comparison.

### 14.2 Improvement ladder

Each rung adds **one** change. Everything else stays fixed: chunking, embedder, LLM, prompt, k budget.

| Rung | Adds | Primary metrics expected to move | Experiment |
|---|---|---|---|
| B0 | static, dense, single query, restart | — | — |
| B1 | + BM25 hybrid (RRF) | Recall@k, MRR | Exp 1, 2 |
| B2 | + cross-encoder rerank (committed intents) | Precision@k, nDCG, groundedness | Exp 3 |
| B3 | + decomposition (multi-query, per-intent quota fusion) | Sub-query recall, intent coverage, G3 | Exp 4 |
| B4 | + streaming controller (early retrieval, suppression) | G2 early rate, false-trigger rate, TTFT, lead time | Exp 5, 7, 8 |
| B5 | + session refinement (ledger, delta retrieval, versioning) | G5 checks, re-issue count, latency and tokens on late-detail turns | Exp 6 |
| B6 | + grounding verifier and uncertainty | Unsupported-claim rate, uncertainty recall | Exp 9 |

---

## 15. Experiment Plan

**Shared setup for all experiments:**

- Dev corpus (official, or the fallback from §4.4).
- Dev test set from §17, split into **tune** and **test** halves. Thresholds are calibrated on *tune* only.
- Fixed seeds and temperature 0.
- Real-time replay for latency metrics; virtual time for quality metrics.
- Every run's manifest is logged.

| # | Experiment | Hypothesis | Independent variable | Dependent variables | Data needed | Expected failure modes |
|---|---|---|---|---|---|---|
| 1 | Dense vs BM25 (and the embedder sweep) | BM25 wins on entity- and number-heavy spoken queries; dense wins on paraphrase; the retrieval-trained small embedder beats MiniLM | Retriever ∈ {BM25, dense:MiniLM, dense:bge-small, dense:e5-small(, bge-base)} | Recall@{1,3,5,10}, MRR, query latency p50/p95, index build time | Sub-queries with gold chunk IDs (≥100 intent-level queries); spoken-style phrasing | Gold labels biased toward lexical overlap (annotator wrote queries from chunk text); MiniLM truncation hides long-section relevance |
| 2 | Dense vs hybrid | Hybrid RRF ≥ max(dense, BM25) on Recall@5 across query styles | {dense, hybrid-RRF, hybrid-weighted α∈{0.3,0.5,0.7}} | Recall@k, MRR, latency | Same as Exp 1 | Weighted fusion overfits the tune split; RRF k insensitive |
| 3 | With vs without reranking | Cross-encoder improves Precision@3 and groundedness at an acceptable latency | {none, TinyBERT-L2, MiniLM-L6(, bge-reranker-base)} × candidates {10, 20, 30} | P@3, nDCG@5, Recall@5, rerank latency, end-to-end TTFT, groundedness | Same + full pipeline runs | Domain mismatch (web QA → policy) degrades; truncation of long chunks |
| 4 | Single-query vs multi-query retrieval (4b: decomposition method) | Per-intent retrieval with quota fusion raises intent coverage on compound utterances without hurting single-intent ones | {single full-utterance query, multi-query rules, multi-query LLM, hybrid gated} | Intent coverage@k, sub-query recall/precision, G3 rate, LLM calls, latency | Compound utterances with gold intents + gold chunks per intent (Category B, I) | Over-splitting fixed phrases; LLM inventing intents; single-intent regressions |
| 5 | Static vs early retrieval | The streaming controller achieves ≥80% early rate with a low false-trigger rate and lowers TTFT vs B0 | {retrieve at utterance end, controller rules (thresholds swept)} | G2 early rate, useful-early rate, false-trigger rate, lead time, TTFT, retrieval calls per turn | Multi-chunk streams with timestamps; needs-retrieval labels (Categories C, D, E, H) | Single-chunk turns; aggressive thresholds causing noise; synthetic chunking unlike real ASR cadence |
| 6 | Restart-on-late-detail vs incremental refinement | Refinement preserves v1-valid claims and citations, halves retrieval and tokens on late turns, and keeps the stale-fact rate ≤ restart | {restart (B0 behavior), ledger + delta refinement} | G5 checks, re-issued queries, tokens, latency, retained-citation ratio, stale-fact rate, delta-identification accuracy | Multi-turn sessions with gold constraints and gold v2 claim sets (Categories F, J) | New-topic misclassified as refinement; the LLM rewrites retained claims |
| 7 | Retrieve-always vs retrieval controller | The controller suppresses retrieval on presentation and social turns with ≥95% precision (target) and no recall loss on real queries | {always retrieve, controller} | False-trigger rate, missed-retrieval rate, citation changes on presentation turns, tokens | Category E + H + matched real-query turns | "Summarize X" (topic) vs "summarize that" (answer) confusion; empty-session edge |
| 8 | Rule-based vs model-based controller (required ablation [G§8 p6]) | Rules match or beat the model on timing (G2) at a far lower cost; the model is better on ambiguous suppression | {rules, embedding-prototype classifier, small-LLM classifier} | G2 early rate, false-trigger rate, decision latency, LLM calls, cost | Same as Exp 5 + 7 | Prototype seeds overlapping test phrasing (C2 risk); LLM latency pushing retrieval late |
| 9 | Grounding verifier on/off | The verifier reduces the unsupported-claim rate with a small loss in answer completeness | {no verifier, numbers-only, numbers+overlap, +NLI} | Unsupported-claim rate, citation support, uncertainty precision/recall, latency | Human-annotated claim sample (≥100) | Verifier false positives on paraphrase; NLI CPU cost |
| 10 | Chunking granularity | Section-bounded chunks of ~200–400 tokens maximize Recall@k and citation precision | {whole section, 200, 400 tokens with overlap} | Recall@k, citation correctness, context tokens | Gold labels at section level | Gold labels tied to one granularity |
| 11 | LLM backend comparison | The hosted small model meets TTFT ≤1.5 s; the local CPU model works but is slow; extractive is the most grounded but least fluent | {hosted Haiku-class, local 1.5–4B, extractive} | TTFT, total latency, groundedness, G3 coverage in the answer, tokens, cost | Full pipeline set | Network variance; local model ignoring citation format |

**Mapping to deliverables [G§8 p6]:**

- Baseline comparison: the B0→B6 ladder.
- At least two ablations: Exp 2 (hybrid vs dense) and Exp 8 (rule vs model controller), plus Exp 6 and Exp 7.
- At least three edge-case failure analyses: from Exp 4, 6, 7 and 9 traces.

---

## 16. Metrics

**Notation.** `u` = utterance or turn; `I(u)` = gold intents; `G(i)` = gold relevant chunk IDs for intent *i*; `R_k(q)` = top-k retrieved for query *q*; `t_*` = stream time unless marked wall.

### 16.1 Retrieval

| Metric | Definition / calculation |
|---|---|
| Recall@k (intent-level) | For each gold intent *i* and its mapped sub-query *q*: `|R_k(q) ∩ G(i)| / |G(i)|`; average over intents. If no sub-query maps to *i*, recall = 0. This couples decomposition and retrieval, which is intended for end-to-end runs. Retrieval-only runs (Exp 1–3) use the gold sub-query text. |
| Precision@k | `|R_k(q) ∩ G(i)| / k` |
| MRR | `mean_i 1/rank of first relevant` (0 if none in the top-N) |
| nDCG@k (optional) | With graded relevance (2 = answers it, 1 = related) |
| Intent coverage@k | Share of gold intents with ≥1 gold chunk in the **final fused evidence set** passed to synthesis |
| Retrieval latency | Wall ms from `RetrievalStarted` to `RetrievalCompleted`, per query; report p50/p95 |

### 16.2 Streaming

| Metric | Definition / calculation |
|---|---|
| Time-to-first-retrieval (TTFR) | `t_first RetrievalStarted − t_first ChunkReceived` (stream time) |
| Retrieval lead time | `t_UtteranceEnded − t_first RetrievalStarted`. Positive means early. Report the distribution. |
| Early retrieval rate (G2) | `#eligible u with lead time > 0 / #eligible u`. Eligible = needs retrieval and ≥2 chunks (our definition). |
| Useful early rate | Share of early retrievals whose results include ≥1 chunk cited in the final answer |
| False-trigger rate | `#no-retrieval u with ≥1 RetrievalStarted / #no-retrieval u` |
| Utterance completion time | `t_UtteranceEnded − t_first chunk` (input property; used to normalize) |
| Time-to-first-token (TTFT) | **Wall** ms `FirstToken − UtteranceEnded` in real-time replay. Also report from the first chunk. |
| Answer completion time | Wall ms `AnswerVersionCommitted − UtteranceEnded` |
| Evidence-ready-at-end rate | Share of turns where all committed intents had completed retrieval before `UtteranceEnded` |

### 16.3 Multi-intent

| Metric | Definition / calculation |
|---|---|
| Intent count accuracy | `1[|pred(u)| == |I(u)|]`, averaged over compound utterances |
| Sub-query recall | Matched gold intents / gold intents. Matching = one-to-one assignment maximizing similarity, threshold τ calibrated on tune; a human spot-check sample is reported. |
| Sub-query precision | Matched predictions / predictions (penalizes over-splitting) |
| G3 rate | Share of compound utterances with ≥2 matched distinct gold intents. Also report a strict variant: all gold intents matched and no extras. |
| Duplicate-query rate | Share of issued retrieval queries with cos ≥ τ_dup to an earlier query in the same session |
| Context retention | Share of sub-queries containing the gold shared slots (e.g. location) when the intent needs them |

### 16.4 Grounding

| Metric | Definition / calculation |
|---|---|
| Citation validity | Share of emitted citations whose `Doc_ID §Section` exists in the index. **Must be 100%.** |
| Citation correctness / support | Share of (claim, cited chunk) pairs where the chunk supports the claim. Automatic verifier verdict, plus agreement with a human-labeled sample (report Cohen's κ or percent agreement). |
| Groundedness (G4) | Share of factual claims with ≥1 supporting citation |
| Unsupported-claim rate | `1 − groundedness` (also count claims with no citation at all separately) |
| Uncertainty recall / precision | On intents with no gold evidence (unanswerable): flagged / total; and flagged-and-truly-unanswerable / flagged |

### 16.5 Refinement

| Metric | Definition / calculation |
|---|---|
| Delta identification accuracy | Predicted constraints vs gold constraints (slot-level F1) |
| Turn-type accuracy | REFINE / NEW_TOPIC / PRESENTATION / SOCIAL classification vs gold |
| Re-issue count | Number of refinement-turn retrieval queries matching earlier ledger entries. Target 0. |
| Full re-run flag | True if a refinement turn retrieved for all prior intents again |
| Retained-citation ratio | `|cit(v_{n−1}) ∩ cit(v_n)| / |cit(v_{n−1}) that remain valid per gold|` |
| Stale-fact rate | Claims in v_n that the gold marks invalidated by the new constraint / claims in v_n |
| Session continuity | Share of multi-turn sessions where the version chain is unbroken (`parent` correct) and no state reset occurred |

### 16.6 Efficiency

| Metric | Definition / calculation |
|---|---|
| Stage latency | Wall ms per component (controller, embed, search, rerank, fusion, LLM TTFT, LLM total, verify); p50/p95 |
| Tokens per turn | Input and output tokens from provider usage (hosted) or tokenizer counts (local); sum over LLM calls |
| Retrieval calls per turn | Count of `RetrievalStarted` |
| LLM calls per turn | Count of LLM requests (including cancelled ones; report cancelled separately) |
| CPU time | Process CPU seconds per turn (`time.process_time` deltas) and per stage |
| Memory | Peak RSS of the app container; index size on disk |
| Cost per turn | Σ tokens × price, with prices in a config table (e.g. Haiku 4.5: $1 / $5 per MTok in/out). Local = $0 marginal; optionally report CPU-seconds. |
| Index build time | Seconds from start to ready for the given corpus size |

### 16.7 Observability

| Metric | Definition / calculation |
|---|---|
| Trace coverage (G6) | Turns whose trace passes JSON-Schema validation with all required fields / total turns. Target 100%. |

---

## 17. Test Scenario Taxonomy

**Ground rules.**

- These are **test inputs** written by us. They are not benchmark prompts.
- They follow the domain the guide implies (corporate events and venues, travel reimbursement, catering).
- They assert **no corpus facts**. Expected answers and gold chunk IDs can only be attached once the corpus exists.
- They live **only** under `eval/` and must never be referenced by application code (C2).
- Timestamps are illustrative.
- Format per turn: `[(t, text), …, (t_end, END)]`.

| Cat | Definition | Expected behavior | Representative streaming example | Labels to annotate |
|---|---|---|---|---|
| **A** Single-intent | One clear information need | RETRIEVE (early if multi-chunk), 1 sub-query, cited answer | `(0.0,"what's the cancellation policy for") (0.7,"workshop venues?") (1.1,END)` | needs_retrieval=T; intents=1; gold chunks |
| **B** Multi-intent | ≥2 explicit needs, coordinated | Decompose into N; parallel retrieval; one unified answer | `(0.0,"which venues can host about forty people") (1.0,"and do they provide projectors,") (1.9,"and what's the deposit?") (2.4,END)` | intents=3; shared slots {count=40} |
| **C** Incomplete streaming | Chunks end mid-phrase; fillers; hesitations | WAIT on dangling segments; no retrieval on fragments | `(0.0,"so um I wanted to ask about the") (0.9,"uh the") (1.5,"reimbursement for hotel stays") (2.1,END)` | Expected WAIT ticks at 0.0, 0.9; RETRIEVE ≥1.5 |
| **D** Early retrieval opportunity | Specific need stated early, then a long tail | RETRIEVE well before END; lead time ≥1 s | `(0.0,"for the Pune customer workshop next month") (0.8,"I need the venue capacity details") (1.7,"because we might invite a few more people") (2.6,"from the regional office") (3.4,END)` | Earliest acceptable retrieval t=0.8 |
| **E** Retrieval suppression | Presentation or meta requests; acknowledgments | NO_RETRIEVE; reuse prior citations; no new IDs | `"repeat that as two bullets"`, `"make it shorter"`, `"translate that to Hindi"`, `"thanks, got it"` (each after a prior answer); plus the **empty-session** variant | needs_retrieval=F; reason class |
| **F** Late-arriving detail | Constraint added in a later turn (or late in the same turn) | REFINE: delta queries only; v_n+1 with parent; retained citations | T1 `"summarize the travel reimbursement rule for an employee trip"` → T2 `"the trip was international"` → T3 `"oh and it was booked after the travel"`; same-turn variant: `"what's the venue deposit for thirty people — actually make that forty-five"` | Gold constraints; retained vs modified claims |
| **G** Contradictory or ambiguous | Self-correction; unresolved referent; or corpus conflict | Use the corrected value; ask a clarification for missing critical slots; surface conflicts as uncertainty | `"what's the refund if we cancel — no wait, if we postpone"`; `"what's the limit?"` (no context) → clarify; a conflict case needs a corpus pair | Expected clarification or uncertainty |
| **H** Irrelevant conversational content | Small talk around a query; out-of-corpus questions | Ignore fillers; retrieve only for the real need; out-of-corpus → "not in the knowledge base" without retrieval noise | `(0.0,"hey good morning, hope your week's going ok") (1.2,"so quick one") (1.8,"what's the catering lead time for events?") (2.6,END)`; `"how's the weather in Pune today?"` | needs_retrieval segments only |
| **I** Very long utterance | 15–40 s, 3–5 intents, digressions | Progressive provisional retrieval; no over-splitting; quota fusion keeps all intents | A narrative covering venue, capacity, AV, catering dietary options and cancellation, with digressions | intents=4–5; duplicate-query budget |
| **J** Multiple refinements in one session | ≥3 turns mixing refine, presentation, topic switch and return | Correct turn-type per turn; no contamination after a topic switch; versions chain correctly | T1 venue question → T2 "for 45 people" → T3 "two bullets please" → T4 new topic: travel reimbursement → T5 "back to the venue — does it include parking?" | Turn types; expected version graph |
| **S** Session isolation (extra) | Two concurrent sessions with disjoint topics | No cross-session evidence or claims | Sessions S1 (venues) and S2 (travel), interleaved chunks | Leakage = 0 |

**Size target for the dev suite:** about 60–80 sessions, with ≥8 per category and ≥15 compound utterances (B, I) for a G3 estimate with a usable confidence interval. Split tune/test 50/50 by session.

---

## 18. Recommended Target Architecture

**Recommendation: Architecture B**, the streaming hybrid RAG with a rule-first controller, query ledger and versioned session state.

### 18.1 Rationale

| Basis | Why B |
|---|---|
| Official requirements | Covers all four target capabilities [G§1] and all five pipeline components [G§2] |
| Guide constraints | No web, no hardcoding (generic rules plus corpus-derived signals), session-only state, parsimony (≤2 LLM calls per turn) |
| Evaluation gates | Each gate has a specific mechanism (§10) and an automated measurement |
| Repository reality | Greenfield, so no legacy constraints. B is buildable incrementally as the B0→B6 ladder, so there is always a working system. |
| Corpus characteristics | Corpus-agnostic ingestion with section-aware chunking; exact search is fast at any plausible size |
| CPU feasibility | Measured retrieval costs are in milliseconds; small embedder and cross-encoder on CPU |
| Latency | Retrieval finishes during speech; the post-utterance critical path is fusion plus LLM TTFT |
| Engineering effort | Medium. Every component is standard. The novelty is in orchestration (ledger, controller, versioning), which is also where the judged depth and innovation are. |

### 18.2 Component view

```
                         ┌──────────────────────── Session (in-memory, per session_id, TTL) ─────────────────────────┐
 Replay JSONL / mic ─►   │ Segment Tracker ─► Controller (rules + optional prototype/LLM classifier for ablation)     │
 (timestamped chunks)    │        │                 │ WAIT / RETRIEVE(trigger) / NO_RETRIEVE(reason)                    │
                         │        ▼                 ▼                                                                  │
                         │   Frame & Slots ◄── Decomposer (rules; gated LLM structured check) ──► Query Ledger         │
                         │                                                       (normalize, dedup, reuse, delta)     │
                         │                                   ▼                                                        │
                         │      Retriever: BM25 (scipy sparse) ⊕ Dense (bge-small, numpy exact) → RRF                 │
                         │                                   ▼                                                        │
                         │      Candidate Pool → per-intent Rerank (MiniLM-L6 CE, committed only) → Fusion            │
                         │      (quota, dedup, near-dup, coverage, conflict check) → Evidence Store                   │
                         │                                   ▼  (on utterance end / refinement)                       │
                         │      Synthesizer (LLM adapter: hosted | local | extractive; labeled evidence)              │
                         │                                   ▼                                                        │
                         │      Grounding: label→Doc_ID §Section, ID validation, claim verifier, uncertainty           │
                         │                                   ▼                                                        │
                         │      Answer Versions (v_n, parent, kind, claims, citations, diff)                          │
                         └───────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                             ▼
                     Event Bus ─► Telemetry sink (JSONL traces, per-turn guide-format record, metrics) ─► Eval harness
                                                             ▼
                                         CLI stream / FastAPI SSE demo view (timeline of decisions)
 Offline: Ingestion (txt/md/pdf/docx/html/json → Doc_ID, §Section, chunks) → index cache keyed by corpus hash
```

### 18.3 Component-to-gate traceability

| Component | G1 | G2 | G3 | G4 | G5 | G6 |
|---|---|---|---|---|---|---|
| Ingestion and index cache | ● | | | ● | | |
| Segment tracker and controller | | ● | | | | ● |
| Decomposer and frame/slots | | ● | ● | | ● | |
| Query ledger | | ● | ● | | ● | ● |
| Hybrid retriever, rerank, fusion | | | ● | ● | | |
| Synthesizer and LLM adapters (with extractive fallback) | ● | | | ● | | ● |
| Grounding verifier | | | | ● | | ● |
| Answer versions | | | | | ● | ● |
| Event bus and telemetry | ● | ● | ● | ● | ● | ● |

### 18.4 Latency budget

Status of each number: M = measured on the dev machine, E = estimate, T = target.

| Stage | Budget | Basis |
|---|---|---|
| Controller per chunk | < 5 ms | E (rules, string ops) |
| Query embedding | 3–4 ms (MiniLM, M), ~6–9 ms (bge-small, E) | M/E |
| BM25 + dense search | < 2 ms at ≤100k chunks | M |
| RRF and ledger | < 1 ms | E |
| Rerank (30 candidates, per intent) | ~150 ms | E (from M) |
| LLM decomposition check (gated) | 0.3–1.5 s hosted; overlaps speech | E |
| Fusion and evidence assembly | < 10 ms | E |
| **TTFT after the end of the utterance** | **≤ 1.5 s hosted** (T); local CPU: measure and report | T |
| Verifier (numbers + overlap) | < 20 ms per answer | E |

### 18.5 Draft telemetry and output contract (to be finalized in Phase 2)

Per-turn record. This is a superset of the guide's example [G§4 p4]:

```json
{
  "session_id": "s-…", "turn_id": 1, "turn_type": "query|refinement|presentation|social",
  "retrieval_required": true, "reason": null,
  "retrieval_events": [{"timestamp_s": 0.8, "query": "…", "trigger": "provisional|multi_intent|refinement|final",
                        "intent_ids": ["I1"], "latency_ms": 0.0, "result_ids": ["Doc_x §y"], "ledger_hit": false}],
  "sub_queries": ["…"],
  "answer": "…",
  "citations": ["Doc_x §y"],
  "uncertainty": "…|null",
  "answer_version": {"version": 2, "parent": 1, "kind": "refinement",
                     "diff": {"retained": [], "modified": [], "added": [], "retracted": []}},
  "controller_decisions": [{"timestamp_s": 0.0, "decision": "WAIT", "reasons": ["trailing_function_word"]}],
  "timings": {"utterance_end_s": 2.1, "first_retrieval_s": 0.8, "lead_time_s": 1.3, "ttft_ms": 0, "total_ms": 0},
  "usage": {"llm_calls": 1, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "backend": "hosted|local|extractive"},
  "grounding": {"claims": 0, "supported": 0, "invalid_ids_dropped": 0}
}
```

Event stream: one JSON object per event (§8.6), with `session_id`, `turn_id`, `seq`, `t_stream_s`, `t_wall_ms`, `component`, `event` and `payload`.

### 18.6 Cost arithmetic (illustrative; token counts assumed, not measured)

| Call | Assumed tokens (in / out) | Haiku 4.5 ($1 / $5 per MTok) | Sonnet 5.5 ($2 / $10 per MTok) |
|---|---|---|---|
| Synthesis | 2,000 / 250 | $0.00325 | $0.0065 |
| Decomposition check (when gated on) | 400 / 120 | $0.0010 | $0.0020 |
| **Compound turn total** | | **≈ $0.004** | **≈ $0.0085** |
| Local or extractive | | $0 marginal | $0 marginal |

Prices are taken from the Claude API reference as of 2026-09-25 and must be re-checked before reporting. Other providers' prices are not stated here because they are unverified.

### 18.7 Explicit non-goals for the target

- No vector DB server.
- No agent framework.
- No LLM on the per-chunk path in the main configuration.
- No persistence across sessions.
- No web access.
- No ASR in the core path (optional demo add-on only).

---

## 19. Open Technical Questions

| # | Question | Why it matters | Who resolves |
|---|---|---|---|
| Q1 | **Where is the official Theme 4 corpus?** Format, size, ID scheme? | Blocks chunking, gold labels, quality experiments and realistic tests | Team → organizers (prism@samsung.com [D s14]) |
| Q2 | **Which LLM backend is primary**, and will judges have API keys? Are hosted APIs acceptable at all in evaluation? | G1 path, TTFT, cost reporting | Team, maybe organizers |
| Q3 | What input format does the **private replay harness** use (chunk JSON shape, end marker, session framing) and what output does it parse? | Our I/O must match or adapt. The guide's example JSON is the only evidence. | Organizers; otherwise make I/O adapters configurable |
| Q4 | How is **"eligible query"** defined for G2? Are single-chunk utterances excluded? | The G2 denominator | Organizers; otherwise state our definition |
| Q5 | How are **G3 and G4 judged**: exact strings, LLM judge, or human? | Determines how to phrase sub-queries and what counts as support | Organizers |
| Q6 | **Deadline and phase timeline.** The deck says submission closed 25 Sep 2026 and today is 2 Oct 2026. What is the real deadline (shortlist demo 15 Oct?) and how many people are on the team? | Scope of Phase 2 and what to cut | Team |
| Q7 | Language: English only, or also Hindi or code-mixed speech? | Embedder choice (multilingual) | Organizers / team |
| Q8 | Under C2, are **generic classifier seed phrases and prompt templates** in config acceptable? | Model-based controller design | Team (disclose in the brief); optionally ask organizers |
| Q9 | Do chunks arrive **append-only**, or can ASR revise earlier partials? | Segment tracker design | Organizers; default append-only |
| Q10 | Exact **citation string format**: `Doc_12 §2` vs `[Doc_12 §2]`; section granularity (§2 vs §2.1)? | G4 parsing | Organizers; follow the guide's example |
| Q11 | Judge hardware: CPU cores, RAM, OS, architecture (amd64?), internet available during build and run? | Docker image design, local LLM viability | Organizers / assume worst case |
| Q12 | Should live microphone input be in the demo, or only transcript replay? | Optional scope | Team |

---

## 20. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | No official corpus is available in time | High (currently missing) | Critical | Escalate now (Q1). Build a corpus-agnostic pipeline plus a clearly labeled dev corpus (§4.4). |
| R2 | The deadline has passed or is very close | Unknown (deck says 25 Sep) | Critical | Confirm (Q6). Cut to the B0→B4 core plus refinement first. |
| R3 | The private harness I/O doesn't match ours | Medium | High | Mirror the guide's JSON exactly. Make I/O adapters configurable. Document the formats in the README. |
| R4 | LLM latency on CPU (local mode) is too slow for a convincing demo | High | Medium | Small context budget; hosted option; extractive fallback; report honestly |
| R5 | No API key on judge machines | Medium | High (G1) | Local or extractive fallback: auto-select and never block |
| R6 | Rule-based controller and decomposer are brittle on held-out phrasing | Medium | High (G2, G3) | Generic linguistic rules; corpus-derived signals; gated LLM check; tune/test split; Exp 8 |
| R7 | The automatic grounding verifier disagrees with the judges | Medium | High (G4) | Calibrate against human labels; conservative uncertainty flags; label-constrained citations |
| R8 | Over-engineering: drifting toward Architecture C | Medium | Medium | Ladder discipline: every component must show an ablation gain (C5) |
| R9 | Dependency conflicts (already observed: transformers 5.9 vs torch 2.2.2) | Observed | Medium | Fresh pinned env and lockfile; consider ONNX runtime to avoid torch in the image |
| R10 | arm64 dev vs amd64 judge mismatch; large image size | Medium | Medium (G1) | Build-on-host Dockerfile or multi-arch build; slim base; CPU-only wheels; bake models at build |
| R11 | Index build on first start is slow on judge hardware | Medium | Medium | Cache by corpus hash; progress logging; small embedder; build at image-build time when the corpus is known |
| R12 | Nondeterminism making replay results vary | Medium | Low–Medium | Temperature 0, seeds, pinned models; report variance over repeated runs |
| R13 | C2 interpretation: test utterances or seed phrases seen as "hardcoding" | Low–Medium | High | Keep all test data in `eval/`; generic seeds in config; disclose in the brief |
| R14 | AI-disclosure form is incomplete (we are building with AI assistance) | Medium | Medium (compliance) | Keep an AI-usage log per feature from Phase 2 onward [T-ai] |

---

## 21. Phase 2 Implementation Prerequisites

### 21.1 Decisions needed from the team (blocking)

1. **Corpus** (Q1): an official corpus path and format, *or* approval of the fallback dev corpus approach (public-domain policy documents and/or a labeled synthetic set).
2. **Deadline and team capacity** (Q6). This determines how far up the B0→B6 ladder Phase 2 goes.
3. **Primary LLM backend** (Q2). The recommendation is hosted small (Haiku 4.5 class) for the demo, with local/extractive as the G1 fallback. If hosted: an API key provided via `.env` (never committed).
4. **Approval to download** the model weights and packages Phase 2 needs:
   - `BAAI/bge-small-en-v1.5` (~130 MB)
   - `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB)
   - optionally `intfloat/e5-small-v2`
   - one small Ollama model (~1–3 GB) if local mode is wanted
   - pip packages for a fresh pinned environment

   Sizes are approximate and will be confirmed at download time.

### 21.2 Files and data Phase 2 depends on

| Item | Path / status |
|---|---|
| Theme 4 guide (spec) | `~/Downloads/Theme 4 Guide_RAG.pdf` ✓ |
| Hackathon deck (rules, judging, submission) | `~/Downloads/Samsung PRISM_Y2026_GenAI_Hackathon_3rd_Edition.V2(2).pdf` ✓ |
| Submission PPT template | `~/Downloads/Samsung PRISM Gen AI Hackathon 3.0/CollegeName_TeamName_Submission.pptx` ✓ |
| AI disclosure form | `~/Downloads/Samsung PRISM Gen AI Hackathon 3.0/LangAI3.0_AI_Disclosure.docx` ✓ |
| **Theme 4 corpus** | **Missing.** To be placed at `./corpus/` (proposed). |
| Dev test suite (§17) with gold labels | To be authored after the corpus arrives → `./eval/sessions/*.jsonl` (proposed) |
| Phase 1 micro-benchmarks | `./research/phase1/` ✓ (scripts + `RESULTS.md`) |
| Cached embedders (optional) | `~/.cache/huggingface/hub/` (MiniLM-L6-v2, multilingual-e5-base) ✓ |

### 21.3 Environment setup (first Phase 2 steps; not done yet)

1. `git init`; create the GitHub repo; plan for the release tag `PRISM_GENAI_HACKATHON_Y2026` [D s13].
2. Fresh isolated Python env (3.11 or 3.12) with a lockfile: pip-tools `requirements.lock`, or install `uv`. Do **not** use the global site-packages (§3).
3. Decide on the torch CPU ≥2.4 vs ONNX runtime path for the embedder and cross-encoder (smaller image with ONNX).
4. Proposed repo skeleton, created only when Phase 2 is approved:
   - `src/` with ingestion, index, controller, decomposer, ledger, retrieval, fusion, synthesis, grounding, session, telemetry and api
   - `eval/` with the harness, sessions, gold labels and metrics
   - `configs/` with prompts, thresholds and prices
   - `docker/` and `docs/`
5. Start an AI-usage log for the disclosure form.

### 21.4 Phase 2 entry criteria

- Corpus available (official or approved dev corpus).
- Backend decision made.
- Downloads approved.
- Deadline confirmed.

Then Phase 2 starts with the guide's own Phase 1 roadmap items [G§7 p5]: corpus audit and indexing, the baseline B0/B1 pipeline, and event schemas. That is followed by the controller and stream simulator.
