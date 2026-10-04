# Architecture

The authoritative description is `../docs/architecture/14_final_architecture.md`: the frozen pipeline, the system
diagram with module paths, cross-cutting concerns and the Phase 11 changes. Component design documents:

| layer | documents |
|---|---|
| retrieval foundation (BM25, dense, hybrid, RRF, chunking, citations) | `../docs/retrieval/01_corpus_pipeline.md` … `08_benchmarking.md`, ADR-001…003 |
| streaming engine and retrieval controller | `../docs/streaming/`, ADR-004, ADR-014 |
| multi-intent decomposition and fusion | `../docs/multi_intent/`, ADR-015 |
| session memory, delta retrieval, evidence lifecycle | `../docs/session/`, `../docs/memory/README.md`, ADR-016 |
| grounded generation, claim verification, citations | `../docs/answer/`, ADR-017 |
| asynchronous runtime (scheduling, cancellation, backpressure, recovery, replay) | `../docs/runtime/`, ADR-018 |
| adaptive retrieval (analysis, claim requirements, routing, stopping, caches) | `../docs/retrieval/01_query_analysis.md` … `11_budget_management.md`, ADR-019 |
| deployment, API, health checks | `../docs/deployment/README.md` |
| security, privacy, edge vs cloud | `../docs/security/README.md` |
