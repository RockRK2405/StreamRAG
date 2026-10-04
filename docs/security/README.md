# Security and privacy (Phase 11 audit)

## 1. Security checklist (brief §19, §42)

| check | status | how |
|---|---|---|
| no secrets committed | ✅ | repository scan for API-key, token and password patterns: none found (`FINAL_PROJECT_AUDIT.md` §3). The system needs no secrets. |
| `.env` excluded | ✅ | `.gitignore` excludes `.env`. `.dockerignore` lets only code, configs, demo data and the two models into the image. |
| `.env.example` exists | ✅ | placeholders only |
| retrieved documents treated as untrusted | ✅ | quoted, delimited data in prompts. Instruction-like sentences are excluded from facts. Every generated claim is verified against the evidence. Citations are rebuilt from verification, never copied from the model. The UI renders document text with `textContent` only. Tests: `tests/answer_state::test_prompt_injection_in_corpus_cannot_become_a_fact`, robustness `malformed_query`. |
| prompt-injection mitigations | ✅ (heuristic plus verification) | `claims/textcheck.py` injection markers; JSON-schema-bound output; claim verification. Limitation: the marker list is heuristic. Verification is the guarantee that injected *facts* cannot reach the answer, not that the model ignores every instruction. |
| bounded loops | ✅ | adaptive retrieval: at most 3 iterations and 5 queries per need (`adaptive_retrieval.budget`). Answer revision: at most `max_answer_revision_attempts`. Multi-hop: bounded hops. Per-need and per-utterance query budgets. |
| bounded retrieval | ✅ | top-k caps, result budget (40) and latency budget (1.5 s) per need, cancellation of superseded searches |
| bounded token usage | ✅ | `generation.max_output_tokens` 1024, `num_ctx` 8192, evidence quoted to at most 1,500 characters, at most 6 claims per section |
| resource exhaustion | ✅ | HTTP: 16 KB headers, 64 KB bodies, 10 s read timeout. At most 8 sessions with idle eviction. Bounded input queues and event subscribers. Chunk length capped at 4,000 characters. Turn deadline 30 s. |
| safe error messages | ✅ | the HTTP layer returns generic messages (`internal error`). Exceptions are logged by class name only, and transcripts are never logged. |
| unsafe tool execution | n/a | the system executes no tools or code from model output |
| network egress | ✅ | the only network call is to the LLM endpoint, which must be loopback unless explicitly allowlisted (`generation.allowed_llm_hosts`; tests in `tests/test_config.py`). Retrieval, embedding and verification are in process. |
| web UI | ✅ | Content-Security-Policy `default-src 'self'` with no inline script, `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, same-origin API, no cookies |

## 2. Privacy

| data | where it is processed | leaves the device? |
|---|---|---|
| transcript chunks (user speech or text) | the StreamRAG process: intent analysis, query building, retrieval | **no**, in the default and documented deployments |
| retrieval queries and evidence | in process (BM25 and dense index, local files) | no |
| prompts (question needs + evidence facts) | local LLM (Ollama on the same machine or a private compose network) | no. A remote LLM would need an explicit allowlist entry and would then receive needs and evidence facts. |
| session state (needs, evidence, answer versions) | memory only, per session; destroyed when the session closes or after 15 min idle | no; not persisted |
| logs | stderr, structured | contain request metadata only, no transcript or answer text |
| evaluation traces | written only by the evaluation runners, on fixture data | n/a |

* **Not claimed.** No encryption at rest is needed: nothing user-related is stored. There is no authentication: the
  demo server binds to `127.0.0.1` by default.
* **Before exposing it on a network,** add authentication and TLS through a reverse proxy. This is listed in
  `LIMITATIONS.md`.

## 3. Edge vs cloud (design; on-device execution NOT MEASURED)

This is a design analysis, backed by measured footprints. No Samsung device, SDK or Samsung model was available in
this environment, so nothing here claims Samsung hardware execution.

```
              DEVICE / EDGE                                     SERVER (private)
  mic -> ASR (platform) -> transcript chunks
  stream ingestion, retrieve/wait/skip decision,      ->   adaptive retrieval over the large corpus
  intent decomposition, session state                      (BM25 + dense), LLM generation,
  (rule-based, ~ms, no model)                               claim verification (NLI), citations
  local cache of validated claims                    <-    streamed verified answer + citations
```

| component | measured footprint (`FINAL_BENCHMARK_RESULTS/resources`) | edge suitability |
|---|---|---|
| controller, intent decomposition, session state | rule-based, no model; in-process CPU per runtime turn about 0.13–0.16 s (Phase 10, all stages) | good: no model, privacy-preserving (the raw transcript can stay local) |
| bge-small embedder (ONNX) | 128 MB on disk; query embedding about 3 ms wall on M5 Pro | plausible on a flagship phone (not measured on one) |
| NLI verifier (deberta-v3-xsmall) | 279 MB; about 22 ms per claim against 5 sections (wall, median) | plausible on a server; on device NOT MEASURED |
| LLM qwen3:4b (Q4_K_M) | 3.9 GB resident (Metal); generation about 1.6–1.8 s per answer (p50) | server or a high-end device; on device NOT MEASURED |
| int8 variants (MiniLM embedder, cross-encoder) | Phase 3: less memory (758 vs 1,053 MB peak), no latency gain; the int8 reranker was slower | no measured benefit here, so not adopted |

The split is justified by the measurements:
* the parts that see the raw transcript continuously are cheap and model-free, so they can stay on the device;
* the parts that need the corpus and the LLM are the expensive ones.

Running the LLM on device would need a smaller model. None was available or measured, so on-device generation is
**NOT MEASURED** (`LIMITATIONS.md`).
