# Deployment

Every command below was run on the development machine: Apple M5 Pro, macOS 26.5.1, Docker 29.8.1. Results are in
`FINAL_BENCHMARK_RESULTS/`.

## 1. Quickstart, local (Python 3.11 or newer, CPU only)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps
```
```bash
.venv/bin/streamrag fetch-models bge-small-en-v1.5 nli-deberta-v3-xsmall
```
```bash
ollama serve
```
```bash
ollama pull qwen3:4b
```
```bash
.venv/bin/streamrag serve
```

* **Open the demo** at http://127.0.0.1:8080.
* **Without Ollama:** `streamrag serve --no-llm`, or just don't start Ollama. Answers then come from the verified
  extractive generator, and `/ready` reports `"degraded": true`.
* **Automated demo:** `streamrag demo-check` runs all ten demo scenarios headlessly and writes
  `runs/demo_check.json`. The exit code is 1 if any scenario fails.

## 2. Docker (the one-command path, spec K2 / REQ-DEP-001)

```bash
docker compose up --build
```

* **What the image is.** One image (1.46 GB, `linux/arm64` here) holding the frozen final pipeline, the demo UI, the
  two models and a prebuilt demo index.
* **Weights are baked in at build time.** They are copied from `./models` when present, otherwise downloaded and
  checked against the sha256 values in `configs/models.yaml`.
* **Offline and keyless.** The default compose run has no LLM: it serves verified extractive answers with no API keys.
  Health check: `/ready`.

**Optional LLM, either way:**

| option | command |
|---|---|
| LLM in a compose sidecar | `docker compose --profile llm up --build`, then once: `docker compose exec ollama ollama pull qwen3:4b` |
| Ollama on the host (Docker Desktop) | `STREAMRAG_LLM_URL=http://host.docker.internal:11434 STREAMRAG_ALLOWED_LLM_HOSTS=host.docker.internal docker compose up` |

On a Mac the host option is faster, because host Ollama uses the GPU through Metal.

**Automated demo inside the container:**

```bash
docker compose exec app streamrag demo-check
```

**Verified results:**
* healthy 8 s after start;
* `demo-check` 10 / 10 without an LLM;
* 3 / 3 checked scenarios with the host LLM, `/ready` reporting `"llm": {"ok": true}`.

**Not verified:** an amd64 build. Only arm64 was built here; the Dockerfile has no architecture-specific steps.

## 3. Configuration

* **Precedence:** `configs/default.yaml` < `configs/profiles/final.yaml` (the frozen pipeline) < `STREAMRAG_*`
  environment variables < explicit `--set key=value` overrides.
* **Variables:** `STREAMRAG_CORPUS`, `STREAMRAG_INDEX_ROOT`, `STREAMRAG_MODELS_DIR`, `STREAMRAG_LLM_BACKEND`
  (auto / ollama / extractive), `STREAMRAG_LLM_URL`, `STREAMRAG_LLM_MODEL`, `STREAMRAG_ALLOWED_LLM_HOSTS`,
  `STREAMRAG_LOG_LEVEL`, `STREAMRAG_HOST`, `STREAMRAG_PORT`, `STREAMRAG_ROOT`. Each is documented in `.env.example`.
* **No secrets.** None are needed or read.

## 4. Health and readiness

| endpoint | meaning |
|---|---|
| `GET /health` | liveness: the process serves requests (always 200 while it runs) |
| `GET /ready` | readiness. Checks the index (chunks loaded), the embedder, the verifier (NLI model), the runtime and the LLM (re-probed every 10 s). Returns **200** when retrieval, verification and the runtime are up, even without an LLM (then `"degraded": true` and extractive answers). Returns **503** when startup failed, with the error. |

## 5. Logging and tracing

* **Structured logs.** One JSON object per line on stderr (`telemetry/logging.py`). Request logs carry the method,
  the path (session ids masked), the status and the duration. They never carry transcripts, answers or document text.
* **Per-session event log.** Every event has `session_id`, `utterance_id`, a `correlation_id` (one per turn),
  `causation_id`, `parent_event_id` and `state_version`. A turn can be followed through query → intent → retrieval
  plan → retrieval → evidence → generation → verification → citation → response.
* **Replay.** `streamrag replay TRACE.jsonl` re-runs a saved trace.
* **Evaluation traces.** The evaluation stores traces per session under `*/runs/traces/`.

## 6. Failure behaviour (each case is tested or measured)

| failure | behaviour | evidence |
|---|---|---|
| LLM unreachable at start | extractive answers; `/ready` degraded | `tests/server`, `demo-check --no-llm` |
| LLM errors mid-session | per-answer extractive fallback (`GENERATION_DEGRADED`) | robustness `llm_error` |
| LLM hangs (timeout) | extractive redo with its own deadline (fixed in Phase 11) | robustness `llm_timeout`, `tests/timeouts` |
| dense index / embedder fails | lexical-only retrieval | robustness `vector_db_failure` |
| all retrieval fails | explicit "not established" answer, nothing invented | robustness `retrieval_failure` |
| entailment model fails | rules-only verification (`VALIDATION_DEGRADED`) | robustness `verifier_failure` |
| malformed or oversized input | rejected with 400 / 413, or bounded by `runtime.max_chunk_chars` | `tests/server`, robustness `malformed_query` |
| client disconnects mid-stream | session stays bounded; idle sessions evicted, at most 8 | `tests/server` |
