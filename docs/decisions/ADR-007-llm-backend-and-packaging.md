# ADR-007: LLM Backend Strategy and Packaging

- **Status:** Accepted (primary backend pending team decision Q2)
- **Date:** 2026-10-02
- **Spec:** §18.11, §18.15, §23, §25.2; correction **K2**

## Context

- G1 requires a **container** to launch with one command on a clean machine, and the automated replay to complete without manual intervention [G§5 p4]. Phase 1 had treated a CLI runner as equivalent (K2).
- Judges may have no API key and no GPU.
- TTFT and cost per turn must be reported [D s7].
- Locally installed Ollama has only a 23 GB model, which is unsuitable for a container.

## Decision

1. **One LLM Gateway interface with three adapters, in fallback order:**

   | Adapter | Notes |
   |---|---|
   | Hosted | Anthropic SDK, e.g. `claude-haiku-4-5` at $1 / $5 per MTok in/out; prices kept in a config table |
   | Local | Ollama with a small instruct model, behind an optional compose profile |
   | Extractive | No LLM; always available |

   Backend auto-selection: key present → hosted; Ollama reachable → local; otherwise extractive. The backend used is recorded in every `LLM_CALL` and manifest.
2. **`docker compose up` is the primary path.** The CLI replay is a development convenience.
3. Model weights are baked at image build. Dependencies are pinned by lockfile. Images build for amd64 and arm64.
4. Temperature 0, at most 1 retry, explicit TTFT and total timeouts.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Hosted only | G1 fails without a key |
| Local only | CPU prefill makes TTFT slow [E]; weaker instruction following |
| Large local model | Unsuitable for a container |
| OpenAI-compatible shim for every provider | For Claude we use the official SDK |

## Why selected

It guarantees G1 completion on any machine, keeps quality options open for the demo, and makes cost measurable.

## Trade-offs

- Three code paths to test. Mitigated by a shared interface and contract tests.
- Results differ by backend, so they must be reported per backend.

## Consequences

- The team must choose the primary demo backend and supply a key via `.env` (Q2).
- The keyless CI run is mandatory (REQ-REPRO-003).
