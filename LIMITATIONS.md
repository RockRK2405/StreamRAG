# Limitations

Every limitation below is measured or directly observed. Numbers are from the held-out v2 benchmark
(`FINAL_BENCHMARK_RESULTS/README.md`, 81 turns) unless stated otherwise. Nothing here is hidden elsewhere.

## 1. Evidence base

* **No official corpus.**
  * The Theme 4 corpus was never available. Every result comes from fictional fixture corpora with labels written by
    the implementer.
  * All numbers are **TEST FIXTURE ONLY — NOT REPORTABLE** as official results.
* **Small samples.**
  * v2 has 81 turns. Per-category n is 4–15, and the answerability and conflict cases have n = 2–4.
  * Most paired differences in answer correctness are not statistically significant.
  * There is no multiple-comparison correction, so p-values are descriptive.
* **No human evaluation.** A blinded sheet exists (`experiments/datasets/human_eval_sheet.csv`), but no human rated
  anything.
* **Biased verifier metrics.**
  * Claim support and citation precision are judged with the system's own NLI verifier, so for the verified systems
    they are 1.000 by construction (Phase 10 §23).
  * The model-free hallucination metric only catches values (numbers, dates, names) that appear nowhere in the
    evidence or the question.

## 2. Quality

* **No answer-correctness gain over the baselines on v2.**

  | system | answer correct |
  |---|---|
  | full system | 0.718 |
  | naive RAG | 0.761 |
  | hybrid + rerank | 0.803 |

  Neither difference is significant (McNemar p = 0.68 and p = 0.24). The measured gains are elsewhere:
  * no stale or forbidden values (0 vs 0.154–0.538);
  * no hallucinated values (0 vs 0.062–0.093);
  * about 2.2 times the evidence precision;
  * evidence and first answer content about 0.26–0.27 s after the first word.
* **Lower recall from adaptive retrieval.**
  * Recall@5 is 0.883 for the full system and 0.935 for the batch adaptive pipeline, vs 0.994 for naive top-5.
  * On a corpus of 14 short documents, top-5 nearly always contains the answer. The adaptive pipeline hands fewer,
    more precise sections to the answer stage, and sometimes misses one.
  * Multi-constraint questions are the weakest category: 0.375 vs 1.000 for naive RAG.
* **Session refinement does not transfer well to a new domain.** This is guide gate G5.
  * On v2 follow-up, correction and repeat turns, the streaming system answers 0.611 correctly vs 0.722 in batch mode
    (n = 18).
  * Observed causes:
    * Elliptical follow-ups are rewritten with the wrong slot. "And in the south?" is still searched as "…in Service
      Area North".
    * Entity corrections lose the original question: "Sorry, I meant Fernhill, not Oakridge" is searched as
      "Fernhill".
    * In one follow-up, the previous turn's claims were kept instead of answering the new need.
  * These rules were developed on other domains. v2 is held out, so they were not tuned on it.
* **Insufficient evidence is often not recognised.**
  * Only 1 of 4 unanswerable v2 questions gets an explicit "not in the sources". The other three state an adjacent,
    true fact.
  * The optional LLM answerability flag raised this to 2 of 4, but cost answer correctness: 0.662 vs 0.718 on v2, and
    the same direction on the development data. It is therefore off by default.
* **Multi-hop questions remain hard.**
  * Answer correctness on v2 multi-hop questions is 0.571 (hybrid + rerank: 0.857).
  * The bridge-entity analysis rarely fires on unseen domains.
* **The cross-encoder reranker is off.**
  * It was switched off from Phase 3–10 measurements: it cost about 30 ms wall and 280 ms CPU per call, with no gain
    on the development data.
  * On v2 the reranked hybrid baseline has the best answer correctness. Re-tuning on v2 would break the held-out
    protocol, so this is left as a finding, not a change.

## 3. Streaming and latency

* **More work per turn.**
  * Retrieving while the user speaks costs 1.86 retrieval calls per turn vs 1.00 for naive RAG.
  * It also costs 1.27 LLM calls per turn vs 1.00, for drafts and revisions.
* **The verified answer is not faster after the user stops.**

  | system | p50 time to verified answer after the utterance ends |
  |---|---|
  | full system | 1,643 ms |
  | naive RAG | 1,411 ms |
  | batch mode of the same pipeline | 1,659 ms |

  The difference with naive RAG is not significant. Streaming wins on time to first evidence and first answer
  content, not on the final verified answer.
* **Early commitment is reduced, not eliminated.**
  * The Phase 11 fixes removed stale values on v2 (0 forbidden values).
  * The streaming system still has lower recall than batch mode (0.883 vs 0.935).
* **Latency depends on the machine and the local LLM.**
  * All measurements are on one Apple M5 Pro laptop with `qwen3:4b` via Ollama.
  * During the first v1 regression run, the LLM server's own per-request time doubled for about ten minutes for
    reasons outside the system, and the end-to-end latency doubled with it (`FINAL_BENCHMARK_RESULTS/README.md` §5).
  * No real-time guarantee is claimed. Multi-user throughput and load were not measured. The demo server is capped at
    8 concurrent sessions.
* **Simulated speech input.**
  * Transcript chunks are replayed word by word with timestamps.
  * ASR revisions are scripted.
  * No real ASR, audio or speech-recognition errors were tested.

## 4. Generality

* **Rule-based language analysis.**
  * Retrieve / wait / skip decisions, intent decomposition and session rules are English-only and rule-based.
  * Corpus-derived anchors adapt them partly to a new corpus. Phrasing outside the lexicons degrades them (see §2).
* **Local LLM.** `qwen3:4b` follows the JSON schema well, but sometimes states an adjacent fact instead of the asked
  value. Verification catches the unsupported cases. It cannot catch true-but-irrelevant ones.
* **Scale.** The largest index tested is a small fixture corpus. Behaviour on a corpus of thousands of documents
  (recall, latency, memory) is NOT MEASURED.

## 5. Deployment and security

* **Not hardened for network exposure.**
  * There is no authentication, TLS or rate limiting per client.
  * The server binds to `127.0.0.1` by default, and the Docker image binds `0.0.0.0` inside the container network.
  * Before exposing it, put it behind an authenticating TLS proxy.
* **Heuristic prompt-injection handling.**
  * Injection markers are a heuristic list.
  * The guarantee is narrower: verification stops injected *facts* from reaching the answer. It does not guarantee
    that the model ignores every instruction.
* **In-memory sessions.** Sessions are lost on restart, and there is no persistence or horizontal scaling.
* **Image size.** The Docker image is 1.46 GB. The LLM is not in the image: it needs the optional Ollama service and
  a separate model download, about 2.5 GB.

## 6. Samsung relevance

* No Samsung device, SDK, model or service was used. The relevance is a design argument (`FINAL_PROBLEM_STATEMENT.md`).
* On-device execution of any component is **NOT MEASURED**. The edge / cloud split in `docs/security/README.md` §3
  rests on footprints measured on a laptop.
* The int8 model variants gave no measured benefit here (Phase 3), so they were not adopted.
