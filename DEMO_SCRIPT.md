# Demo script (6 minutes)

**Setup, before judging.**

```bash
ollama serve
```
```bash
.venv/bin/streamrag serve
```

* Open http://127.0.0.1:8080.
* Check the header: **Ready**, **LLM: qwen3:4b (local)**, **Demo data**.
* Rehearse with the automated check, which must pass:

```bash
.venv/bin/streamrag demo-check
```

**Fallbacks.**
* If the LLM is unavailable: `streamrag serve --no-llm`, or just carry on. The header switches to "verified extractive
  answers", and every scenario still runs (10 / 10 in `demo-check --no-llm`).
* If the laptop fails: `docker compose up` on any machine with Docker. Same UI, offline, no keys.

The demo corpus is fictional (Riverbank University scholarship rules). Say so once at the start.

| time | what to say | what to show |
|---|---|---|
| 0:00 | **Problem.** People ask while they speak: several questions at once, details that arrive late, corrections. The answer has to come from trusted documents, with sources. | the empty UI and the scenario list |
| 0:30 | **Conventional RAG** waits for the end, runs one query and answers once. Silence first, then one blurred query; a correction means starting over. | (talking) |
| 1:00 | **Our solution.** StreamRAG retrieves while you talk, splits the request into needs, verifies every statement against its source and updates the answer instead of restarting. | the architecture slide (`docs/architecture/14_final_architecture.md` §14.3) |
| 1:30 | **Live streaming query.** Click **A. Normal question**. | the transcript grows word by word; stage chips move Query → Retrieval → Evidence before the question finishes; the verified answer with its citation; the timing line (first evidence / first answer / verified answer) |
| 2:00 | **Adaptive retrieval.** Click **C1** (form code), then **C2** (one applicant group). | C1: the plan chip "Fast path: keyword lookup", one keyword search, no embedding. C2: "Filtered search (applicant type …)". Not every question pays for the expensive pipeline. |
| 2:30 | **Multi-intent.** Click **B**. | three need chips; one answer with a section per need, each with citations |
| 3:00 | **Late correction.** Click **D**. | turn 1 answers for all students; turn 2 "Sorry, I mean for international students" shows *New condition added*, *Dropped an outdated search*, a filtered plan and an updated answer citing the international policy |
| 3:30 | **Delta retrieval.** Same turn: only the changed need was searched again, and evidence that is still valid was kept. Click **D2** for a speech-recognition revision mid-sentence. | the activity panel (searches dropped and re-run); D2 answers for the *revised* word |
| 4:00 | **Evidence and citations.** Click a citation chip. | the source panel shows the exact section text; "every statement is checked against this text before it is marked Verified" |
| 4:30 | **Contradiction and uncertainty.** Click **F1**, then **F3**. | F1: both notices reported (25 August vs 1 September) instead of picking one. F3: "The retrieved documents do not contain an answer to 'cover health insurance'" instead of guessing |
| 5:00 | **Evaluation results.** Held-out set written before the final changes; baselines; what improved and what did not. | `FINAL_PRESENTATION.md` slides 12–13 (numbers from `FINAL_BENCHMARK_RESULTS/README.md`) |
| 5:30 | **Samsung relevance.** Voice-first products; a device front end that is cheap and model-free, with the heavy stages on a server; privacy (nothing leaves the machine); measured cost per stage. No Samsung hardware or SDK is claimed. | the edge / cloud table (`docs/security/README.md` §3) |
| 6:00 | **Impact.** Answers that start while you speak, stay correct when you change your mind, and show their sources. | the final answer with citations |

**If something goes wrong live.**
* Click **New conversation** and re-run the scenario. Scenarios are deterministic: temperature 0 and fixed seeds.
* Draft text can differ slightly between runs. The *verified* answer and its citations are what to point at.
