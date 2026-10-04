# Final Problem Statement: Streaming Live RAG (Samsung PRISM Theme 4)

## What problem are we solving?

A person talking to an assistant does not type a finished query. They speak, and the system receives a growing
transcript in pieces, with:
- several questions in one breath;
- details that arrive late ("…for international students");
- corrections ("sorry, I mean…").

The assistant should answer from a trusted document corpus. Every statement in the answer must be backed by a
citation, and the answer should start before the user finishes and improve as they keep talking.

Theme 4 asks for exactly this pipeline (official guide, `PHASE_1_RESEARCH_DOSSIER.md` §2):
1. retrieve while the user is still talking;
2. split the request into its real questions;
3. fuse the evidence into one grounded, cited answer;
4. refine that answer when new details arrive, without starting over.

## Why does conventional RAG struggle?

Conventional RAG is request → retrieve → generate:
- it waits for the full turn, then sends it as one query;
- it answers once.

That costs several seconds of silence after the user stops. A multi-part question becomes one blurred query. A late
detail means throwing the state away and running everything again. It also has no notion of what was already
retrieved, verified or said.

## Why does streaming make this harder?

The input is unstable.
- A partial transcript can mean something else than the finished sentence: "What did a single ride cost" before "in
  2025?" arrives.
- Speech recognisers revise earlier words.
- Retrieving on every word wastes work; waiting for the end loses the head start.

The system has to decide continuously whether to retrieve, wait or skip. It has to cancel work that became obsolete,
and it must never let an answer built on a partial transcript survive once the full sentence says otherwise.
Phase 10 measured exactly this failure ("early commitment"), and Phase 11 fixed most of it (see
`FINAL_BENCHMARK_RESULTS/README.md`).

## Why are late-arriving details difficult?

A detail such as "for international students" changes which documents apply:
- some evidence stays valid ("applications close on 30 April");
- some becomes wrong (the domestic requirements);
- some new evidence is needed (the international requirements).

Restarting from scratch is simple but wasteful and loses context. Keeping everything risks stating facts about the
wrong group. The system needs a lifecycle per need and per piece of evidence: kept, re-validated or dropped. Then
only the changed need is retrieved again.

## Why is repeated retrieval inefficient?

In a conversation the same needs recur:
- follow-ups ("and how long does it take?");
- repeats ("what was the fee again?");
- refinements of a need already answered.

Re-running retrieval, embeddings and LLM calls for unchanged needs adds latency and compute with no benefit. Phase 10
measured savings from delta retrieval and session reuse (fewer retrieval calls per conversation turn) and also their
risk, stale reuse. Both are reported.

## Why is grounding important?

The answers are about rules, fees, deadlines and eligibility, so a wrong number is worse than no answer. Every claim
must be:
1. traceable to a specific section (`Doc_ID §Section`, as the guide requires);
2. checked against that section before it is shown as verified;
3. replaced by an explicit "not in the sources" when no evidence supports it.

Conflicting or superseded documents must be reported as such, not silently merged.

## Why is adaptive retrieval necessary?

Questions differ widely:
- "What is Form ISO-7 for?" needs one keyword lookup;
- a question about one applicant group needs a filtered search;
- a two-document question needs a second hop.

Running the most expensive pipeline for everything wastes latency and compute. Running the cheapest for everything
misses evidence. Phase 10 and 11 measured both sides:
- adaptive retrieval ranks better and hands far less noise to the answer stage, with fewer embeddings;
- on a tiny corpus it can lose recall against simply taking the top 5.

## Why is this relevant to Samsung?

This section describes design relevance only. **No Samsung device, SDK, model or service was used or integrated in
this prototype**, and none is claimed.

- **Voice-first products.** Samsung ships voice-driven assistants and AI features across phones, wearables, TVs and
  home appliances. These are exactly the settings where users speak incrementally and correct themselves, and where
  answers about manuals, warranties, settings or service policies must be correct and sourced.
- **Hybrid device and cloud execution.** The pipeline separates a cheap, model-free front end from the expensive
  corpus and LLM stages:
  - front end: stream ingestion, retrieve/wait decisions, intent decomposition, session state;
  - expensive stages: retrieval over a large corpus, generation, verification.

  That split fits a device-plus-server deployment, where the raw transcript can stay on the device
  (`docs/security/README.md` §3). The measured footprints support the split. On-device execution itself is NOT
  MEASURED.
- **Cost and latency discipline.** The guide penalises heavy orchestration. StreamRAG measures every component's cost
  (retrieval calls, embeddings, LLM calls, tokens) and keeps only what earns its place in the ablations.
- **Privacy.** No transcript, query or answer leaves the machine in the documented deployments. The LLM is local, and
  network egress is limited to an explicitly allowlisted LLM endpoint.
