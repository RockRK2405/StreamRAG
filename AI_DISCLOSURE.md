# AI Usage Disclosure (draft for the official form)

> **For the team.** Fill sections 1 and 6, check every line of this draft, and copy it into the official
> `LangAI3.0_AI_Disclosure.docx`. The signed form is the authoritative declaration, not this file.

## 1. Team details

* **Team name:** _to be filled by the team_
* **Project / product name:** StreamRAG: Streaming Live RAG (Samsung PRISM GenAI Hackathon 2026, Theme 4)
* **Organization / institution:** _to be filled by the team_
* **Submission date:** 4 October 2026

## 2. AI usage declaration

**Yes.** AI was used in developing this project.

## 3. Purpose of AI usage

| purpose | used | details |
|---|---|---|
| idea generation / brainstorming | yes | Claude Code (Anthropic, Claude models) produced the Phase 1 research dossier and the Phase 2 specification from the team's phase briefs |
| code generation or assistance | yes | Claude Code wrote the implementation, configuration, Docker packaging and tests, following the team's phase-by-phase briefs |
| UI / UX design | yes | Claude Code wrote the demo web UI (`src/streamrag/server/static/`) |
| content creation | yes | Claude Code wrote the documentation, reports, presentation outline, demo script and judge Q&A |
| data analysis | yes | Claude Code wrote the evaluation framework and analysis scripts. Every reported number comes from stored runs of the system, not from an AI model. |
| testing / debugging | yes | Claude Code wrote and ran the tests and diagnosed failures |
| other: AI inside the product | yes | at run time the system uses a local LLM (`qwen3:4b` via Ollama) for answer wording, an NLI model (`nli-deberta-v3-xsmall`) for claim verification, and an embedding model (`bge-small-en-v1.5`) for retrieval. All run locally. |

## 4. Feature origin classification

Every feature was specified by the team in phase briefs. Implementation and documentation were generated with Claude
Code and reviewed by the team.

| feature | origin | description |
|---|---|---|
| streaming retrieval controller (retrieve / wait / skip) | Both | Team: requirements and phase brief. Claude Code: design, rule-based implementation, tests, measurements (Phase 4). |
| multi-intent decomposition and evidence fusion | Both | Same split (Phase 5). |
| session memory, delta retrieval, evidence lifecycle | Both | Same split (Phase 6). |
| grounded generation, claim verification, citations | Both | Same split (Phase 7). Runtime models: local `qwen3:4b`, NLI verifier. |
| asynchronous streaming runtime (cancellation, backpressure, recovery) | Both | Same split (Phase 8). |
| adaptive retrieval | Both | Same split (Phase 9). |
| evaluation framework, datasets, benchmark | Both | Same split (Phases 10–11). The fixture corpora and labels are fictional. They were AI-written under the team's direction and are labelled TEST FIXTURE ONLY. |
| demo server, UI, Docker, documentation | Both | Same split (Phase 11). |

## 5. Ethical and compliance confirmation

* AI usage complies with the hackathon guidelines and policies: _team to confirm_.
* No proprietary or copyrighted data was misused. All corpora in this repository are fictional fixtures written for
  testing. No Samsung data, model, SDK or hardware was used.

## 6. Declaration and sign-off

_Name, role, signature and date: to be completed by the team representative on the official form._
