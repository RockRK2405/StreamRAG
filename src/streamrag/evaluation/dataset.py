"""Evaluation dataset contract, difficulty rules, loading and leakage checks (Phase 10; docs/evaluation/dataset.md).

One ``EvalSample`` = one scored user turn. Turns of the same conversation share ``session_id`` and are run in
``turn_index`` order; ``conversation_context`` lists the earlier utterances of the session (what the system heard
before this turn). ``stream`` optionally gives the chunks as they arrive (an item ``{"text", "replaces"}`` is an ASR
revision of an earlier chunk); batch systems receive ``query`` (the final transcript of the turn).

Labels (gold evidence, expected claims / answer, forbidden values, expected evidence state) are written by the
implementer from the fixture corpus text; they are evaluation-side only: nothing under ``src/streamrag`` except this
package reads them (tests/evaluation/test_leakage.py).

Difficulty is computed, never assigned by hand (``difficulty()``):
  points = (needs - 1) + 2 * multi-hop + [>= 1 constraint] + [>= 2 constraints] + temporal + context-dependent
           + conflicting evidence + insufficient evidence + streaming revision + lexical gap
  lexical gap = < 34 % of the query's content terms occur in its gold evidence (analyzer terms; computed at build time)
  EASY = 0, MEDIUM = 1, HARD = 2-3, VERY_HARD >= 4
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract

QueryType = Literal["SIMPLE", "SEMANTIC", "EXACT_TERM", "MULTI_INTENT", "MULTI_CONSTRAINT", "TEMPORAL",
                    "CONTEXTUAL_FOLLOWUP", "ENTITY_CORRECTION", "MULTI_HOP", "AMBIGUOUS", "CONTRADICTORY",
                    "INSUFFICIENT_EVIDENCE", "REPEATED_QUERY", "STREAMING_CORRECTION"]
QUERY_TYPES: tuple[str, ...] = ("SIMPLE", "SEMANTIC", "EXACT_TERM", "MULTI_INTENT", "MULTI_CONSTRAINT", "TEMPORAL",
                                "CONTEXTUAL_FOLLOWUP", "ENTITY_CORRECTION", "MULTI_HOP", "AMBIGUOUS", "CONTRADICTORY",
                                "INSUFFICIENT_EVIDENCE", "REPEATED_QUERY", "STREAMING_CORRECTION")
Difficulty = Literal["EASY", "MEDIUM", "HARD", "VERY_HARD"]


class ExpectedClaim(Contract):
    text: str                                 # reference statement (from the corpus)
    citation: str | None = None               # section that states it ("DOC §n")
    key: list[str] = []                       # strings an answer stating this claim must contain (all of them)


class StreamChunk(Contract):
    text: str
    replaces: int | None = None               # ASR revision of chunk n (0-based)


class EvalSample(Contract):
    sample_id: str
    split: Literal["dev", "test"]
    source: str                               # where the item comes from (dataset / phase)
    corpus: str                               # corpus key (experiments/configs/corpora.yaml)
    session_id: str
    turn_index: int = Field(ge=1)
    query: str
    stream: list[StreamChunk] = []
    conversation_context: list[str] = []
    query_type: QueryType
    expected_intents: list[str] = []
    required_entities: list[str] = []
    required_constraints: list[str] = []
    ground_truth_evidence: list[str] = []     # section citations
    gold_semantics: Literal["all", "any"] = "all"   # "any": several sections can each answer (ambiguous queries)
    expected_claims: list[ExpectedClaim] = []
    expected_answer: str = ""
    expected_state: Literal["SUFFICIENT", "INSUFFICIENT", "CONTRADICTORY"] = "SUFFICIENT"
    conflict_values: list[str] = []           # every one must be reported when the conflict cannot be resolved
    forbidden: list[str] = []                 # regexes an answer must not match (stale / unsupported values)
    source_documents: list[str] = []
    difficulty: Difficulty = "EASY"
    difficulty_features: dict[str, int] = {}
    labeler: str = "implementer"


def difficulty(s: EvalSample, lexical_gap: bool) -> tuple[str, dict[str, int]]:
    f = {"extra_needs": max(0, len(s.expected_intents) - 1),
         "multi_hop": 2 if s.query_type == "MULTI_HOP" or "multi_hop" in s.source else 0,
         "constraint": 1 if s.required_constraints else 0,
         "multi_constraint": 1 if len(s.required_constraints) >= 2 else 0,
         "temporal": 1 if s.query_type == "TEMPORAL" else 0,
         "context": 1 if s.conversation_context and s.query_type in ("CONTEXTUAL_FOLLOWUP", "ENTITY_CORRECTION",
                                                                      "REPEATED_QUERY") else 0,
         "conflict": 1 if s.query_type == "CONTRADICTORY" else 0,
         "insufficient": 1 if s.expected_state == "INSUFFICIENT" else 0,
         "streaming_revision": 1 if any(c.replaces is not None for c in s.stream) else 0,
         "lexical_gap": 1 if lexical_gap else 0}
    pts = sum(f.values())
    return ("EASY" if pts == 0 else "MEDIUM" if pts == 1 else "HARD" if pts <= 3 else "VERY_HARD"), {**f, "points": pts}


def load(path: Path) -> list[EvalSample]:
    return [EvalSample.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sessions(samples: list[EvalSample]) -> list[list[EvalSample]]:
    """Samples grouped by session, turns in order (sessions in first-appearance order)."""
    out: dict[str, list[EvalSample]] = {}
    for s in samples:
        out.setdefault(s.session_id, []).append(s)
    return [sorted(v, key=lambda x: x.turn_index) for v in out.values()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def manifest(paths: dict[str, Path], extra: dict | None = None) -> dict:
    out = {"version": "streamrag_eval_v1", "files": {}}
    for split, p in paths.items():
        rows = load(p)
        out["files"][split] = {"path": str(p.name), "sha256_16": file_hash(p), "samples": len(rows),
                               "sessions": len({r.session_id for r in rows}),
                               "by_type": _count(r.query_type for r in rows),
                               "by_difficulty": _count(r.difficulty for r in rows),
                               "by_corpus": _count(r.corpus for r in rows)}
    return {**out, **(extra or {})}


def _count(xs) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return dict(sorted(d.items()))


def dumps(samples: list[EvalSample]) -> str:
    return "".join(json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n" for s in samples)
