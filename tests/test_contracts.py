"""M1 contracts: validation, (de)serialization round-trips, deterministic representation, schema drift."""

import json

import pytest
from pydantic import ValidationError

from streamrag.models import (
    SCHEMA_MODELS,
    AnswerVersion,
    BenchmarkCase,
    Citation,
    Claim,
    Evidence,
    EvidenceSet,
    Intent,
    IntentQuery,
    IntentSet,
    IntentSetDelta,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    SessionEnd,
    SessionStart,
    TelemetryEvent,
    TranscriptChunk,
    UtteranceEnd,
)
from streamrag.models.schemas import render_schemas

from conftest import REPO

SPAN = {"utterance_id": "u1", "start": 0, "end": 5, "text": "topic"}
INTENT = Intent(intent_id="I1", session_id="s-1", utterance_id="u1", text="topic", resolved_text="topic", confidence=0.9,
                provenance={"source": "rule", "split": "single", "span": SPAN},
                components=[{"text": "topic", "source": "intent", "span": SPAN}])
TRACE = RetrievalTrace(request_id="rq-1", mode="hybrid", status="ok", index_version="3.0", corpus_hash="c" * 64,
                       index_config_hash="i" * 64)
EVIDENCE = Evidence(evidence_id="D§1#1", document_id="D", section_id="1", chunk_id="D§1#1", citation="D §1",
                    text="t", source_path="d.txt", char_start=0, char_end=1, rank=1, score=0.5, retrieval_method="hybrid_rrf")

EXAMPLES = [
    SessionStart(session_id="s-1"),
    TranscriptChunk(session_id="s-1", utterance_id="u1", payload={"chunk_index": 0, "timestamp_s": 0.0, "text": "hi"}),
    UtteranceEnd(session_id="s-1", utterance_id="u1", payload={"timestamp_s": 2.1, "last_chunk_index": 2}),
    SessionEnd(session_id="s-1"),
    TelemetryEvent(event_id="s-1:000003", type="RETRIEVAL_STARTED", session_id="s-1", seq=3, t_wall_ms=1.5,
                   component="retrieval", payload={"query": "q"}),
    INTENT,
    IntentSet(session_id="s-1", utterance_id="u1", version=1, source="rule", original_text="topic", intents=[INTENT],
              global_constraints=[{"constraint_id": "K1", "kind": "restriction", "text": "for z", "scope": "global",
                                   "applies_to": ["I1"], "source_span": SPAN}]),
    IntentSetDelta(utterance_id="u1", version=2, previous_version=1, added=["I2"], affected_intents=["I2"]),
    IntentQuery(intent_id="I1", intent_version=1, utterance_id="u1", text="topic",
                components=[{"text": "topic", "source": "intent", "span": SPAN}]),
    EVIDENCE,
    EvidenceSet(evidence_set_id="es-1", query="q", items=[EVIDENCE], trace=TRACE),
    Claim(claim_id="C1", text="A claim.", introduced_in=1, evidence_ids=["D§1#1"], intent_id="I1", intent_version=1,
          source={"evidence_id": "D§1#1", "char_start": 0, "char_end": 8}, status="SUPPORTED", confidence=0.5),
    Citation(key="D §1", evidence_ids=["D§1#1"], document_id="D", section_id="1"),
    AnswerVersion(version=2, parent_version=1, topic_id="T1", utterance_id="u2", kind="refinement", text="x"),
    RetrievalRequest(query="  padded  "),
    RetrievalResult(retrieval_id="r1", status="ok", latency_ms={"total": 1.0}, n_candidates=0),
]


@pytest.mark.parametrize("obj", EXAMPLES, ids=lambda o: type(o).__name__)
def test_roundtrip_and_determinism(obj):
    data = obj.canonical_json()
    again = type(obj).from_json(data)
    assert again == obj
    assert again.canonical_json() == data                      # stable representation
    assert list(json.loads(data)) == sorted(json.loads(data))  # sorted keys


def test_request_query_is_stripped():
    assert RetrievalRequest(query="  q  ").query == "q"


@pytest.mark.parametrize("bad", [
    lambda: TranscriptChunk(session_id="bad id!", utterance_id="u", payload={"chunk_index": 0, "timestamp_s": 0}),
    lambda: TranscriptChunk(session_id="s", utterance_id="u", payload={"chunk_index": -1, "timestamp_s": 0}),
    lambda: TranscriptChunk(session_id="s", utterance_id="u", payload={"chunk_index": 1, "timestamp_s": 0,
                                                                       "replaces_chunk_index": 2}),
    lambda: TranscriptChunk(session_id="s", utterance_id="u", payload={"chunk_index": 0, "timestamp_s": 0}, extra=1),
    lambda: TelemetryEvent(event_id="random-uuid", type="ERROR", session_id="s", seq=1, t_wall_ms=0, component="x"),
    lambda: AnswerVersion(version=1, parent_version=1, topic_id="T", utterance_id="u", kind="initial", text="x"),
    lambda: Citation(key="no-section-sign", evidence_ids=["e"], document_id="D", section_id="1"),
    lambda: Evidence(**{**EVIDENCE.model_dump(), "rank": 0}),
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValidationError):
        bad()


def test_benchmark_case_requires_corpus_evidence_for_answerable_intents():
    case = {"case_id": "c1", "category": "single_intent", "split": "tune", "is_fixture": True,
            "session": {"session_id": "s", "turns": [{"utterance_id": "u1", "utterance_text": "q", "utterance_end_s": 1.0,
                        "expected": {"turn_type": "query", "retrieval_required": True,
                                     "intents": [{"gold_intent_id": "g1", "description": "q", "answerable": True}]}}]}}
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate(case)
    case["session"]["turns"][0]["expected"]["intents"][0]["gold_evidence"] = ["D §1"]
    items = BenchmarkCase.model_validate(case).retrieval_items()
    assert len(items) == 1 and items[0].is_fixture and items[0].gold == ["D §1"]


def test_committed_schemas_match_models():
    """docs/schemas must be regenerated (`streamrag export-schemas`) whenever a contract changes."""
    from streamrag.models.schemas import all_schema_models
    rendered = render_schemas()
    assert set(all_schema_models()) == {n.replace(".schema.json", "") for n in rendered}
    for name, text in rendered.items():
        committed = REPO / "docs" / "schemas" / name
        assert committed.exists(), f"missing {committed}"
        assert committed.read_text() == text, f"schema drift in {name}"
