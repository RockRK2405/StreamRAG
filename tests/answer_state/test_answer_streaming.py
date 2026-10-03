"""Streaming answers inside the session (brief §35-40, §38, §56) and deterministic replay with recorded LLM output.
TEST FIXTURE corpus; scripted generator."""

from dataclasses import replace

import pytest

from grounding_helpers import echo, grounding_stack, requires_nli, scripted
from session_helpers import of, stream_turns
from streamrag.replay import ReplayEngine

pytestmark = requires_nli

CONTRACT = ["ANSWER_STARTED", "ANSWER_SECTION_STARTED", "ANSWER_CLAIM_READY", "ANSWER_CITATION_READY",
            "ANSWER_SECTION_COMPLETED", "ANSWER_VALIDATED", "ANSWER_COMPLETED"]
OBSERVABILITY = ["CLAIM_PLAN_CREATED", "ANSWER_GENERATION_STARTED", "ANSWER_GENERATION_COMPLETED", "LLM_CALL",
                 "CLAIMS_EXTRACTED", "CLAIM_VERIFICATION_STARTED", "CLAIM_VERIFIED", "CITATION_CREATED",
                 "CITATION_VALIDATED", "ANSWER_REVISED", "ANSWER_FINALIZED"]
TURNS = [["How are applications", "for the fixture permit", "submitted?"], ["Specifically", "online."]]


@pytest.fixture(scope="module")
def llm_stack(tmp_path_factory):
    st = grounding_stack(tmp_path_factory)
    st._intent_stack = replace(st.intent_stack, grounding=st.grounding.with_backend(scripted(echo)))
    return st


@pytest.fixture(scope="module")
def run(llm_stack):
    return stream_turns(llm_stack, TURNS)


def test_drafts_precede_the_validated_final(run):
    val = [e for e in of(run.events, "ANSWER_VALIDATED") if e.utterance_id == "u1"]
    statuses = [e.payload["status"] for e in val]
    assert statuses[-1] == "VALIDATED_FINAL" and "DRAFT" in statuses[:-1]
    final = of(run.events, "ANSWER_FINALIZED")
    assert [e.payload["status"] for e in final] == ["VALIDATED_FINAL"] * len(final)
    llm = [e for e in of(run.events, "LLM_CALL") if e.utterance_id == "u1"]
    assert llm and all(e.t_session_ms >= val[-2].t_session_ms for e in llm)   # drafts are extractive, no LLM call


def test_event_contract_and_ids(run):
    types = {e.type.value for e in run.events}
    assert set(CONTRACT) <= types and set(OBSERVABILITY) <= types
    for e in run.events:
        if e.type.value in CONTRACT + ["CITATION_CREATED", "ANSWER_FINALIZED"]:
            assert e.session_id and e.payload["answer_id"].startswith("GA") and e.payload["version"] >= 1
        if e.type.value in ("ANSWER_CLAIM_READY", "ANSWER_CITATION_READY"):
            assert e.payload["claim_id"].startswith("AC-") and e.intent_id
        if e.type.value == "ANSWER_CITATION_READY":
            assert e.payload["citation_id"] and e.payload["key"]


def test_turn_payload_carries_the_grounded_answer(run):
    turns = {e.utterance_id: e.payload for e in of(run.events, "TURN_COMPLETED")}
    a1, a2 = turns["u1"]["answer"], turns["u2"]["answer"]
    assert a1["status"] == "VALIDATED_FINAL" and a1["changed"] and "[fixture_permit_handbook §2]" in a1["text"]
    assert a2["answer_id"] != a1["answer_id"] and a2["version"] > a1["version"]
    ready = [e.payload["claim_id"] for e in of(run.events, "ANSWER_CLAIM_READY")
             if e.payload["answer_id"] == a1["answer_id"]]
    assert ready == a1["claims"]


def test_virtual_replay_is_exact_with_recorded_llm_output(llm_stack, run):
    rep = ReplayEngine(llm_stack.cfg, llm_stack.service, llm_stack.policy, llm_stack.index_hash,
                       llm_stack.intent_stack).replay(run.events)
    assert rep.identical, rep.summary()


def test_needs_streamed_one_chunk_at_a_time_share_the_utterance_answer(llm_stack):
    """'eligibility requirements | and the application process' share no topic word; streamed chunk by chunk, the second
    need used to open its own frame, so the final answer covered only the first (found in the Phase 7 e2e run)."""
    r = stream_turns(llm_stack, [["Tell me the eligibility requirements", "and the application process",
                                  "for the fixture permit."]])
    final = [e.payload for e in of(r.events, "ANSWER_VALIDATED") if e.payload["status"] == "VALIDATED_FINAL"][-1]
    turn = of(r.events, "TURN_COMPLETED")[-1].payload
    assert len(final["coverage"]["covered"]) == 2
    assert "permit portal" in turn["answer"]["text"] and "18 years" in turn["answer"]["text"]


def test_console_printer_shows_drafts_and_the_final(run):
    from streamrag.streaming.printer import format_event
    lines = [x for x in (format_event(e) for e in run.events) if x and " GROUNDED " in x]
    assert any(" DRAFT" in x for x in lines)
    final = [x for x in lines if "VALIDATED_FINAL" in x]
    assert final and "[fixture_permit_handbook §2]" in final[0]
