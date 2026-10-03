"""Phase 8 event model: structured envelope, correlation, causation, trace tree, ordering (brief §4-7, §34, §50)."""

import pytest

from grounding_helpers import requires_nli
from runtime_helpers import of, rt_stack, virtual
from streamrag.models.events import EventType as E
from streamrag.runtime.events import RuntimeEventBus, path_to_root, trace_tree, untraceable

ELIG_PROC = ["Tell me the eligibility requirements", "and the application process", "for the fixture permit."]


def test_dispatch_context_sets_causation_and_parents():
    t = [0.0]
    bus = RuntimeEventBus("s", lambda: t[0], version_fn=lambda: 3)
    root = bus.emit(E.SESSION_STARTED, "x", {})
    with bus.dispatch():
        a = bus.emit(E.CHUNK_RECEIVED, "x", {}, "u1")
        b = bus.emit(E.TRANSCRIPT_UPDATED, "x", {}, "u1")
    with bus.dispatch(cause=a.event_id):
        c = bus.emit(E.INTENT_DETECTED, "x", {}, "u1", intent_id="I1")
        d = bus.emit(E.QUERY_GENERATED, "x", {"query_id": "Q1"}, "u1", intent_id="I1", query_id="Q1")
    assert a.causation_id is None and b.causation_id == a.event_id and c.causation_id == a.event_id
    assert a.parent_event_id == root.event_id                     # utterance hangs below the session root
    assert d.parent_event_id == c.event_id                        # query below its intent
    assert {e.correlation_id for e in (a, b, c, d)} == {"s/u1"} and all(e.state_version == 3 for e in (a, d))
    assert [x.type for x in path_to_root([root, a, b, c, d], d.event_id)] == [E.QUERY_GENERATED, E.INTENT_DETECTED,
                                                                                E.CHUNK_RECEIVED, E.SESSION_STARTED]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return virtual(rt_stack(tmp_path_factory), [ELIG_PROC])


@requires_nli
def test_every_event_is_traceable(run):
    _, evs = run
    assert evs and untraceable(evs) == []
    assert all(e.correlation_id and e.state_version is not None for e in evs)
    assert all(e.event_id == f"s1:{i:06d}" for i, e in enumerate(evs))


@requires_nli
def test_retrieval_is_caused_by_its_query_and_task(run):
    _, evs = run
    by = {e.event_id: e for e in evs}
    started = of(evs, "RETRIEVAL_STARTED")[0]
    task_started = by[started.causation_id]
    scheduled = by[task_started.causation_id]
    assert task_started.type == E.TASK_STARTED and scheduled.type == E.TASK_SCHEDULED
    assert by[scheduled.causation_id].type in (E.QUERY_GENERATED, E.QUERY_UPDATED)
    assert by[scheduled.causation_id].query_id == started.query_id
    tree = trace_tree(evs)
    assert scheduled.event_id in tree and started.event_id in tree[scheduled.event_id]


@requires_nli
def test_answer_events_hang_below_their_generation_task(run):
    _, evs = run
    by = {e.event_id: e for e in evs}
    commit = [e for e in of(evs, "ANSWER_COMMITTED")][-1]
    chain = [x.type for x in path_to_root(evs, commit.event_id)]
    assert E.TASK_SCHEDULED in chain and chain[-1] == E.SESSION_STARTED
    assert by[commit.causation_id].type == E.TASK_COMPLETED
    assert by[commit.causation_id].payload["task_type"] == "answer_extractive"   # no LLM in this run


@requires_nli
def test_user_visible_stream_is_ordered_and_contiguous(run):
    _, evs = run
    vis = [e for e in evs if e.output_seq is not None]
    assert [e.output_seq for e in vis] == list(range(len(vis)))
    assert all(e.output_seq is None for e in of(evs, "TASK_STARTED", "RETRIEVAL_PARTIAL", "CLAIM_VERIFIED"))
    vers = [e.state_version for e in evs]
    assert vers == sorted(vers)                                    # state versions never go backwards
    types = [e.type.value for e in vis]
    first_final = types.index("ANSWER_COMMITTED")
    assert "TURN_COMPLETED" in types[:first_final] and types[-1] == "SESSION_CLOSED"
