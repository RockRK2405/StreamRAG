"""Shared components used from several worker threads at once (Phase 8 runtime)."""

from concurrent.futures import ThreadPoolExecutor

from streamrag.retrieval.text import Analyzer


def test_analyzer_is_thread_safe():
    """A shared Snowball stemmer raised IndexError when the runtime analysed text on several threads at once."""
    a = Analyzer()
    text = ("requirements applications submitted officers reviewing permitted harvesting crates " * 40).strip()
    ref = a.tokens(text)
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda _: a.tokens(text), range(300)))
    assert all(o == ref for o in outs)



def test_split_retrieval_stages_reproduce_the_phase3_result(fixture_bundle):
    """The runtime runs lexical and dense as separate subtasks and assembles them: the result must be exactly the
    Phase 3 ``retrieve()`` result (same evidence, ranks, scores, status)."""
    from streamrag.models.retrieval import RetrievalOptions, RetrievalRequest
    from streamrag.retrieval.embedders import HashingEmbedder
    from streamrag.retrieval.service import RetrievalService
    cfg, bundle = fixture_bundle
    svc = RetrievalService(bundle, cfg, embedder=HashingEmbedder(bundle.analyzer))
    for q in ["How high should the wicks be trimmed?", "crates per shift at harvest", "ladders overnight"]:
        req = RetrievalRequest(query=q, options=RetrievalOptions(mode="hybrid", top_k=5))
        ref = svc.retrieve(req)
        plan = svc.plan(req)
        got = svc.assemble(plan, svc.search_lexical(plan), svc.search_dense(plan, inline=True))
        strip = lambda es: [(e.evidence_id, e.rank, e.bm25_rank, e.dense_rank, round(e.rrf_score, 9))  # noqa: E731
                            for e in es.items]
        assert strip(got) == strip(ref) and got.evidence_set_id == ref.evidence_set_id
        assert got.trace.status == ref.trace.status and got.trace.warnings == ref.trace.warnings


def test_failed_commit_rolls_back_to_the_last_valid_state():
    from streamrag.runtime.state import StateCoordinator
    from streamrag.runtime.tasks import TaskResult, TaskStatus, TaskType
    state = {"x": 1}
    sc = StateCoordinator("s")
    events = []
    sc.emit = lambda t, c, p, u=None: events.append((t.value, p))
    sc.checkpoint, sc.restore = (lambda: dict(state)), (lambda snap: (state.clear(), state.update(snap)))
    r = TaskResult("T1", TaskType.GENERATION, "s", 0, 0, None, TaskStatus.COMPLETED, "ok")

    def bad_apply():
        state["x"] = 99
        raise RuntimeError("half-applied")
    assert not sc.commit(r, bad_apply, transactional=True)
    assert state == {"x": 1} and sc.version == 0 and sc.rollbacks == 1
    assert events[-1][0] == "ERROR" and events[-1][1]["action"] == "rolled_back_to_last_valid_state"
    assert sc.commit(r, lambda: state.update(x=2), relevant=lambda: True) and state["x"] == 2 and sc.version == 1
    stale = TaskResult("T2", TaskType.DRAFT, "s", 0, 0, None, TaskStatus.COMPLETED, "late")
    sc.new_epoch()
    assert not sc.commit(stale, lambda: state.update(x=3)) and state["x"] == 2
    assert events[-1][0] == "STALE_RESULT_DISCARDED" and events[-1][1]["reason"] == "epoch_changed"
