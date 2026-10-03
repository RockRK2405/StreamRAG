"""Timeouts and deadline propagation (brief §24-26)."""

import pytest

from conftest import REPO
from grounding_helpers import requires_nli
from runtime_helpers import commits, of, realtime, rt_stack, virtual, with_runtime
from streamrag.config import load_config
from streamrag.runtime import Fault, FaultInjector, SimulatedLLM
from streamrag.runtime.tasks import TaskType
from streamrag.runtime.timeouts import TimeoutManager


def test_deadline_propagation_reserves_downstream_time():
    cfg = load_config(REPO / "configs" / "default.yaml", {"runtime.budget.turn_ms": 6000,
                                                          "runtime.budget.generation_reserve_ms": 4000,
                                                          "runtime.budget.validation_reserve_ms": 500}, base_dir=REPO)
    tm = TimeoutManager(cfg.runtime)
    turn = tm.turn_deadline(1000.0)                               # 7000
    d = tm.deadline_for(TaskType.DENSE, 1200.0, turn)
    assert d.limited_by == "turn_budget" and d.at_ms == 7000 - 4500  # retrieval may not eat generation's time
    g = tm.deadline_for(TaskType.GENERATION, 1200.0, turn)
    assert g.at_ms == 6500 and g.limited_by == "turn_budget"
    own = tm.deadline_for(TaskType.LEXICAL, 1200.0, None)
    assert own.limited_by == "task_timeout" and own.at_ms == 1200 + cfg.runtime.timeouts_ms.lexical
    late = tm.deadline_for(TaskType.DENSE, 9000.0, turn)          # budget already spent: fail fast, never negative
    assert late.at_ms == 9001.0


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return rt_stack(tmp_path_factory, "corpus")


@requires_nli
def test_retrieval_timeout_degrades_to_lexical(stack):
    faults = FaultInjector([Fault("dense", "timeout", times=-1)])
    rt, evs = virtual(stack, [["How high should the wicks be trimmed?"]], faults=faults)
    assert of(evs, "TASK_TIMED_OUT") and all(e.payload["task_type"] == "dense" for e in of(evs, "TASK_TIMED_OUT"))
    done = of(evs, "RETRIEVAL_COMPLETED")
    assert done and all(e.payload["status"] == "degraded" for e in done)
    assert of(evs, "DEGRADED_MODE_CHANGED")[0].payload["mode"] == "RETRIEVAL_DEGRADED"
    assert "4 millimetres" in commits(evs)[-1].payload["text"]   # still answered from lexical evidence


@requires_nli
def test_generation_timeout_falls_back_to_extractive(stack):
    st = with_runtime(stack, **{"timeouts_ms.generation": 400})
    rt, evs = realtime(st, [["How high should the wicks be trimmed?"]], llm=SimulatedLLM(latency_ms=3000))
    to = [e for e in of(evs, "TASK_TIMED_OUT") if e.payload["task_type"] == "generation"]
    assert to
    modes = [e.payload["mode"] for e in of(evs, "DEGRADED_MODE_CHANGED")]
    assert "GENERATION_DEGRADED" in modes
    final = commits(evs)[-1]
    assert final.payload["status"] == "VALIDATED_FINAL" and final.payload["mode"] == "extractive"
    assert rt.scheduler.zombies() == 0                            # the timed-out worker returned and freed its slot
