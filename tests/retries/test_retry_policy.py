"""Retries (brief §36-38, §73): transient vs deterministic failures, bounded backoff, idempotent retries."""

import urllib.error

import pytest

from conftest import REPO
from grounding_helpers import requires_nli
from runtime_helpers import commits, of, rt_stack, virtual
from streamrag.config import load_config
from streamrag.generation.llm import LLMResponse
from streamrag.runtime import Fault, FaultInjector
from streamrag.runtime.cancellation import CancellationManager
from streamrag.runtime.retry import PermanentError, RetryManager, TransientError, is_transient
from streamrag.runtime.runtime import RetryingLLM
from streamrag.runtime.scheduler import TaskScheduler, VirtualRunner
from streamrag.runtime.tasks import Task, TaskStatus, TaskType
from streamrag.streaming.clock import VirtualScheduler

RCFG = load_config(REPO / "configs" / "default.yaml", base_dir=REPO).runtime


def test_classification_and_bounded_backoff():
    assert all(is_transient(x) for x in (TimeoutError(), ConnectionError(), TransientError("x"),
                                         urllib.error.URLError("refused"), "HTTP 503 Service Unavailable",
                                         "rate limit exceeded (429)"))
    assert not any(is_transient(x) for x in (ValueError("bad"), PermanentError("x"), "schema_invalid: ...", None))
    a, b = RetryManager(RCFG.retry), RetryManager(RCFG.retry)
    d1 = [a.delay_ms(i) for i in range(8)]
    assert d1 == [b.delay_ms(i) for i in range(8)]                 # seeded jitter: replayable
    assert all(x <= RCFG.retry.max_delay_ms * (1 + RCFG.retry.jitter) for x in d1)
    assert d1[0] < d1[3] and RCFG.retry.initial_delay_ms * 0.8 <= d1[0] <= RCFG.retry.initial_delay_ms * 1.2


def _sched():
    clock = VirtualScheduler()
    return TaskScheduler(RCFG, clock, VirtualRunner(clock, RCFG.sim_latency_ms), CancellationManager(),
                         RetryManager(RCFG.retry)), clock


@pytest.mark.parametrize("errors,expected,attempts", [
    ([TransientError("blip"), TransientError("blip")], TaskStatus.COMPLETED, 3),       # transient -> retried
    ([TransientError("down")] * 5, TaskStatus.FAILED, 3),                             # persistent -> bounded
    ([ValueError("deterministic")], TaskStatus.FAILED, 1),                             # never retried
])
def test_scheduler_retries(errors, expected, attempts):
    sched, clock = _sched()
    errs = list(errors)

    def fn(w):
        if errs:
            raise errs.pop(0)
        return "ok"
    got = []
    sched.submit(Task(sched.new_task_id(), TaskType.LEXICAL, "s", 1, 0, 0, 0.0, meta={"sim_ms": 5}), fn, got.append)
    clock.run()
    assert got[0].status == expected and got[0].metadata["attempts"] == attempts


def test_retrying_llm_retries_only_transient_errors():
    class Flaky:
        name, model = "flaky", "m"

        def __init__(self, errors):
            self.errors, self.calls = list(errors), 0

        def complete(self, messages, schema=None):
            self.calls += 1
            if self.errors:
                return LLMResponse(ok=False, backend="flaky", model="m", error=self.errors.pop(0))
            return LLMResponse(ok=True, text="{}", backend="flaky", model="m")
    f = Flaky(["ConnectionError: reset by peer"])
    r = RetryingLLM(f, RetryManager(RCFG.retry), virtual=True).complete([])
    assert r.ok and f.calls == 2 and r.meta["attempts"] == 2
    f = Flaky(["ValueError: bad request"])
    r = RetryingLLM(f, RetryManager(RCFG.retry), virtual=True).complete([])
    assert not r.ok and f.calls == 1


@requires_nli
def test_retried_retrieval_does_not_duplicate_results(tmp_path_factory):
    st = rt_stack(tmp_path_factory, "corpus")
    faults = FaultInjector([Fault("network", "transient", times=2)])
    rt, evs = virtual(st, [["How high should the wicks be trimmed?"]], faults=faults)
    retried = of(evs, "TASK_RETRIED")
    assert len(retried) == 2 and rt.retry.stats["retried"] == 2
    partial_ids = [(e.query_id, e.payload["kind"]) for e in of(evs, "RETRIEVAL_PARTIAL")]
    assert len(partial_ids) == len(set(partial_ids))              # one result per subtask despite retries
    done = of(evs, "RETRIEVAL_COMPLETED")
    assert len(done) == len({e.query_id for e in done}) and all(e.payload["status"] == "ok" for e in done)
    assert "4 millimetres" in commits(evs)[-1].payload["text"]
