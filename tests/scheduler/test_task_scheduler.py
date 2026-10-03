"""TaskScheduler unit tests on the virtual clock (brief §8-12, §16, §38): bounded pools, priority with aging,
queue capacity / shedding, idempotency, deadlines."""

from conftest import REPO
from streamrag.config import load_config
from streamrag.runtime.cancellation import CancellationManager
from streamrag.runtime.retry import RetryManager
from streamrag.runtime.scheduler import TaskScheduler, VirtualRunner
from streamrag.runtime.tasks import Task, TaskStatus, TaskType
from streamrag.streaming.clock import VirtualScheduler


def make(**rt):
    cfg = load_config(REPO / "configs" / "default.yaml", {f"runtime.{k}": v for k, v in rt.items()}, base_dir=REPO)
    clock = VirtualScheduler()
    sched = TaskScheduler(cfg.runtime, clock, VirtualRunner(clock, cfg.runtime.sim_latency_ms), CancellationManager(),
                          RetryManager(cfg.runtime.retry))
    return sched, clock


def task(sched, ttype=TaskType.LEXICAL, prio=1, key=None, deadline=None, sim_ms=10.0, sid="s"):
    return Task(sched.new_task_id(), ttype, sid, prio, 0, 0, sched.clock.now_ms(), deadline_ms=deadline,
                idempotency_key=key, meta={"sim_ms": sim_ms})


def test_bounded_concurrency_per_pool():
    sched, clock = make(max_concurrent_retrievals=2)
    results = []
    for _ in range(6):
        sched.submit(task(sched), lambda w: 1, results.append)
    assert sched.pools["retrieval"].busy_slots == 2 and len(sched.pools["retrieval"].pending) == 4
    clock.run()
    assert len(results) == 6 and all(r.ok for r in results) and sched.pools["retrieval"].max_running == 2


def test_priority_then_fifo_and_aging_prevents_starvation():
    sched, clock = make(max_concurrent_retrievals=1, **{"priorities.aging_ms": 1000.0})
    order = []
    sched.submit(task(sched, prio=1, sim_ms=100), lambda w: order.append("first"), lambda r: None)
    sched.submit(task(sched, prio=3, sim_ms=10), lambda w: order.append("low"), lambda r: None)
    sched.submit(task(sched, prio=1, sim_ms=10), lambda w: order.append("high"), lambda r: None)
    clock.run()
    assert order == ["first", "high", "low"]                       # higher priority first, FIFO within a level
    sched, clock = make(max_concurrent_retrievals=1, **{"priorities.aging_ms": 50.0})
    order = []
    sched.submit(task(sched, prio=1, sim_ms=500), lambda w: order.append("busy"), lambda r: None)
    sched.submit(task(sched, prio=3, sim_ms=10), lambda w: order.append("old_low"), lambda r: None)
    clock.call_at(400, 1, lambda: sched.submit(task(sched, prio=1, sim_ms=10), lambda w: order.append("new_high"),
                                               lambda r: None))
    clock.run()
    assert order == ["busy", "old_low", "new_high"]                 # waited 500 ms = 10 levels of aging: not starved


def test_full_queue_sheds_less_important_work_then_rejects():
    sched, clock = make(max_concurrent_retrievals=1, **{"queues.retrieval": 2})
    out = {}
    for name, prio in [("run", 1), ("a", 2), ("b", 3)]:
        sched.submit(task(sched, prio=prio), lambda w: 1, lambda r, n=name: out.setdefault(n, r.status))
    sched.submit(task(sched, prio=1), lambda w: 1, lambda r: out.setdefault("urgent", r.status))   # sheds "b"
    sched.submit(task(sched, prio=3), lambda w: 1, lambda r: out.setdefault("late_low", r.status))  # rejected
    clock.run()
    assert out["b"] == TaskStatus.CANCELLED and out["late_low"] == TaskStatus.FAILED
    assert out["urgent"] == TaskStatus.COMPLETED and sched.stats["shed"] == 1 and sched.stats["rejected"] == 1
    assert sched.pools["retrieval"].max_pending <= 2


def test_idempotent_tasks_run_once():
    sched, clock = make()
    calls, got = [], []
    sched.submit(task(sched, key="k1"), lambda w: calls.append(1) or "r", got.append)
    sched.submit(task(sched, key="k1"), lambda w: calls.append(2) or "r", got.append)   # joins the running one
    clock.run()
    sched.submit(task(sched, key="k1"), lambda w: calls.append(3) or "r", got.append)   # reuses the result
    clock.run()
    assert calls == [1] and [g.result for g in got] == ["r", "r", "r"] and sched.stats["result_reused"] == 1


def test_deadlines_pending_and_running():
    sched, clock = make(max_concurrent_retrievals=1)
    got = {}
    sched.submit(task(sched, sim_ms=100), lambda w: 1, lambda r: got.setdefault("blocker", r.status))
    sched.submit(task(sched, deadline=50.0), lambda w: got.setdefault("ran", True), lambda r: got.setdefault("p", r))
    clock.run()
    assert got["p"].status == TaskStatus.TIMED_OUT and "ran" not in got          # expired while pending: never ran
    sched, clock = make(max_concurrent_retrievals=1)
    sched.submit(task(sched, sim_ms=100, deadline=30.0), lambda w: 1, lambda r: got.setdefault("slow", r))
    sched.submit(task(sched, sim_ms=5), lambda w: 1, lambda r: got.setdefault("next", r))
    seen = []
    clock.call_at(40, 0, lambda: seen.append((sched.zombies(), sched.pools["retrieval"].busy_slots)))
    clock.run()
    assert got["slow"].status == TaskStatus.TIMED_OUT and got["next"].ok
    assert seen == [(1, 1)]                       # timed out at 30 ms, slot held until the worker returned (100 ms)
    assert sched.stats["late_results_dropped"] == 1


def test_task_result_carries_version_and_correlation():
    sched, clock = make()
    t = Task(sched.new_task_id(), TaskType.ASSEMBLE, "s9", 1, 7, 2, 0.0, correlation_id="s9/u3", meta={"sim_ms": 1})
    got = []
    sched.submit(t, lambda w: "x", got.append)
    clock.run()
    r = got[0]
    assert (r.task_id, r.state_version, r.epoch, r.correlation_id, r.result) == (t.task_id, 7, 2, "s9/u3", "x")
    assert r.metadata["attempts"] == 1 and t.status == TaskStatus.COMPLETED and t.pool == "cpu"


def test_a_failing_result_handler_does_not_stall_the_pool():
    sched, clock = make(max_concurrent_retrievals=1)
    events, got = [], []
    sched.on_event = lambda t, kind, p: events.append(kind)

    def boom(r):
        raise RuntimeError("handler bug")
    sched.submit(task(sched), lambda w: 1, boom)
    sched.submit(task(sched), lambda w: 2, got.append)
    clock.run()
    assert [r.result for r in got] == [2] and "callback_error" in events and sched.stats["callback_errors"] == 1
