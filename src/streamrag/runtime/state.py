"""StateCoordinator (docs/runtime/09, race-condition prevention).

A session's state is owned by one actor: every mutation runs on the event loop, one at a time, in the order the
runtime dispatches it (inputs, timers, committed task results). Workers never mutate session state - they compute
from the inputs or an immutable snapshot handed to them and return a ``TaskResult``. So there is no lock on the
state; coordination is about **which** results may be applied:

* every task records the session ``state_version`` and ``epoch`` it was scheduled at;
* a result is applied only if its epoch is still current (a reset bumps the epoch) and the task is still relevant
  (its query is not superseded, its answer request is still the latest, ...). Otherwise it is discarded with a
  STALE_RESULT_DISCARDED event - it can never overwrite newer state;
* an applied result bumps ``state_version``. If applying it raises, the session is rolled back to the snapshot
  taken just before (last valid state) and the failure is reported.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from streamrag.models.events import EventType as E
from streamrag.runtime.tasks import TaskResult


class StateCoordinator:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.version = 0
        self.epoch = 0
        self.discarded: Counter = Counter()
        self.applied = 0
        self.rollbacks = 0
        self.emit: Callable[..., object] = lambda *a, **k: None
        self.checkpoint: Callable[[], object] | None = None
        self.restore: Callable[[object], None] | None = None

    def bump(self) -> int:
        self.version += 1
        return self.version

    def new_epoch(self) -> int:
        self.epoch += 1
        self.bump()
        return self.epoch

    def stale_reason(self, result: TaskResult, relevant: Callable[[], bool] | None = None) -> str | None:
        if result.epoch != self.epoch:
            return "epoch_changed"
        if relevant is not None and not relevant():
            return "superseded"
        return None

    def commit(self, result: TaskResult, apply: Callable[[], None], relevant: Callable[[], bool] | None = None,
               describe: dict | None = None, utterance_id: str | None = None, transactional: bool = False) -> bool:
        """``transactional``: snapshot the session first and roll back if ``apply`` raises (multi-step commits:
        an answer's events plus its fallback evidence). Single-step commits skip the snapshot (its cost)."""
        reason = self.stale_reason(result, relevant)
        if reason is not None:
            self.discarded[reason] += 1
            self.emit(E.STALE_RESULT_DISCARDED, "state_coordinator",
                      {"task_id": result.task_id, "task_type": result.task_type.value, "reason": reason,
                       "task_state_version": result.state_version, "current_state_version": self.version,
                       "task_epoch": result.epoch, "current_epoch": self.epoch, **(describe or {})}, utterance_id)
            return False
        snap = self.checkpoint() if transactional and self.checkpoint is not None else None
        try:
            apply()
        except Exception as exc:            # noqa: BLE001 - reported, state rolled back
            self.rollbacks += 1
            if snap is not None and self.restore is not None:
                self.restore(snap)
            self.emit(E.ERROR, "state_coordinator",
                      {"component": "state_coordinator", "error_class": exc.__class__.__name__, "recoverable": True,
                       "action": "rolled_back_to_last_valid_state", "detail": str(exc), "task_id": result.task_id},
                      utterance_id)
            return False
        self.applied += 1
        self.bump()
        return True
