"""Deadlines and latency budgets (docs/runtime/06).

Every task gets an absolute deadline on the runtime clock. A child task never gets more time than its parent has
left minus the time reserved for the stages that still follow it: retrieval cannot use the time generation and
validation will need (deadline propagation). The turn budget and the reserves are configuration
(``runtime.budget``), not claimed performance targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from streamrag.runtime.tasks import TaskType

_RETRIEVAL = {TaskType.LEXICAL, TaskType.DENSE, TaskType.RETRIEVAL, TaskType.ASSEMBLE}


@dataclass(frozen=True)
class Deadline:
    at_ms: float
    limited_by: str                      # "task_timeout" | "turn_budget"

    def remaining_ms(self, now_ms: float) -> float:
        return self.at_ms - now_ms

    def expired(self, now_ms: float) -> bool:
        return now_ms >= self.at_ms


class TimeoutManager:
    def __init__(self, rcfg) -> None:
        self.cfg = rcfg
        self.timeouts = rcfg.timeouts_ms

    def timeout_ms(self, task_type: TaskType) -> float:
        name = {TaskType.RETRIEVAL: "dense", TaskType.ANSWER_EXTRACTIVE: "draft"}.get(task_type, task_type.value)
        return float(getattr(self.timeouts, name, self.timeouts.generation if task_type == TaskType.ANALYTICS
                             else self.timeouts.assemble))

    def reserve_after_ms(self, task_type: TaskType) -> float:
        b = self.cfg.budget
        if task_type in _RETRIEVAL:
            return b.generation_reserve_ms + b.validation_reserve_ms
        if task_type in (TaskType.GENERATION, TaskType.DRAFT, TaskType.ANSWER_EXTRACTIVE):
            return b.validation_reserve_ms
        return 0.0

    def turn_deadline(self, turn_start_ms: float) -> float:
        return turn_start_ms + self.cfg.budget.turn_ms

    def deadline_for(self, task_type: TaskType, now_ms: float, parent_deadline_ms: float | None = None) -> Deadline:
        own = now_ms + self.timeout_ms(task_type)
        if parent_deadline_ms is None:
            return Deadline(own, "task_timeout")
        budget = parent_deadline_ms - self.reserve_after_ms(task_type)
        # at least a minimal slot: a budget already spent still lets the task fail fast instead of never running
        budget = max(budget, now_ms + 1.0)
        return Deadline(budget, "turn_budget") if budget < own else Deadline(own, "task_timeout")
