"""Task model of the streaming runtime (docs/runtime/03).

A task is one unit of asynchronous work owned by one session: a retrieval subtask, an evidence assembly, a draft or a
final answer. It records when and why it exists (``correlation_id``: the turn; ``causation_id``: the event that
caused it; ``parent_task_id``), the session state it was computed from (``state_version`` / ``epoch``, checked
before its result may change anything) and its scheduling attributes (priority, deadline, idempotency key).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    SUPERSEDED = "SUPERSEDED"


TERMINAL = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMED_OUT,
                      TaskStatus.SUPERSEDED})


class TaskType(str, Enum):
    LEXICAL = "lexical"                       # BM25 search for one query
    DENSE = "dense"                           # query embedding + vector search for one query
    RETRIEVAL = "retrieval"                   # unsplit hybrid retrieval (split_retrieval = false)
    ASSEMBLE = "assemble"                     # RRF fusion + dedup (+ rerank) of the partial results
    GENERATION = "generation"                 # final grounded answer (LLM + verification + citations)
    ANSWER_EXTRACTIVE = "answer_extractive"   # final answer without a model call (degraded / extractive backend)
    DRAFT = "draft"                           # verified extractive draft while the user speaks
    VALIDATION_RETRIEVAL = "validation_retrieval"
    ANALYTICS = "analytics"                   # optional enrichment (lowest priority)


# worker pool (resource class) per task type: bounded separately so a slow LLM never starves retrieval
POOL_OF: dict[TaskType, str] = {
    TaskType.LEXICAL: "retrieval", TaskType.DENSE: "retrieval", TaskType.RETRIEVAL: "retrieval",
    TaskType.VALIDATION_RETRIEVAL: "retrieval", TaskType.ASSEMBLE: "cpu", TaskType.DRAFT: "cpu",
    TaskType.ANALYTICS: "cpu", TaskType.GENERATION: "llm", TaskType.ANSWER_EXTRACTIVE: "cpu",
}


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    session_id: str
    priority: int
    state_version: int
    epoch: int
    created_at_ms: float
    correlation_id: str | None = None
    causation_id: str | None = None
    parent_task_id: str | None = None
    deadline_ms: float | None = None          # absolute, runtime clock
    idempotency_key: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    started_at_ms: float | None = None
    completed_at_ms: float | None = None
    attempt: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pool(self) -> str:
        return POOL_OF[self.task_type]

    def summary(self) -> dict:
        return {"task_id": self.task_id, "task_type": self.task_type.value, "priority": self.priority,
                "status": self.status.value, "attempt": self.attempt, "state_version": self.state_version,
                "epoch": self.epoch, "deadline_ms": None if self.deadline_ms is None else round(self.deadline_ms, 3),
                "parent_task_id": self.parent_task_id, "idempotency_key": self.idempotency_key,
                **{k: v for k, v in self.meta.items() if k in ("query_id", "intent_id", "answer_kind", "kind",
                                                                "deadline_limited_by", "mode")}}


@dataclass(frozen=True)
class TaskResult:
    """What a worker hands back. Nothing in it is applied before the state coordinator checked that the task is
    still relevant (docs/runtime/05, race-condition prevention)."""

    task_id: str
    task_type: TaskType
    session_id: str
    state_version: int
    epoch: int
    correlation_id: str | None
    status: TaskStatus
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == TaskStatus.COMPLETED
