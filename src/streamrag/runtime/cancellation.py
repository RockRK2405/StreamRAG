"""Cooperative cancellation (docs/runtime/05).

Work is never killed. A ``CancellationToken`` is a thread-safe flag that the worker checks at its checkpoints
(between retrieval stages, per streamed LLM chunk, between answer stages) and inside interruptible waits; the worker
then stops and releases what it holds. Tokens form a tree: cancelling a session token cancels every task token
under it (session reset / cancel), cancelling a task token cancels only that task.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable


class TaskCancelled(Exception):
    """Raised at a worker checkpoint when its token was cancelled."""


class CancellationToken:
    def __init__(self, parent: "CancellationToken | None" = None, name: str = "") -> None:
        self.name = name
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.reason: str | None = None
        self._callbacks: list[Callable[[str], None]] = []
        self._children: list[CancellationToken] = []
        self.parent = parent
        if parent is not None:
            parent._adopt(self)

    def _adopt(self, child: "CancellationToken") -> None:
        with self._lock:
            self._children.append(child)
            cancelled = self._event.is_set()
        if cancelled:
            child.cancel(self.reason or "parent_cancelled")

    def cancel(self, reason: str = "cancelled") -> bool:
        """Returns False when it was already cancelled."""
        with self._lock:
            if self._event.is_set():
                return False
            self.reason = reason
            self._event.set()
            callbacks, children = list(self._callbacks), list(self._children)
        for fn in callbacks:
            fn(reason)
        for c in children:
            c.cancel(reason)
        return True

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TaskCancelled(self.reason or "cancelled")

    def wait(self, seconds: float) -> bool:
        """Interruptible sleep (used for I/O waits and injected delays); True if cancelled meanwhile."""
        return self._event.wait(max(0.0, seconds))

    def add_callback(self, fn: Callable[[str], None]) -> None:
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(fn)
                return
        fn(self.reason or "cancelled")

    def release(self) -> None:
        """Detach from the parent once the task is terminal (no token tree growth over a long session)."""
        if self.parent is not None:
            with self.parent._lock:
                if self in self.parent._children:
                    self.parent._children.remove(self)
        with self._lock:
            self._callbacks.clear()


class CancellationManager:
    """Owns the session token per session (replaced on reset) and the task tokens below it."""

    def __init__(self) -> None:
        self._session: dict[str, CancellationToken] = {}
        self._task: dict[str, CancellationToken] = {}
        self.stats: Counter = Counter()

    def session_token(self, session_id: str) -> CancellationToken:
        tok = self._session.get(session_id)
        if tok is None or tok.is_cancelled():
            tok = self._session[session_id] = CancellationToken(name=f"session:{session_id}")
        return tok

    def task_token(self, task_id: str, session_id: str) -> CancellationToken:
        tok = self._task[task_id] = CancellationToken(self.session_token(session_id), name=f"task:{task_id}")
        return tok

    def token(self, task_id: str) -> CancellationToken | None:
        return self._task.get(task_id)

    def cancel_task(self, task_id: str, reason: str) -> bool:
        tok = self._task.get(task_id)
        if tok is not None and tok.cancel(reason):
            self.stats[reason] += 1
            return True
        return False

    def cancel_session(self, session_id: str, reason: str) -> int:
        tok = self._session.pop(session_id, None)
        if tok is None:
            return 0
        n = sum(1 for t in self._task.values() if t.parent is tok and not t.is_cancelled())
        tok.cancel(reason)
        self.stats[reason] += n
        return n

    def finished(self, task_id: str) -> None:
        tok = self._task.pop(task_id, None)
        if tok is not None:
            tok.release()

    @property
    def live_tokens(self) -> int:
        return len(self._task)
