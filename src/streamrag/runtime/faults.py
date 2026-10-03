"""Controlled failure injection (docs/runtime/09; brief §56). TEST / BENCHMARK USE ONLY.

A ``FaultInjector`` holds fault specs and is consulted by the runtime's workers at well-defined points:

=============  =====================================================================================
target         where
=============  =====================================================================================
lexical        BM25 subtask start
dense          dense subtask start (``error`` = vector index unavailable -> lexical-only, degraded)
network        any retrieval subtask (connection-level failure; ``transient`` -> retried)
llm            every LLM call (``error`` / ``transient`` / ``timeout`` / ``delay``)
verification   the entailment model during answer validation (-> rules-only verification, degraded)
=============  =====================================================================================

Kinds: ``error`` (permanent), ``transient`` (retryable), ``timeout`` (blocks past the deadline), ``delay``
(adds latency). ``times`` = how many hits are affected (-1: all). Delays use the work context, so they are real
waits in realtime mode and modelled latency on the virtual clock.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from streamrag.errors import ModelNotAvailableError
from streamrag.generation.llm import CALL_CONTEXT, GenerationCancelled, LLMResponse, request_sha1
from streamrag.runtime.retry import PermanentError, TransientError


@dataclass
class Fault:
    target: str
    kind: str
    times: int = 1
    delay_ms: float = 0.0
    session_id: str | None = None
    used: int = field(default=0, compare=False)

    def active(self, session_id: str | None) -> bool:
        return (self.times < 0 or self.used < self.times) and (self.session_id is None or self.session_id == session_id)


class FaultInjector:
    def __init__(self, faults: list[Fault] | None = None) -> None:
        self.faults = list(faults or [])
        self.hits: Counter = Counter()
        self._lock = threading.Lock()

    def describe(self) -> list[dict]:
        return [{"target": f.target, "kind": f.kind, "times": f.times, "delay_ms": f.delay_ms,
                 "session_id": f.session_id} for f in self.faults]

    def _take(self, target: str, session_id: str | None) -> Fault | None:
        with self._lock:
            for f in self.faults:
                if f.target == target and f.active(session_id):
                    f.used += 1
                    self.hits[f"{target}:{f.kind}"] += 1
                    return f
        return None

    def hit(self, target: str, wctx, session_id: str | None = None) -> None:
        """Called by a worker; raises / delays according to the first active matching fault."""
        f = self._take(target, session_id)
        if f is None:
            return
        if f.kind == "delay":
            wctx.delay(f.delay_ms)
        elif f.kind == "timeout":
            wctx.delay(f.delay_ms or 10 ** 6)               # far past any deadline; cancellation ends the wait
        elif f.kind == "transient":
            raise TransientError(f"injected transient {target} failure")
        elif target == "dense":
            raise ModelNotAvailableError("injected: vector index unavailable")
        else:
            raise PermanentError(f"injected {target} failure")


class FaultyLLM:
    """Wraps an LLM backend; consults the injector on every call (target ``llm``)."""

    def __init__(self, inner, injector: FaultInjector, session_id: str | None = None) -> None:
        self.inner, self.injector, self.session_id = inner, injector, session_id
        self.name, self.model = inner.name, getattr(inner, "model", "unknown")

    def complete(self, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        f = self.injector._take("llm", self.session_id)
        if f is None:
            return self.inner.complete(messages, schema)
        sha = request_sha1(self.model, messages, schema)
        ctx = CALL_CONTEXT.get()
        if f.kind in ("delay", "timeout"):
            wait_s = (f.delay_ms or 10 ** 6) / 1000.0
            if ctx.deadline is not None:
                wait_s = min(wait_s, max(0.0, ctx.deadline - time.monotonic()))
            end = time.monotonic() + wait_s
            while time.monotonic() < end:
                if ctx.is_cancelled():
                    raise GenerationCancelled("cancelled while waiting for the LLM")
                time.sleep(min(0.01, max(0.0, end - time.monotonic())))
            if f.kind == "timeout" or (ctx.deadline is not None and time.monotonic() >= ctx.deadline):
                return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha,
                                   total_ms=wait_s * 1000.0, error="TimeoutError: injected LLM timeout (timed out)")
            return self.inner.complete(messages, schema)
        err = ("ConnectionError: injected transient LLM failure (connection reset)" if f.kind == "transient"
               else "ValueError: injected LLM error")
        return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha, error=err)
