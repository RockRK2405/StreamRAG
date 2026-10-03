"""Simulated LLM backend for load and failure tests. SYNTHETIC - every result produced with it is labelled so.

It answers the real Phase 7 facts prompt deterministically: one sentence per planned fact, verbatim, with the fact
id and its labels (the behaviour of a perfectly obedient writer), after a fixed latency. The latency is a wait that
honours the runtime call context (cancellation, deadline), so concurrency, cancellation and timeouts behave as with
a slow remote model without measuring any real model.
"""

from __future__ import annotations

import json
import re
import time

from streamrag.generation.llm import CALL_CONTEXT, GenerationCancelled, LLMResponse, request_sha1

_SEC = re.compile(r"^SECTION (\S+) ")
_FACT = re.compile(r"^\s+(F\d+) \[([^\]]*)\]: (\".*\")$")


def _facts(messages: list[dict]) -> dict[str, list[tuple[str, list[str], str]]]:
    out, cur = {}, None
    for line in messages[-1]["content"].splitlines():
        m = _SEC.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = _FACT.match(line)
        if m and cur:
            out[cur].append((m.group(1), [x.strip() for x in m.group(2).split(",") if x.strip()],
                             json.loads(m.group(3))))
    return out


class SimulatedLLM:
    name = "simulated"

    def __init__(self, latency_ms: float = 1500.0, ttft_ms: float = 100.0, model: str = "simulated-echo") -> None:
        self.latency_ms, self.ttft_ms, self.model = latency_ms, ttft_ms, model
        self.calls = 0

    def complete(self, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        self.calls += 1
        sha = request_sha1(self.model, messages, schema)
        ctx = CALL_CONTEXT.get()
        t0 = time.monotonic()
        end = t0 + self.latency_ms / 1000.0
        while time.monotonic() < end:
            if ctx.is_cancelled():
                raise GenerationCancelled("cancelled during simulated generation")
            if ctx.deadline is not None and time.monotonic() >= ctx.deadline:
                return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha,
                                   total_ms=(time.monotonic() - t0) * 1000.0, error="TimeoutError: timed out")
            time.sleep(min(0.005, max(0.0, end - time.monotonic())))
        secs = _facts(messages)
        text = json.dumps({"sections": [{"section_id": sid, "sentences": [
            {"text": t, "facts": [f], "evidence": labs} for f, labs, t in facts]} for sid, facts in secs.items()]})
        return LLMResponse(ok=True, text=text, backend=self.name, model=self.model, request_sha1=sha,
                           ttft_ms=self.ttft_ms, total_ms=(time.monotonic() - t0) * 1000.0,
                           prompt_tokens=None, output_tokens=None, meta={"synthetic": True})
