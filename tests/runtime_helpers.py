"""Phase 8 runtime test helpers. TEST FIXTURE corpora; scripted / simulated LLMs (never a real model).

``rt_stack`` builds a fixture stack (hashing embedder, multi-intent + session + generation enabled, extractive unless
an LLM is passed to the runtime). ``virtual`` runs a scenario on the deterministic clock; ``realtime`` on the wall
clock with real worker threads.
"""

from __future__ import annotations

import asyncio

from grounding_helpers import grounding_stack
from streamrag.runtime import StreamingRuntime


def rt_stack(tmp_path_factory, corpus: str = "corpus_grounding", **runtime_overrides):
    ov = {f"runtime.{k}": v for k, v in runtime_overrides.items()}
    return grounding_stack(tmp_path_factory, corpus, overrides=ov)


def with_runtime(stack, **overrides):
    """Same stack, different runtime config (dotted keys below ``runtime``)."""
    rcfg = stack.cfg.runtime
    for key, val in overrides.items():
        parts = key.split(".")
        if len(parts) == 1:
            rcfg = rcfg.model_copy(update={key: val})
        else:
            sub = getattr(rcfg, parts[0]).model_copy(update={parts[1]: val})
            rcfg = rcfg.model_copy(update={parts[0]: sub})
    cfg = stack.cfg.model_copy(update={"runtime": rcfg})
    st = stack.with_config(cfg)
    st._intent_stack = stack.intent_stack
    st._grounding = stack.grounding
    return st


def utter(rt, sid: str, uid: str, chunks: list[str], t0: float, step: float = 300.0, end_gap: float = 400.0) -> float:
    """Virtual mode: stream chunks of one utterance from ``t0``; returns the end time."""
    for i, c in enumerate(chunks):
        rt.push_transcript_delta(sid, uid, c, at_ms=t0 + i * step)
    t_end = t0 + (len(chunks) - 1) * step + end_gap
    rt.end_utterance(sid, uid, at_ms=t_end)
    return t_end


def virtual(stack, turns: list[list[str]], llm=None, faults=None, gap: float = 3000.0, sid: str = "s1",
            end_at: float | None = None):
    rt = StreamingRuntime(stack.cfg, stack, mode="virtual", llm=llm, faults=faults)
    rt.start_session(sid)
    t = 0.0
    for n, chunks in enumerate(turns, start=1):
        t = utter(rt, sid, f"u{n}", chunks, t) + gap
    rt.end_session(sid, at_ms=end_at if end_at is not None else t)
    rt.run()
    return rt, rt.events(sid)


async def drive(rt, sid: str, turns: list[list[str]], step_s: float = 0.1, gap_s: float = 0.2) -> None:
    for n, chunks in enumerate(turns, start=1):
        for c in chunks:
            rt.push_transcript_delta(sid, f"u{n}", c)
            await asyncio.sleep(step_s)
        rt.end_utterance(sid, f"u{n}")
        await asyncio.sleep(gap_s)


def realtime(stack, turns: list[list[str]], llm=None, faults=None, step_s: float = 0.1, gap_s: float = 0.2,
             timeout_s: float = 60.0):
    async def main():
        rt = await StreamingRuntime(stack.cfg, stack, mode="realtime", llm=llm, faults=faults).start()
        sid = rt.start_session("s1")
        await drive(rt, sid, turns, step_s, gap_s)
        await rt.complete_session(sid, timeout_s)
        await rt.shutdown()
        return rt, rt.events(sid)
    return asyncio.run(main())


def of(events, *types: str) -> list:
    return [e for e in events if e.type.value in types]


def commits(events) -> list:
    return [e for e in events if e.type.value == "ANSWER_COMMITTED"]
