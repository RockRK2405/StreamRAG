"""Feed a list of input events (Phase 4 format: chunks with stream timestamps) into a StreamingRuntime at their
arrival times - virtual clock or wall clock. Used by the CLI (``stream --runtime``), tests and research scripts."""

from __future__ import annotations

import asyncio

from streamrag.models.events import SessionStart
from streamrag.streaming.runner import input_times


def push_virtual(rt, inputs: list, offset_ms: float = 0.0) -> str:
    sid = inputs[0].session_id
    rt.start_session(sid)
    for t, ev in input_times(inputs):
        if not isinstance(ev, SessionStart):
            rt.push(ev, at_ms=offset_ms + t)
    return sid


async def drive_realtime(rt, inputs: list, speed: float = 1.0, close: bool = True, timeout_s: float = 120.0) -> str:
    """Push each input when its stream time is reached (``speed`` > 1 compresses time), then wait for the session
    to close (``close``)."""
    loop = asyncio.get_running_loop()
    sid = inputs[0].session_id
    rt.start_session(sid)
    t0 = loop.time()
    for t, ev in input_times(inputs):
        if isinstance(ev, SessionStart):
            continue
        delay = t0 + t / 1000.0 / speed - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        rt.push(ev)
    if close:
        await rt.wait_closed(sid, timeout_s)
    return sid


def run_inputs(cfg, stack, inputs: list, mode: str = "virtual", llm=None, faults=None, trace_dir=None,
               speed: float = 1.0):
    """One session end to end; returns (runtime, session_id). Realtime mode shuts the runtime down at the end."""
    from streamrag.runtime.runtime import StreamingRuntime
    if mode == "virtual":
        rt = StreamingRuntime(cfg, stack, mode="virtual", llm=llm, faults=faults, trace_dir=trace_dir)
        sid = push_virtual(rt, inputs)
        rt.run()
        return rt, sid

    async def main():
        rt = await StreamingRuntime(cfg, stack, mode="realtime", llm=llm, faults=faults, trace_dir=trace_dir).start()
        sid = await drive_realtime(rt, inputs, speed)
        await rt.shutdown()
        return rt, sid
    return asyncio.run(main())
