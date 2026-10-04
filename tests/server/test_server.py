"""Phase 11 demo server: health / readiness, API validation and limits, a streamed session end to end, demo-check.
Runs the real server (asyncio, random port) on the demo corpus with the LLM switched off (verified extractive)."""

import asyncio
import json
from urllib.parse import quote

import pytest

from conftest import REPO, requires_bge
from grounding_helpers import requires_nli

pytestmark = [requires_bge, requires_nli]


async def _http(port, method, path, body=None, raw=None, read_all=True, length=None):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(data) if length is None else length}\r\n"
    head += "Content-Type: application/json\r\n\r\n" if body is not None else "\r\n"
    w.write(head.encode() + data)
    await w.drain()
    resp = await r.read() if read_all else b""
    w.close()
    status = int(resp.split(b" ", 2)[1]) if resp else 0
    hdrs, _, payload = resp.partition(b"\r\n\r\n")
    return status, hdrs.decode("latin-1"), payload


async def _with_server(tmp_path, fn, **kw):
    from streamrag.server.app import DemoApp
    app = DemoApp(REPO, corpus=REPO / "demo" / "corpus", llm="off", index_root=tmp_path / "idx", **kw)
    await app.start()
    srv = await asyncio.start_server(app.handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        return await fn(app, port)
    finally:
        srv.close()
        await srv.wait_closed()
        await app.shutdown()


def test_health_ready_static_and_validation(tmp_path):
    async def fn(app, port):
        s, _, b = await _http(port, "GET", "/health")
        assert s == 200 and json.loads(b)["status"] == "ok"
        s, _, b = await _http(port, "GET", "/ready")
        ready = json.loads(b)
        assert s == 200 and ready["ready"] and ready["degraded"]               # no LLM: ready, extractive answers
        assert ready["checks"]["index"]["ok"] and ready["checks"]["index"]["test_fixture"]
        assert ready["checks"]["llm"]["backend"] == "extractive"
        s, h, b = await _http(port, "GET", "/")
        assert s == 200 and b"StreamRAG" in b and "Content-Security-Policy" in h and "nosniff" in h
        assert (await _http(port, "GET", "/static/../app.py"))[0] == 404     # no path traversal
        assert (await _http(port, "GET", "/api/nope"))[0] == 404
        s, _, b = await _http(port, "POST", "/api/sessions")
        sid = json.loads(b)["session_id"]
        assert (await _http(port, "POST", f"/api/sessions/{sid}/chunks", raw=b"{bad"))[0] == 400
        assert (await _http(port, "POST", f"/api/sessions/{sid}/chunks", {"text": ""}))[0] == 400
        assert (await _http(port, "POST", f"/api/sessions/{sid}/chunks", {"text": "x", "replaces": 5}))[0] == 400
        assert (await _http(port, "POST", f"/api/sessions/{sid}/chunks", raw=b"", length=70000))[0] == 413
        assert (await _http(port, "POST", f"/api/sessions/{sid}/end"))[0] == 409   # nothing open
        assert (await _http(port, "POST", "/api/sessions/s9999/end"))[0] == 404
        s, _, b = await _http(port, "GET", "/api/sources/" + quote("FEES §1"))       # as the UI encodes it
        assert s == 200 and "4,800 euros" in json.loads(b)["text"]
        assert (await _http(port, "DELETE", f"/api/sessions/{sid}"))[0] == 200
    asyncio.run(_with_server(tmp_path, fn))


def test_streamed_session_end_to_end_over_sse(tmp_path):
    async def fn(app, port):
        _, _, b = await _http(port, "POST", "/api/sessions")
        sid = json.loads(b)["session_id"]
        r, w = await asyncio.open_connection("127.0.0.1", port)
        w.write(f"GET /api/sessions/{sid}/events HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        await w.drain()
        for c in ["How much is tuition", "per semester for", "international students?"]:
            assert (await _http(port, "POST", f"/api/sessions/{sid}/chunks", {"text": c}))[0] == 200
            await asyncio.sleep(0.3)
        assert (await _http(port, "POST", f"/api/sessions/{sid}/end"))[0] == 200
        events, buf = [], b""
        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            chunk = await asyncio.wait_for(r.read(4096), 30)
            buf += chunk
            events = [json.loads(x[6:]) for x in buf.decode("utf-8", "ignore").split("\n") if x.startswith("data: ")]
            if any(e["type"] == "answer" and e["status"] == "VALIDATED_FINAL" for e in events):
                break
        w.close()
        types = {e["type"] for e in events}
        assert {"transcript", "status", "intents", "plan", "evidence", "answer", "metrics"} <= types
        final = [e for e in events if e["type"] == "answer" and e["status"] == "VALIDATED_FINAL"][-1]
        assert any("FEES §1" in c["citations"] for c in final["claims"])
        assert "4,800 euros" in " ".join(c["text"] for c in final["claims"])
        m = [e for e in events if e["type"] == "metrics"][-1]
        assert m["ttfe_ms"] <= m["ttva_ms"]
        stages = [e["stage"] for e in events if e["type"] == "status"]
        assert stages[0] == "query" and stages[-1] == "final"
    asyncio.run(_with_server(tmp_path, fn))


def test_sessions_are_bounded_and_idle_ones_evicted(tmp_path):
    async def fn(app, port):
        ids = [json.loads((await _http(port, "POST", "/api/sessions"))[2])["session_id"] for _ in range(2)]
        assert (await _http(port, "POST", "/api/sessions"))[0] == 429           # all sessions active
        app.sessions[ids[0]].touched -= 60                                      # idle -> evicted for the new one
        s, _, b = await _http(port, "POST", "/api/sessions")
        assert s == 201 and ids[0] not in app.sessions
    asyncio.run(_with_server(tmp_path, fn, max_sessions=2))


def test_demo_check_scenarios_pass_without_llm(tmp_path):
    from streamrag.server.app import DemoApp
    from streamrag.server.democheck import demo_check
    app = DemoApp(REPO, corpus=REPO / "demo" / "corpus", llm="off", index_root=tmp_path / "idx")
    rep = asyncio.run(demo_check(app, tmp_path / "dc.json", only=["normal", "uncertainty"]))
    assert rep["summary"] == {**rep["summary"], "passed": 2, "total": 2}
    assert (tmp_path / "dc.json").exists()
