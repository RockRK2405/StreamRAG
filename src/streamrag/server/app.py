"""StreamRAG demo server (Phase 11): the frozen final pipeline behind a small HTTP API and a live web UI.

    GET  /                          demo UI (static, same origin)
    GET  /health                    liveness: the process serves requests
    GET  /ready                     readiness: index, embedder, verifier, runtime, LLM (200 ready / 503 not ready;
                                    "degraded": true when the LLM is unavailable and extractive answers are used)
    GET  /api/info                  corpus, pipeline switches, LLM backend
    GET  /api/scenarios             scripted demo scenarios (demo/scenarios.yaml)
    POST /api/sessions              new session -> {"session_id"}
    POST /api/sessions/{id}/chunks  {"text", "replaces"?} transcript chunk of the current utterance
    POST /api/sessions/{id}/end     end of the current utterance
    POST /api/sessions/{id}/scenario {"id"} the server streams a scripted scenario into the session
    GET  /api/sessions/{id}/events  Server-Sent Events: safe UI events (ui_events.py)
    DELETE /api/sessions/{id}       cancel and close
    GET  /api/sources/{citation}    the cited section (document title, section title, text) for the source panel

Everything runs in one asyncio loop with one shared StreamingRuntime (Phase 8). Limits: bounded request sizes
(http.py), at most ``max_sessions`` open sessions, idle sessions closed after ``idle_s``, chunk length bounded by
``runtime.max_chunk_chars``. Retrieved text is untrusted data: the UI renders it as text only.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from streamrag.server import http as H
from streamrag.server.ui_events import UiMapper
from streamrag.telemetry import get_logger

log = get_logger("streamrag.server")
STATIC = Path(__file__).resolve().parent / "static"
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
SID_RE = re.compile(r"^[a-z0-9]{6,32}$")


@dataclass
class DemoSession:
    sid: str
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    n_utt: int = 0
    open_uid: str | None = None
    chunks: int = 0
    task: asyncio.Task | None = None

    def next_uid(self) -> str:
        if self.open_uid is None:
            self.n_utt += 1
            self.open_uid, self.chunks = f"u{self.n_utt}", 0
        return self.open_uid


class DemoApp:
    def __init__(self, repo: Path, corpus: Path | None = None, llm: str = "auto", profile: Path | None = None,
                 scenarios: Path | None = None, index_root: Path | None = None, max_sessions: int = 8,
                 idle_s: float = 900.0) -> None:
        self.repo, self.llm_mode = Path(repo), llm
        self.corpus = corpus
        self.profile, self.index_root = profile, index_root
        self.scenarios_path = scenarios or self.repo / "demo" / "scenarios.yaml"
        self.max_sessions, self.idle_s = max_sessions, idle_s
        self.sessions: dict[str, DemoSession] = {}
        self.started = time.monotonic()
        self.rt = None
        self.stack = None
        self.cfg = None
        self.error: str | None = None
        self._ids = itertools.count(1)
        self._llm_probe: tuple[float, bool] = (0.0, False)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> "DemoApp":
        from streamrag.config import load_final_config
        from streamrag.retrieval import build_index
        from streamrag.runtime import StreamingRuntime
        from streamrag.streaming.factory import build_stack
        ov = {"telemetry.log_level": "WARNING"}
        if self.corpus is not None:
            ov["paths.corpus"] = str(self.corpus)
        if self.index_root is not None:
            ov["paths.index_root"] = str(self.index_root)
        if self.llm_mode == "off":
            ov["generation.backend"] = "extractive"
        try:
            cfg = load_final_config(self.repo, ov, self.profile)
            bundle = build_index(cfg)
            stack = build_stack(cfg, bundle.path)
            grounding = stack.grounding                       # verifier + catalog + LLM backend (auto / extractive)
            self.cfg, self.stack = cfg, stack
            self.rt = await StreamingRuntime(cfg, stack, llm=grounding.backend).start()
            self._llm_probe = (time.monotonic(), grounding.backend is not None)
            if grounding.backend is not None:                 # warm the model so the first question is not a cold start
                await asyncio.get_running_loop().run_in_executor(None, self._warm_llm)
        except Exception as exc:     # noqa: BLE001 - reported by /ready instead of crashing the process
            self.error = f"{exc.__class__.__name__}: {exc}"
            log.error("startup_failed", extra={"fields": {"error": self.error}})
        asyncio.get_running_loop().create_task(self._reaper())
        return self

    def _warm_llm(self) -> None:
        try:
            self.stack.grounding.backend.complete([{"role": "user", "content": "Reply with {}"}], {"type": "object"})
        except Exception:  # noqa: BLE001 - warm-up is best effort
            pass

    async def shutdown(self) -> None:
        for s in list(self.sessions.values()):
            self._close(s)
        if self.rt is not None:
            await self.rt.shutdown()

    async def _reaper(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for s in list(self.sessions.values()):
                if now - s.touched > self.idle_s:
                    self._close(s)

    def _close(self, s: DemoSession) -> None:
        if s.task is not None and not s.task.done():
            s.task.cancel()
        try:
            self.rt.cancel_session(s.sid, "client_closed")
        except Exception:  # noqa: BLE001
            pass
        self.sessions.pop(s.sid, None)

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        return {"status": "ok", "uptime_s": round(time.monotonic() - self.started, 1)}

    def ready(self) -> tuple[int, dict]:
        checks: dict[str, object] = {}
        ok = self.error is None and self.rt is not None and self.stack is not None
        if self.stack is not None:
            g = self.stack.grounding
            checks["index"] = {"ok": len(self.stack.bundle.chunks) > 0, "chunks": len(self.stack.bundle.chunks),
                               "test_fixture": self.stack.bundle.manifest.is_test_fixture}
            checks["embedder"] = {"ok": self.stack.service.embedder is not None}
            checks["verifier"] = {"ok": g.nli is not None or self.cfg.generation.verifier != "nli",
                                  "model": self.cfg.generation.nli_model if g.nli is not None else "rules"}
            llm_ok = self._llm_alive()
            checks["llm"] = {"ok": llm_ok, "backend": "ollama" if g.backend is not None else "extractive",
                             "model": self.cfg.generation.model if g.backend is not None else None,
                             "note": None if llm_ok else "extractive fallback answers (verified, no LLM)"}
            ok = ok and all(checks[k]["ok"] for k in ("index", "embedder", "verifier"))
        checks["runtime"] = {"ok": self.rt is not None and not getattr(self.rt, "closed", False),
                             "sessions": len(self.sessions), "max_sessions": self.max_sessions}
        degraded = ok and not checks.get("llm", {}).get("ok", False)
        body = {"ready": ok, "degraded": degraded, "checks": checks}
        if self.error:
            body["error"] = self.error
        return (200 if ok else 503), body

    def _llm_alive(self) -> bool:
        """LLM reachability, re-probed at most every 10 s (an unreachable LLM degrades, it does not fail /ready)."""
        g = self.stack.grounding
        if g.backend is None:
            return False
        t, alive = self._llm_probe
        if time.monotonic() - t > 10.0:
            from streamrag.generation.llm import OllamaBackend
            alive = OllamaBackend.reachable(self.cfg.generation.ollama_url, self.cfg.generation.model)
            self._llm_probe = (time.monotonic(), alive)
        return alive

    def info(self) -> dict:
        c = self.cfg
        return {"corpus": str(c.paths.corpus.name) if c else None,
                "test_fixture": self.stack.bundle.manifest.is_test_fixture if self.stack else None,
                "pipeline": {"multi_intent": c.multi_intent.enabled, "session": c.session.enabled,
                             "grounded_generation": c.generation.enabled,
                             "adaptive_retrieval": c.adaptive_retrieval.enabled,
                             "answerability": c.generation.answerability, "rerank": c.streaming.rerank} if c else {},
                "llm": {"backend": "ollama" if self.stack and self.stack.grounding.backend else "extractive",
                        "model": c.generation.model if c else None}}

    def scenarios(self) -> list[dict]:
        if not self.scenarios_path.exists():
            return []
        data = yaml.safe_load(self.scenarios_path.read_text()) or []
        return [{"id": s["id"], "title": s.get("title", s["id"]), "shows": s.get("shows", ""),
                 "turns": [[c if isinstance(c, str) else c.get("text", "") for c in t.get("chunks", [])]
                           for t in s.get("turns", [])]} for s in data]

    # ------------------------------------------------------------------ sessions
    def _session(self, sid: str) -> DemoSession:
        s = self.sessions.get(sid)
        if s is None:
            raise H.HttpError(404, "unknown session")
        s.touched = time.monotonic()
        return s

    def new_session(self) -> DemoSession:
        if self.rt is None:
            raise H.HttpError(503, "not ready")
        if len(self.sessions) >= self.max_sessions:          # evict the least recently used idle session first
            idle = sorted((s for s in self.sessions.values() if time.monotonic() - s.touched > 30.0
                           and (s.task is None or s.task.done())), key=lambda s: s.touched)
            if not idle:
                raise H.HttpError(429, "too many open sessions")
            self._close(idle[0])
        sid = f"s{next(self._ids):04d}{int(time.time()) % 100000:05d}"
        self.rt.start_session(sid)
        s = DemoSession(sid)
        self.sessions[sid] = s
        return s

    def chunk(self, s: DemoSession, text: str, replaces: int | None) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise H.HttpError(400, "text must be a non-empty string")
        if len(text) > self.cfg.runtime.max_chunk_chars:
            raise H.HttpError(413, "chunk too long")
        if replaces is not None and (not isinstance(replaces, int) or replaces < 0 or replaces >= s.chunks):
            raise H.HttpError(400, "replaces must index an earlier chunk of the current utterance")
        uid = s.next_uid()
        accepted = self.rt.push_transcript_delta(s.sid, uid, text, replaces=replaces)
        if replaces is None:
            s.chunks += 1
        return {"utterance": uid, "accepted": bool(accepted)}

    def end(self, s: DemoSession) -> dict:
        if s.open_uid is None:
            raise H.HttpError(409, "no open utterance")
        uid, s.open_uid = s.open_uid, None
        self.rt.end_utterance(s.sid, uid)
        return {"utterance": uid}

    def run_scenario(self, s: DemoSession, sc_id: str) -> dict:
        data = {x["id"]: x for x in (yaml.safe_load(self.scenarios_path.read_text()) or [])}
        if sc_id not in data:
            raise H.HttpError(404, "unknown scenario")
        if s.task is not None and not s.task.done():
            raise H.HttpError(409, "a scenario is already running in this session")
        s.task = asyncio.get_running_loop().create_task(self._drive(s, data[sc_id]))
        return {"scenario": sc_id, "turns": len(data[sc_id].get("turns", []))}

    async def _drive(self, s: DemoSession, sc: dict) -> None:
        interval = float(sc.get("interval_ms", 450)) / 1000.0
        for turn in sc.get("turns", []):
            for c in turn.get("chunks", []):
                if isinstance(c, dict):
                    self.chunk(s, c["text"], int(c["replaces"]))
                else:
                    self.chunk(s, c, None)
                await asyncio.sleep(interval)
            uid = s.open_uid
            self.end(s)
            if "gap_ms" in turn:                    # scripted overlap: the next turn starts during this answer
                await asyncio.sleep(float(turn["gap_ms"]) / 1000.0)
            else:
                await self._wait_turn(s.sid, uid, 90.0)
                await asyncio.sleep(0.6)

    async def _wait_turn(self, sid: str, uid: str, timeout_s: float) -> None:
        end = time.monotonic() + timeout_s
        bus = self.rt.sessions[sid].bus
        while time.monotonic() < end:
            if any(e.utterance_id == uid and e.type.value in ("ANSWER_COMMITTED", "ERROR") for e in bus.events):
                return
            await asyncio.sleep(0.05)

    def source(self, citation: str) -> dict:
        ch = next((c for c in self.stack.bundle.chunks if c.citation == citation), None)
        if ch is None:
            raise H.HttpError(404, "unknown citation")
        return {"citation": ch.citation, "document_id": ch.document_id, "document_title": ch.title,
                "section_title": ch.section_title, "text": ch.text,
                "note": "Retrieved document text is shown as data; it is never executed or followed as instructions."}

    async def stream_events(self, sid: str, writer: asyncio.StreamWriter) -> None:
        s = self._session(sid)
        rs = self.rt.sessions[sid]
        mapper = UiMapper(lambda uid: rs.lane.finals.get(uid) if rs.lane is not None else None)
        await H.start_sse(writer)
        await H.sse_event(writer, {"type": "hello", "session": sid})
        i, last_ping = 0, time.monotonic()
        while sid in self.sessions:
            evs = rs.bus.events
            while i < len(evs):
                for ui in mapper.map(evs[i]):
                    await H.sse_event(writer, ui)
                i += 1
            s.touched = time.monotonic()
            if rs.closed:
                break
            if time.monotonic() - last_ping > 15.0:
                await H.sse_comment(writer)
                last_ping = time.monotonic()
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ routing
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        t0 = time.perf_counter()
        status, path, method = 500, "?", "?"
        try:
            req = await H.read_request(reader)
            path, method = req.path, req.method
            status = await self._route(req, writer)
        except ConnectionError:
            status = 0
        except H.HttpError as exc:
            status = exc.status
            await H.send_json(writer, exc.status, {"error": exc.message})
        except Exception as exc:     # noqa: BLE001 - never leak internals to the client
            status = 500
            log.error("request_failed", extra={"fields": {"path": path, "error": exc.__class__.__name__}})
            try:
                await H.send_json(writer, 500, {"error": "internal error"})
            except Exception:  # noqa: BLE001
                pass
        finally:
            if path not in ("/health", "/ready") and status:
                log.info("request", extra={"fields": {"method": method, "path": re.sub(r"/s\d+", "/{sid}", path),
                                                      "status": status,
                                                      "ms": round((time.perf_counter() - t0) * 1000, 1)}})
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _route(self, req: H.Request, w: asyncio.StreamWriter) -> int:
        m, p = req.method, req.path
        if m == "GET" and p == "/health":
            await H.send_json(w, 200, self.health())
            return 200
        if m == "GET" and p == "/ready":
            code, body = self.ready()
            await H.send_json(w, code, body)
            return code
        if m == "GET" and (p == "/" or p.startswith("/static/")):
            return await self._static(p, w)
        if not p.startswith("/api/"):
            raise H.HttpError(404, "not found")
        if self.rt is None:
            raise H.HttpError(503, "not ready" + (f": {self.error}" if self.error else ""))
        if m == "GET" and p == "/api/info":
            await H.send_json(w, 200, self.info())
            return 200
        if m == "GET" and p == "/api/scenarios":
            await H.send_json(w, 200, {"scenarios": self.scenarios()})
            return 200
        if m == "POST" and p == "/api/sessions":
            s = self.new_session()
            await H.send_json(w, 201, {"session_id": s.sid})
            return 201
        if m == "GET" and p.startswith("/api/sources/"):
            await H.send_json(w, 200, self.source(p[len("/api/sources/"):]))
            return 200
        mt = re.match(r"^/api/sessions/(s\d+)(?:/(chunks|end|scenario|events))?$", p)
        if not mt:
            raise H.HttpError(404, "not found")
        sid, action = mt.group(1), mt.group(2)
        if m == "DELETE" and action is None:
            self._close(self._session(sid))
            await H.send_json(w, 200, {"closed": sid})
            return 200
        if m == "GET" and action == "events":
            await self.stream_events(sid, w)
            return 200
        s = self._session(sid)
        if m == "POST" and action == "chunks":
            body = req.json()
            await H.send_json(w, 200, self.chunk(s, body.get("text"), body.get("replaces")))
            return 200
        if m == "POST" and action == "end":
            await H.send_json(w, 200, self.end(s))
            return 200
        if m == "POST" and action == "scenario":
            await H.send_json(w, 200, self.run_scenario(s, str(req.json().get("id", ""))))
            return 200
        raise H.HttpError(405, "method not allowed")

    async def _static(self, p: str, w: asyncio.StreamWriter) -> int:
        name = "index.html" if p == "/" else p[len("/static/"):]
        f = (STATIC / name).resolve()
        if STATIC not in f.parents or not f.is_file() or f.suffix not in CTYPES:
            raise H.HttpError(404, "not found")
        await H.send(w, 200, f.read_bytes(), CTYPES[f.suffix])
        return 200


async def serve(app: DemoApp, host: str, port: int) -> None:
    await app.start()
    server = await asyncio.start_server(app.handle, host, port, limit=H.MAX_HEADER_BYTES + 1024)
    log.info("server_started", extra={"fields": {"host": host, "port": port, "ready": app.ready()[0] == 200}})
    print(f"StreamRAG demo: http://{host}:{port}/  (health: /health, readiness: /ready)", flush=True)
    async with server:
        try:
            await server.serve_forever()
        finally:
            await app.shutdown()
