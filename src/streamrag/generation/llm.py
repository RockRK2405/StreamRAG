"""LLM gateway (ADR-007; docs/answer/02): one interface, interchangeable backends.

* ``OllamaBackend``   local model through the Ollama HTTP API (``/api/chat``): structured output via a JSON schema
                      (``format``), temperature 0 and a fixed seed, thinking disabled, streamed so that the time to
                      the first token is measured. Token counts come from Ollama's own response
                      (``prompt_eval_count`` / ``eval_count``): measured, not estimated.
* ``ScriptedBackend`` deterministic responses supplied by the caller (unit tests). TEST USE ONLY.
* ``RecordedBackend`` replays responses recorded in a trace's LLM_CALL events (deterministic replay of a session
                      that used an LLM), keyed by the SHA-1 of the request.

There is no hosted adapter in Phase 7 (the team chose the local backend); the interface takes one without changes.
Backends never raise into the pipeline: failures come back as ``LLMResponse(ok=False, error=...)`` and the
generator falls back (extractive generation).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    backend: str = ""
    model: str = ""
    request_sha1: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


def request_sha1(model: str, messages: list[dict], schema: dict | None) -> str:
    blob = json.dumps({"model": model, "messages": messages, "schema": schema}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


class OllamaBackend:
    name = "ollama"

    def __init__(self, url: str, model: str, temperature: float = 0.0, seed: int = 7, num_ctx: int = 8192,
                 max_output_tokens: int = 1024, timeout_s: float = 120.0) -> None:
        self.url, self.model = url.rstrip("/"), model
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": max_output_tokens}
        self.timeout_s = timeout_s

    @staticmethod
    def reachable(url: str, model: str, timeout_s: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout_s) as r:
                names = {m.get("name") for m in json.load(r).get("models", [])}
            return model in names or f"{model}:latest" in names
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def complete(self, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        sha = request_sha1(self.model, messages, schema)
        body = {"model": self.model, "messages": messages, "stream": True, "think": False, "options": self.options}
        if schema is not None:
            body["format"] = schema
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first, parts, last = None, [], {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                for line in r:
                    d = json.loads(line)
                    piece = d.get("message", {}).get("content", "")
                    if piece and first is None:
                        first = (time.perf_counter() - t0) * 1000.0
                    parts.append(piece)
                    if d.get("done"):
                        last = d
        except (OSError, ValueError, urllib.error.URLError) as e:
            return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha,
                               total_ms=(time.perf_counter() - t0) * 1000.0, error=f"{e.__class__.__name__}: {e}")
        return LLMResponse(ok=True, text="".join(parts), backend=self.name, model=self.model, request_sha1=sha,
                           ttft_ms=first, total_ms=(time.perf_counter() - t0) * 1000.0,
                           prompt_tokens=last.get("prompt_eval_count"), output_tokens=last.get("eval_count"),
                           meta={"load_ms": round(last.get("load_duration", 0) / 1e6, 3),
                                 "eval_ms": round(last.get("eval_duration", 0) / 1e6, 3),
                                 "done_reason": last.get("done_reason")})


class ScriptedBackend:
    """TEST USE ONLY: returns caller-supplied outputs (a list consumed in order, or a function of the request)."""

    name = "scripted"

    def __init__(self, outputs: list[str] | Callable[[list[dict], dict | None], str], model: str = "scripted") -> None:
        self.outputs, self.model = outputs, model
        self.requests: list[list[dict]] = []

    def complete(self, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        self.requests.append(messages)
        sha = request_sha1(self.model, messages, schema)
        if callable(self.outputs):
            text = self.outputs(messages, schema)
        elif self.outputs:
            text = self.outputs.pop(0)
        else:
            return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha, error="no output left")
        if text is None:
            return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha, error="scripted failure")
        return LLMResponse(ok=True, text=text, backend=self.name, model=self.model, request_sha1=sha, ttft_ms=0.0,
                           total_ms=0.0)


class RecordedBackend:
    """Replays LLM outputs recorded in LLM_CALL events (payload: request_sha1, output)."""

    name = "recorded"

    def __init__(self, records: list[dict], model: str = "recorded") -> None:
        self.by_sha: dict[str, list[dict]] = {}
        for r in records:
            self.by_sha.setdefault(r["request_sha1"], []).append(r)
        self.model = records[0]["model"] if records else model
        self.name = records[0]["backend"] if records else "recorded"     # replays look like the recorded run

    def complete(self, messages: list[dict], schema: dict | None = None) -> LLMResponse:
        sha = request_sha1(self.model, messages, schema)
        lst = self.by_sha.get(sha) or []
        if not lst:
            return LLMResponse(ok=False, backend=self.name, model=self.model, request_sha1=sha,
                               error="no recorded response for this request")
        r = lst.pop(0)
        return LLMResponse(ok=r["ok"], text=r.get("output") or "", backend=self.name, model=self.model,
                           request_sha1=sha, ttft_ms=r.get("ttft_ms"), total_ms=0.0,
                           prompt_tokens=r.get("prompt_tokens"), output_tokens=r.get("output_tokens"),
                           error=r.get("error"))


LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def check_loopback(url: str) -> None:
    """The LLM backend is local by design (ADR-017): any non-loopback URL is refused."""
    from urllib.parse import urlparse
    if urlparse(url).hostname not in LOOPBACK:
        raise ValueError(f"generation.ollama_url must be a loopback address (got {url}): the LLM backend is local by "
                         "design (ADR-017); no other network access is allowed")


def make_backend(gcfg):
    """ADR-007 selection: the configured LLM backend if reachable, else None (= extractive generation)."""
    if gcfg.backend == "extractive":
        return None
    if OllamaBackend.reachable(gcfg.ollama_url, gcfg.model):
        return OllamaBackend(gcfg.ollama_url, gcfg.model, gcfg.temperature, gcfg.seed, gcfg.num_ctx,
                             gcfg.max_output_tokens, gcfg.timeout_s)
    if gcfg.backend == "ollama":
        raise RuntimeError(f"generation.backend=ollama but model '{gcfg.model}' is not reachable at {gcfg.ollama_url}")
    return None
