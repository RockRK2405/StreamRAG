"""Minimal HTTP/1.1 on asyncio streams (stdlib only) for the demo server: one request per connection, bounded sizes,
JSON / static / Server-Sent-Events responses. Not a general web framework - just enough for a local demo API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlsplit

MAX_HEADER_BYTES = 16 * 1024
MAX_BODY_BYTES = 64 * 1024
READ_TIMEOUT_S = 10.0
REASONS = {200: "OK", 201: "Created", 204: "No Content", 400: "Bad Request", 404: "Not Found",
           405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large", 429: "Too Many Requests",
           500: "Internal Server Error", 503: "Service Unavailable"}
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                    "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; "
                                               "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"}


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes = b""
    params: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        if not self.body:
            return {}
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(400, f"invalid JSON body: {exc.__class__.__name__}") from exc
        if not isinstance(data, dict):
            raise HttpError(400, "JSON body must be an object")
        return data


async def read_request(reader: asyncio.StreamReader) -> Request:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), READ_TIMEOUT_S)
    except asyncio.LimitOverrunError as exc:
        raise HttpError(413, "request headers too large") from exc
    except (asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
        raise ConnectionError("incomplete request") from exc
    if len(head) > MAX_HEADER_BYTES:
        raise HttpError(413, "request headers too large")
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _ = lines[0].split(" ", 2)
    except ValueError as exc:
        raise HttpError(400, "malformed request line") from exc
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0") or 0)
    if n < 0 or n > MAX_BODY_BYTES:
        raise HttpError(413, "request body too large")
    body = await asyncio.wait_for(reader.readexactly(n), READ_TIMEOUT_S) if n else b""
    u = urlsplit(target)
    return Request(method.upper(), unquote(u.path), parse_qs(u.query), headers, body)


def _head(status: int, ctype: str, length: int | None, extra: dict | None = None) -> bytes:
    hdrs = {"Content-Type": ctype, "Cache-Control": "no-store", "Connection": "close", **SECURITY_HEADERS,
            **(extra or {})}
    if length is not None:
        hdrs["Content-Length"] = str(length)
    lines = [f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}"] + [f"{k}: {v}" for k, v in hdrs.items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


async def send(writer: asyncio.StreamWriter, status: int, body: bytes, ctype: str) -> None:
    writer.write(_head(status, ctype, len(body)) + body)
    await writer.drain()


async def send_json(writer: asyncio.StreamWriter, status: int, data) -> None:
    await send(writer, status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
               "application/json; charset=utf-8")


async def start_sse(writer: asyncio.StreamWriter) -> None:
    writer.write(_head(200, "text/event-stream; charset=utf-8", None, {"X-Accel-Buffering": "no"}))
    await writer.drain()


async def sse_event(writer: asyncio.StreamWriter, data: dict) -> None:
    writer.write(f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
    await writer.drain()


async def sse_comment(writer: asyncio.StreamWriter, text: str = "ping") -> None:
    writer.write(f": {text}\n\n".encode("utf-8"))
    await writer.drain()
