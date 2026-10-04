"""Phase 11 demo server: the frozen final pipeline behind a small HTTP API (health / readiness, sessions, SSE events)
and a static live-demo UI. Standard library only (asyncio streams)."""

from streamrag.server.app import DemoApp, serve

__all__ = ["DemoApp", "serve"]
