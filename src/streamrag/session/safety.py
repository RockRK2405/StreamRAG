"""Memory safety (Phase 6; docs/session/10): what never enters session memory.

Transcript memory stores utterances after redaction of e-mail addresses, phone-like and card-like digit runs, and
key/token-like strings (long mixed letter+digit runs, "password/token/key: value"). Ordinary short numbers ("40",
"4 millimetres", "after 5 pm") are kept: they are task content. Archives keep only hashes and counts.
"""

from __future__ import annotations

import re

_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("secret", re.compile(r"(?i)\b(?:password|passcode|pin|token|api[_ -]?key|secret)\s*(?:is|:|=)\s*\S+")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone", re.compile(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)")),
    ("token", re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{24,}\b")),
]


def redact(text: str) -> tuple[str, int]:
    """Return (redacted text, number of redactions)."""
    n = 0
    for name, rx in _PATTERNS:
        text, k = rx.subn(f"[{name.upper()}]", text)
        n += k
    return text, n
