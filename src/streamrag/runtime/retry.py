"""Retries with bounded exponential backoff (docs/runtime/07).

Only transient failures are retried: timeouts, connection / network errors, temporary service errors and rate
limits. Deterministic failures (invalid input, schema violations, validation errors, bugs) fail immediately - a
retry would only repeat them. Jitter comes from a seeded RNG, so a virtual-clock run (and its replay) is
deterministic.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from streamrag.errors import RetrieverTimeoutError


class TransientError(Exception):
    """A failure worth retrying (raised by adapters and by fault injection)."""


class PermanentError(Exception):
    """A failure that must not be retried."""


_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (TransientError, TimeoutError, ConnectionError,
                                                     RetrieverTimeoutError)
_TRANSIENT_TEXT = re.compile(r"timed? ?out|timeout|connection|refused|reset by peer|unavailable|temporar|"
                             r"rate.?limit|\b429\b|\b502\b|\b503\b|\b504\b", re.I)


def is_transient(exc: BaseException | str | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, str):
        return bool(_TRANSIENT_TEXT.search(exc))
    if isinstance(exc, PermanentError):
        return False
    # network-library errors (urllib URLError etc.) without importing a network module here: the runtime itself
    # never opens a socket (tests/test_isolation.py); model calls go through the loopback-only generation package
    net = isinstance(exc, OSError) and type(exc).__module__.split(".")[0] in ("urllib", "http", "socket")
    return net or isinstance(exc, _TRANSIENT_TYPES)


class RetryManager:
    def __init__(self, rcfg_retry) -> None:
        self.cfg = rcfg_retry
        self.rng = random.Random(rcfg_retry.seed)
        self.stats: Counter = Counter()

    def should_retry(self, exc: BaseException | str | None, attempt: int) -> bool:
        """``attempt`` = attempts already made after the first (0 for the first failure)."""
        if not is_transient(exc):
            self.stats["not_retryable"] += 1
            return False
        if attempt >= self.cfg.max_retries:
            self.stats["gave_up"] += 1
            return False
        self.stats["retried"] += 1
        return True

    def delay_ms(self, attempt: int) -> float:
        """Delay before retry number ``attempt + 1``: initial * multiplier**attempt, capped, +- jitter."""
        base = min(self.cfg.max_delay_ms, self.cfg.initial_delay_ms * (self.cfg.multiplier ** attempt))
        j = self.cfg.jitter
        return max(0.0, base * (1.0 + self.rng.uniform(-j, j))) if j else base
