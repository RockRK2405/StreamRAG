"""Typed value extraction for conflict detection (docs/retrieval/09).

Conservative patterns: a value is a number with a unit of the asked kind (amount: currency; duration: time unit;
time: clock time; age: years old / at least N; count / measure: number + word). Values are normalised so that
"55 euros" and "EUR 55" compare equal. Used only to *detect* that two sources state different values for the same
requirement; deciding which one applies is the resolver's job (temporal validity, version, authority).
"""

from __future__ import annotations

import re

_NUM = r"\d+(?:[.,]\d+)?"
_CUR = r"(?:euros?|eur|€|dollars?|usd|\$|pounds?|gbp|£)"
_AMOUNT = re.compile(rf"(?:({_NUM})\s*{_CUR}|{_CUR}\s*({_NUM}))", re.I)
_DUR = re.compile(rf"({_NUM})\s*(?:working\s+|business\s+|calendar\s+)?(seconds?|minutes?|hours?|days?|weeks?|months?|"
                  r"years?)\b", re.I)
_TIME = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b|\b(\d{1,2})\s*(am|pm)\b", re.I)
_AGE = re.compile(rf"\b({_NUM})\s+years?\s+old\b|\b(?:at least|under|over|older than|younger than)\s+({_NUM})\b", re.I)
_MEASURE = re.compile(rf"\b({_NUM})\s*(millimet(?:re|er)s?|mm|centimet(?:re|er)s?|cm|met(?:re|er)s?|kilomet(?:re|er)s?|"
                      r"km|grams?|kg|kilograms?|litres?|liters?|percent|%)", re.I)
_COUNT = re.compile(rf"\b({_NUM})\b")
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _n(x: str) -> str:
    x = x.replace(",", ".")
    return x[:-2] if x.endswith(".0") else x


def values_of(text: str, kind: str | None) -> list[str]:
    """Normalised values of the asked kind in ``text`` (order of appearance, unique)."""
    out: list[str] = []
    if kind == "amount":
        out = [f"{_n(a or b)} cur" for a, b in _AMOUNT.findall(text)]
    elif kind == "duration":
        out = [f"{_n(n)} {u.lower().rstrip('s')}" for n, u in _DUR.findall(text)]
    elif kind == "time":
        for h, m, ap, h2, ap2 in _TIME.findall(text):
            hh, mm, suf = (h, m, ap) if h else (h2, "00", ap2)
            hv = int(hh) % 12 + (12 if suf.lower() == "pm" else 0) if suf else int(hh)
            out.append(f"{hv:02d}:{mm}")
    elif kind == "age":
        out = [f"{_n(a or b)} y" for a, b in _AGE.findall(text)]
    elif kind == "measure":
        out = [f"{_n(n)} {u.lower()[:2]}" for n, u in _MEASURE.findall(text)]
    elif kind in ("count", "frequency"):
        out = [_n(n) for n in _COUNT.findall(text)]
    return list(dict.fromkeys(out))
