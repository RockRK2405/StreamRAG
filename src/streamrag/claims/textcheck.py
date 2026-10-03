"""Deterministic text checks used by claim verification (ADR-008 L1 and polarity signals).

* ``numbers`` - numbers, number words, ordinals and percentages, normalized ("five" -> "5", "first" -> "1",
  "4 millimetres" -> "4", "30%" -> "30%"). A number in a claim that is absent from its premise means the premise
  cannot entail the claim (REQ-GRD-007), whatever an entailment model says.
* ``negated`` - presence of a negation cue (signal only: paraphrases such as "prohibited" carry negation lexically).
"""

from __future__ import annotations

import re

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
          "dozen": 12}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?|[.,]\d+)(%|st|nd|rd|th)?(?![\w])", re.I)
_TOK = re.compile(r"[a-z]+", re.I)
NEGATION = frozenset({"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without"})


def numbers(text: str, ordinals: bool = True) -> set[str]:
    out = set()
    for m in _NUM.finditer(text):
        v = m.group(1)
        v = "0" + v if v[0] in ".," else v                  # ".4" is 0.4, not 4
        v = v.replace(",", ".") if v.count(",") == 1 and len(v.split(",")[1]) != 3 else v.replace(",", "")
        v = v.rstrip("0").rstrip(".") if "." in v else v
        out.add(v + ("%" if (m.group(2) or "").lower() == "%" else ""))
    for w in _TOK.findall(text.lower()):
        if w in _WORDS:
            out.add(str(_WORDS[w]))
        elif ordinals and w in _ORD:
            out.add(str(_ORD[w]))
    return out


_TIME_WORDS = frozenset({"january", "february", "march", "april", "may", "june", "july", "august", "september",
                         "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
                         "saturday", "sunday", "daily", "weekly", "monthly", "yearly", "annually", "every", "each",
                         "once", "twice", "hourly", "morning", "evening", "night", "noon", "midnight", "sunset",
                         "sunrise", "dawn", "dusk", "week", "weeks", "month", "months", "year", "years", "day", "days",
                         "hour", "hours", "minute", "minutes", "second", "seconds"})


def has_value(text: str) -> bool:
    """A number, ordinal, date or frequency expression (answers a how-long / how-much / when question)."""
    return bool(numbers(text)) or bool(_TIME_WORDS & set(_TOK.findall(text.lower())))


_NUMWORD = r"(?:\d+(?:[.,]\d+)?|" + "|".join(sorted(_WORDS, key=len, reverse=True)) + r"|a|an|one)"
_UNIT_TIME = r"(?:second|minute|hour|day|week|fortnight|month|year)s?"
_VALUE_PATTERNS = {
    "duration": re.compile(rf"\b{_NUMWORD}\s+(?:working\s+|business\s+|calendar\s+)?{_UNIT_TIME}\b(?!\s+old)", re.I),
    "amount": re.compile(r"(\d+(?:[.,]\d+)?\s*(%|percent|euros?|eur|dollars?|usd|rupees?|inr|pounds?|gbp|cents?)\b)"
                         r"|([€$£₹]\s*\d)|\bfree\b|\bno (?:fee|charge|cost)\b", re.I),
    "count": re.compile(rf"\b{_NUMWORD}\b", re.I),
    "frequency": re.compile(r"\b(every|each|daily|weekly|monthly|yearly|annually|hourly|once|twice|times|per)\b",
                            re.I),
    "time": re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                       r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|evening|night|noon|midnight|"
                       r"sunset|sunrise|dawn|dusk|week|month|year|season|\d{1,2}(:\d{2})?\s*(am|pm)|\d{4})\b", re.I),
    "age": re.compile(rf"\b{_NUMWORD}\s+years?\s+old\b|\b(?:at least|under|over|older than|younger than)\s+\d+", re.I),
    "measure": re.compile(r"\b\d+(?:[.,]\d+)?\s*(millimet(re|er)s?|mm|centimet(re|er)s?|cm|met(re|er)s?|m|"
                          r"kilomet(re|er)s?|km|grams?|g|kilograms?|kg|litres?|liters?|l|feet|foot|ft|inch(es)?|"
                          r"miles?|high|tall|wide|deep|long)\b", re.I),
}


def has_value_of(text: str, kind: str) -> bool:
    """Does the text state a value of the asked kind (a duration for "how long", an amount for "how much", ...)?"""
    rx = _VALUE_PATTERNS.get(kind)
    return bool(rx.search(text)) if rx else has_value(text)


# Prompt-injection markers (docs/answer/02 §Injection): sentences that address the model or give it instructions are
# DATA, never facts - they are excluded from claim selection and from verification premises. A heuristic first line of
# defence: verification and schema-bound output remain the guarantee.
_INSTRUCTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.]{0,40}\b(previous|prior|above|earlier|all|system|your)\b[^.]{0,20}"
    r"\b(instructions?|prompts?|rules|messages?)\b"
    r"|\bsystem\s+(note|prompt|message|instruction)s?\b|\b(note|message|instructions?)\s+to\s+the\s+(assistant|model|ai)\b"
    r"|\b(assistant|chatbot|language model|ai model)\s*:|\byou are (now )?(an? |the )?(assistant|ai|model|chatbot)\b"
    r"|\b(tell|inform|say to|answer) the user\b|\bdo not (mention|reveal|disclose) (this|these)\b"
    r"|\bas an ai\b|<<<|>>>|\[/?inst\]|<\|?(system|im_start|im_end)\|?>", re.I)


_COUNT = re.compile(r"\b" + _NUMWORD.replace("|a|an|", "|") + r"\b((?:\s+[\w-]+){1,3})", re.I)


def counts_noun(text: str, counted: set[str], terms_fn) -> bool:
    """A number followed within 3 words by the counted noun ("40 crates", "ten working days"): the answer to
    "how many <noun>" must count that noun, not any number."""
    return any(counted & set(terms_fn(m.group(1))) for m in _COUNT.finditer(text))


def instruction_like(text: str) -> bool:
    """True for text that addresses the model or tries to instruct it (prompt-injection markers)."""
    return bool(_INSTRUCTION.search(text))


_MARKER = re.compile(r"\s*[\[(]\s*((?:E\d+|F\d+)(?:\s*[,;/]\s*(?:E\d+|F\d+))*)\s*[\])]")


def strip_markers(text: str) -> tuple[str, list[str]]:
    """Remove inline citation / fact markers ("(E1)", "[E2, E3]") from generated text; return them separately."""
    found = [x.strip() for m in _MARKER.finditer(text) for x in re.split(r"[,;/]", m.group(1))]
    clean = re.sub(r"\s+([.,;:!?])", r"\1", _MARKER.sub("", text)).strip()
    return clean, found


def negated(text: str) -> bool:
    low = text.lower()
    return bool(NEGATION & set(_TOK.findall(low))) or "n't" in low
