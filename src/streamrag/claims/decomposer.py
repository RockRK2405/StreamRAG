"""ClaimDecomposer: compound statement -> atomic claims (Phase 7; docs/answer/03).

Rule-based, deterministic, lexicon-driven (configs/claim_lexicon.yaml: grammar words only):

1. **Clauses.** Split at ";" and at a coordinator (and / but / yet) whose right side starts a new predicate:
   - its first token is an auxiliary, modal or listed finite verb (subject elided: the left clause's subject is
     borrowed, "Ladders are not permitted overnight and must be returned" -> "... Ladders must be returned");
   - or such a token occurs within its first 3 tokens with no coordinator before it (own subject,
     "... and applications take 30 days");
   - or its first token is a participle ("-ed") after a passive left clause (subject + auxiliary borrowed,
     "are submitted online and processed within 7 days" -> "are processed within 7 days").
   "while" / "whereas" are not split: they usually subordinate a condition.
2. **Coordinated lists.** Inside a clause, "P A, B and C S" whose items are short (<= 6 tokens) and contain no
   predicate becomes "P A S", "P B S", "P C S". Not split: "or" lists (disjunction), lists after a pair opener
   ("between the third and fifth week", "both A and B").

The decomposer is a heuristic. Callers guard it: an atom derived from evidence is used only if the entailment
check accepts it as entailed by its source sentence (claims/planner.py); a generated statement is decomposed only
when the whole statement is not supported (claims/verifier.py), so a bad split can only remove content, never add
an unsupported fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_TOK = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


@dataclass(frozen=True)
class ClaimLexicon:
    aux: frozenset[str]
    verbs: frozenset[str]
    coordinators: frozenset[str]
    pair_openers: frozenset[str]
    value_questions: tuple[tuple[str, tuple[str, ...]], ...] = ()      # (kind, phrase words)

    @classmethod
    def load(cls, path: Path) -> "ClaimLexicon":
        d = yaml.safe_load(Path(path).read_text()) or {}
        low = lambda k: frozenset(str(w).lower() for w in d.get(k, []))  # noqa: E731
        return cls(aux=low("auxiliaries") | low("modals"), verbs=low("clause_verbs"),
                   coordinators=low("coordinators") - {"while", "whereas"}, pair_openers=low("pair_openers"),
                   value_questions=tuple((k, tuple(str(p).lower().split()))
                                         for k, lst in (d.get("value_questions") or {}).items() for p in lst))

    def asks_value(self, text: str) -> str | None:
        """The kind of value the need asks for (duration, amount, count, ...), or None."""
        words = re.findall(r"[a-z]+", text.lower())
        for kind, p in self.value_questions:
            if any(tuple(words[i:i + len(p)]) == p for i in range(len(words))):
                return kind
            # "what / which <modifier> date": one modifier between the wh-determiner and the noun
            if len(p) == 2 and p[0] in ("what", "which") and any(
                    words[i] == p[0] and words[i + 2] == p[1] for i in range(len(words) - 2)):
                return kind
        return None

    def predicate(self, w: str) -> bool:
        return w in self.aux or w in self.verbs


_PREP = frozenset({"with", "to", "of", "for", "in", "on", "at", "by", "from", "about", "under", "over", "into", "as",
                   "including", "like", "before", "after", "during", "until", "within", "per", "without", "through",
                   "across", "against", "via", "unless", "if", "when"})
_SKIP = frozenset({"not", "also", "only", "always", "never", "usually", "often", "be", "been", "being", "then"})


class ClaimDecomposer:
    def __init__(self, lexicon: ClaimLexicon, max_item_tokens: int = 6) -> None:
        self.lx, self.max_item = lexicon, max_item_tokens

    def decompose(self, statement: str) -> list[str]:
        """Atomic claims of one statement (the statement itself when it is already atomic)."""
        text = statement.strip()
        if not text:
            return []
        out: list[str] = []
        for clause in self._clauses(text):
            out += self._distribute(clause)
        out = [_finish(a) for a in out if _content(a)]
        return out or [_finish(text)]

    # ------------------------------------------------------------------ clauses
    def _clauses(self, text: str) -> list[str]:
        toks = [(m.group(0), m.start(), m.end()) for m in _TOK.finditer(text)]
        low = [t[0].lower() for t in toks]
        cuts: list[tuple[int, str]] = []                     # (token index of the cut, kind)
        for i, w in enumerate(low):
            if w == ";":
                cuts.append((i, "own"))
            elif w in self.lx.coordinators and 0 < i < len(low) - 1:
                left = low[:i]
                if not any(self.lx.predicate(x) for x in left) or any(x in self.lx.pair_openers for x in left[-3:]):
                    continue
                right = [x for x in low[i + 1:] if x != ","]
                if not right:
                    continue
                if self.lx.predicate(right[0]) or (right[0] in _SKIP and len(right) > 1 and
                                                   self.lx.predicate(right[1])):
                    cuts.append((i, "elided"))
                elif right[0].endswith("ed") and any(x in self.lx.aux for x in left):
                    cuts.append((i, "participle"))
                else:
                    head = right[:3]
                    k = next((j for j, x in enumerate(head) if self.lx.predicate(x)), None)
                    if k is not None and k > 0 and not any(x in self.lx.coordinators for x in head[:k]):
                        cuts.append((i, "own"))
        if not cuts:
            return [text]
        pieces, start, kinds = [], 0, ["own"]
        for i, kind in cuts:
            pieces.append(text[toks[start][1]:toks[i][1]].strip(" ,;"))
            start = i + 1
            kinds.append(kind)
        pieces.append(text[toks[start][1]:].strip(" ,;") if start < len(toks) else "")
        out = []
        first = pieces[0]
        f_toks = [(m.group(0).lower(), m.start()) for m in _TOK.finditer(first)]
        v = next((j for j, (w, _) in enumerate(f_toks) if self.lx.predicate(w)), None)
        subject = first[:f_toks[v][1]].strip() if v else ""
        aux = f_toks[v][0] if v is not None and f_toks[v][0] in self.lx.aux else ""
        for piece, kind in zip(pieces, kinds):
            if not piece:
                continue
            if kind == "elided" and subject:
                piece = f"{subject} {piece}"
            elif kind == "participle" and subject and aux:
                piece = f"{subject} {aux} {piece}"
            out.append(piece)
        return out

    # ------------------------------------------------------------------ coordinated lists
    def _distribute(self, clause: str) -> list[str]:
        toks = [(m.group(0), m.start(), m.end()) for m in _TOK.finditer(clause)]
        low = [t[0].lower() for t in toks]
        if "and" not in low:
            return [clause]
        a = low.index("and")
        if any(x in self.lx.pair_openers for x in low[max(0, a - 4):a]) or "or" in low:
            return [clause]
        # separators of the list: commas before "and" back to the first item, and the "and" itself
        seps = [a]
        j = a - 1
        while j >= 0:
            if low[j] == ",":
                seps.insert(0, j)
            if self.lx.predicate(low[j]) or low[j] in _PREP or low[j] == ":":
                break
            j -= 1
        # first item: starts after the last predicate (skipping the verb after a modal: "must submit A") or after the
        # last preposition / colon; of the two, the one whose length best matches the other items wins (parallel
        # structure: "provide [proof of residence] and [a passport photo]", "logged with [the time], [the weather]")
        cands = []
        b = seps[0] - 1
        while b >= 0 and not (self.lx.predicate(low[b]) or low[b] == ":"):
            b -= 1
        if b >= 0:
            st = b + 1
            if low[b] in self.lx.aux:
                while st < seps[0] and low[st] in _SKIP:
                    st += 1
                st += 1
            cands.append(st)
        b = seps[0] - 1
        while b >= 0 and not (low[b] in _PREP or low[b] == ":" or self.lx.predicate(low[b])):
            b -= 1
        if b >= 0 and low[b] in _PREP:
            cands.append(b + 1)
        cands = [c for c in cands if c < seps[0]]
        if not cands:
            return [clause]
        others = [seps[k + 1] - seps[k] - 1 for k in range(len(seps) - 1)]
        tail = 0
        for k in range(seps[-1] + 1, len(toks)):
            if low[k] in (".", ",", ";") or self.lx.predicate(low[k]) or low[k] in _PREP:
                break
            tail += 1
        others.append(tail)
        mean = sum(others) / len(others)
        start = min(cands, key=lambda c: (abs((seps[0] - c) - mean), -c))
        bounds = [start] + [s + 1 for s in seps]
        ends = [s for s in seps] + [None]
        # the last item ends at the next boundary-free run: stop at a predicate / clause punctuation
        last_end = len(toks)
        for k in range(bounds[-1], len(toks)):
            if low[k] in (".", ",", ";") or self.lx.predicate(low[k]) or low[k] in _PREP:
                last_end = k
                break
        ends[-1] = last_end
        items = [(bounds[k], ends[k]) for k in range(len(bounds))]
        items = [(s, e) for s, e in items if e > s and not (low[s] == "," and e - s == 1)]
        if len(items) < 2 or any(e - s > self.max_item for s, e in items) \
                or any(self.lx.predicate(x) for s, e in items for x in low[s:e]):
            return [clause]
        prefix = clause[:toks[items[0][0]][1]].rstrip()
        suffix = clause[toks[last_end][1]:].strip() if last_end < len(toks) else ""
        out = []
        for s, e in items:
            item = clause[toks[s][1]:toks[e - 1][2]].strip(" ,")
            if item.lower().startswith("and "):
                item = item[4:]
            out.append(" ".join(x for x in (prefix, item, suffix) if x))
        return out


def _content(s: str) -> bool:
    return sum(1 for m in _TOK.finditer(s) if m.group(0).isalnum()) >= 2


def _finish(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s
