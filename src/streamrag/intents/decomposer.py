"""Rule-first intent decomposition (spec §9.3, Phase 5; docs/multi_intent/02-03).

    transcript -> tokens (+punctuation) -> clause segmentation -> clause roles
               -> request clauses: guarded coordination split -> draft intents
               -> constraint clauses: scope assignment (global | local)
               -> correction clauses: supersede a target intent
               -> anaphora / ellipsis resolution (context carry-over, never invented words)
               -> dedup + refinement relations -> type, entities, confidence, priority -> budget

Definition used throughout: an intent is one retrieval-relevant information need. A clause is only a source of
intents if it carries a request cue (question word, auxiliary inversion, request head, imperative request verb,
or an explicit addition marker after a request). Narrative clauses, social talk and constraint phrases never
become intents (over-decomposition guard); coordinated needs under one request become sibling intents only when
every conjunct carries substantive content and the pair is not a fixed corpus phrase (under/over-decomposition).

All confidences and priorities are computed from the documented formulas below; nothing is authored.

  confidence     = 0.35*explicit + 0.30*specificity + 0.20*separation + 0.15*resolution
  priority_score = 0.35*confidence + 0.25*specificity + 0.15*explicit + 0.10*has_constraints
                   + 0.10*novelty + 0.05*has_dependents
  decomposition_confidence = mean(intent confidence) * min(constraint scope_confidence, 1)

  explicit    1.0 own request cue | 0.8 shares a head (coordination) / implicit addition | 0.5 fallback
  specificity anchor strength: max corpus IDF of the intent's terms / IDF of a single-chunk term (capped at 1)
  separation  1.0 clause boundary or single intent | 0.8 coordination split | 0.6 fallback
  resolution  1.0 self-contained | 0.85 relies on inherited context | 0.7 unresolved reference
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from streamrag.controller.query_builder import QueryBuilder
from streamrag.intents.lexicon import IntentLexicon
from streamrag.intents.text import HARD_PUNCT, Tok, match_phrase, tokenize

_PUNCT_STRIP = re.compile(r"^[\s,;:.?!&]+|[\s,;:.?!&]+$")


# ---------------------------------------------------------------------------------------------- data classes
@dataclass
class Segment:
    """A verbatim piece of an intent's resolved text, traceable to a transcript span."""

    text: str
    utterance_id: str
    start: int
    end: int
    source: str = "intent"            # intent | inherited
    ref: str | None = None            # draft key / intent id the inherited words come from
    reason: str | None = None         # inheritance reason
    replaces: str | None = None


@dataclass
class DraftIntent:
    key: str
    clause_index: int
    split: str                        # single | clause | coordination | correction | fallback
    tok_idx: list[int]                # word-token indices (into the utterance tokens) forming the intent
    text: str
    start: int
    end: int
    explicit: float
    separation: float
    segments: list[Segment] = field(default_factory=list)
    inherited: list[Segment] = field(default_factory=list)   # non-anaphora context appended to the query
    unresolved: list[str] = field(default_factory=list)
    intent_type: str = "OTHER"
    type_cues: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topic: tuple[str, int, int, str] | None = None           # (text, start, end, utterance_id)
    aspect: str | None = None
    terms: list[str] = field(default_factory=list)
    anchor_strength: float = 0.0
    confidence: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    priority_score: float = 0.0
    priority: int = 1
    order: int = 0
    status: str = "ACTIVE"            # ACTIVE | SUPERSEDED | MERGED | DROPPED
    supersedes: str | None = None     # draft key or prior intent id
    superseded_by: str | None = None
    correction_cue: str | None = None
    constraint_keys: list[str] = field(default_factory=list)
    replace_in: "DraftIntent | PriorIntent | None" = None   # correction: target whose words are replaced
    replace_words: list[str] = field(default_factory=list)
    aspect_from: "DraftIntent | PriorIntent | None" = None   # correction of a bare topic keeps the target's aspect
    origin: str = "rule"              # rule | rule_fallback | llm (per-intent provenance)

    @property
    def from_llm(self) -> bool:
        return self.origin == "llm"

    @property
    def resolved_text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass
class DraftConstraint:
    key: str
    kind: str                         # focus | condition | restriction
    marker: str | None
    text: str
    start: int
    end: int
    clause_index: int
    scope: str = "global"
    applies_to: list[str] = field(default_factory=list)
    scope_reason: str = ""
    scope_confidence: float = 1.0
    terms: list[str] = field(default_factory=list)
    pinned: bool = False              # scope fixed by construction (trailing PP after a coordination)
    scope_all: bool = False           # explicit "for both / for all of them"
    has_question: bool = False        # "specifically how high ..." refines the previous need
    preposition: bool = False         # opens with a restriction preposition ("for category z")


@dataclass
class DraftRelation:
    type: str
    source: str
    target: str
    cue: str


@dataclass
class Clause:
    index: int
    a: int                            # token range [a, b)
    b: int
    opened_by: str                    # start | punct | comma | coordinator | addition | question | marker | head | correction
    role: str = "CONTEXT"             # REQUEST | CONSTRAINT | CORRECTION | CONTEXT | IGNORE
    link: str | None = None
    head: str | None = None
    marker: str | None = None
    kind: str | None = None           # constraint kind
    body: list[int] = field(default_factory=list)   # word token indices after link/head/marker
    implicit: bool = False            # elliptical request: a facet noun phrase without its request head


@dataclass
class PriorIntent:
    """An active intent from an earlier utterance of the session (context carry-over / corrections)."""

    intent_id: str
    utterance_id: str
    text: str
    terms: list[str]
    topic: tuple[str, int, int, str] | None              # (text, start, end, utterance_id)
    aspect: str | None
    aspect_span: tuple[int, int] | None = None
    segments: list["Segment"] = field(default_factory=list)   # its resolved words, span-traceable

    @property
    def key(self) -> str:
        return self.intent_id


@dataclass
class DecompositionContext:
    prior: list[PriorIntent] = field(default_factory=list)       # all earlier active intents, most recent last
    recent: list[PriorIntent] = field(default_factory=list)      # those of the most recent utterance (anaphora)
    prior_query_terms: list[list[str]] = field(default_factory=list)


@dataclass
class Decomposition:
    utterance_id: str
    transcript: str
    source: str                       # rule | rule_fallback
    intents: list[DraftIntent]        # every draft (ACTIVE and not), in order of mention
    constraints: list[DraftConstraint]
    relations: list[DraftRelation]
    merged: list[tuple[str, str, str]]          # (text, into_key, reason)
    dropped: list[tuple[str, str | None, str]]  # (text, key, reason)
    clauses: list[Clause]
    confidence: float
    external_supersessions: list[tuple[str, str, str]] = field(default_factory=list)  # (prior_id, new_key, cue)
    ctx: DecompositionContext | None = None

    @property
    def active(self) -> list[DraftIntent]:
        return [i for i in self.intents if i.status == "ACTIVE"]


# ---------------------------------------------------------------------------------------------- decomposer
class IntentDecomposer:
    def __init__(self, lex: IntentLexicon, stopwords: frozenset[str], terms_fn: Callable[[str], list[str]],
                 idf_fn: Callable[[str], float | None], anchor_norm: float, anchor_floor: float = 1.0,
                 corpus_pairs: frozenset[tuple[str, str]] = frozenset(), max_intents: int = 4,
                 duplicate_jaccard: float = 0.8, carryover: bool = True) -> None:
        self.lx, self.stop = lex, stopwords
        self.terms_fn, self.idf_fn = terms_fn, idf_fn
        self.anchor_norm, self.anchor_floor = anchor_norm or 1.0, anchor_floor
        self.corpus_pairs = corpus_pairs
        self.max_intents, self.dup_j, self.carryover = max_intents, duplicate_jaccard, carryover
        self.fallback_builder = QueryBuilder(lex.base)
        b = lex.base
        self._non_content = (stopwords | b.question_words | b.aux_questions | b.fillers | lex.generic_nouns
                             | lex.anaphora | lex.locative_anaphora | lex.request_verbs | b.social | b.backchannel
                             | lex.coordinators | lex.alternatives | {"please", "me", "us", "i", "we", "you", "also"})

    # ------------------------------------------------------------------ public
    def decompose(self, transcript: str, utterance_id: str, ctx: DecompositionContext | None = None) -> Decomposition:
        ctx = ctx or DecompositionContext()
        toks = tokenize(transcript)
        lows = [t.lower for t in toks]
        drop = self._filler_mask(lows)
        clauses = self._segment(toks, lows, drop)
        for c in clauses:
            self._assign_role(c, toks, lows, drop, clauses)
        intents: list[DraftIntent] = []
        constraints: list[DraftConstraint] = []
        relations: list[DraftRelation] = []
        merged: list[tuple[str, str, str]] = []
        dropped: list[tuple[str, str | None, str]] = []
        ext_sup: list[tuple[str, str, str]] = []
        n_key = [0]

        def new_key() -> str:
            n_key[0] += 1
            return f"d{n_key[0]}"

        for c in clauses:
            if c.role == "REQUEST":
                intents += self._request_intents(c, toks, lows, utterance_id, transcript, constraints, new_key)
            elif c.role == "CONSTRAINT" and any(self._is_content(lows[j]) for j in c.body):
                # (a marker without content - "especially during" mid-stream - is not a constraint yet)
                constraints.append(self._constraint(c, toks, lows, transcript, f"k{len(constraints) + 1}"))
            elif c.role == "CORRECTION" and c.body:
                d = self._correction(c, toks, lows, utterance_id, transcript, intents, ctx, new_key, ext_sup)
                if d is not None:
                    intents.append(d)
        source = "rule"
        pending_correction = any(c.role == "CORRECTION" for c in clauses)   # "actually I meant ..." (no content yet)
        if not [i for i in intents if i.status == "ACTIVE"] and not pending_correction:
            fb = self._fallback(transcript, toks, utterance_id, new_key)
            if fb is not None:
                intents.append(fb)
                source = "rule_fallback"
                if constraints:
                    dropped.append(("; ".join(k.text for k in constraints), None,
                                    "constraint_without_intent_in_utterance (late-detail refinement: Phase 6)"))
                    constraints = []
        for i in intents:
            self._annotate(i, toks, lows, utterance_id)
        if self.carryover:
            self._resolve_references(intents, toks, lows, utterance_id, transcript, ctx, relations)
        self._scope_constraints(constraints, intents, relations)
        for i in intents:
            self._finalize_terms(i)
        self._dedup_and_refine(intents, merged, relations)
        for order, i in enumerate([i for i in intents if i.status == "ACTIVE"]):
            i.order = order
        self._score(intents, constraints, relations, ctx)
        self._budget(intents, dropped)
        active = [i for i in intents if i.status == "ACTIVE"]
        scope_conf = min([k.scope_confidence for k in constraints if k.applies_to] or [1.0])
        conf = (sum(i.confidence for i in active) / len(active) * scope_conf) if active else 0.0
        return Decomposition(utterance_id, transcript, source, intents, constraints, relations, merged, dropped,
                             clauses, round(conf, 3), ext_sup, ctx)

    def decompose_with_extra_spans(self, transcript: str, utterance_id: str, spans: list[tuple[int, int]],
                                   base: Decomposition) -> Decomposition:
        """Add validated, transcript-grounded LLM needs (char spans) to a rule decomposition; rule intents are kept
        (duplicates of them merge away). Provenance of added intents is 'llm', of the set 'reconciled'."""
        toks = tokenize(transcript)
        lows = [t.lower for t in toks]
        ctx = base.ctx or DecompositionContext()
        n0 = len(base.intents)
        extra = []
        for k, (s, e) in enumerate(spans):
            idx = [j for j, t in enumerate(toks) if not t.punct and s <= t.start and t.end <= e]
            idx = [j for j in idx if self._is_content(lows[j]) or lows[j] in self.lx.base.question_words
                   or lows[j] not in self.lx.base.fillers]
            if not idx:
                continue
            d = self._draft(f"d{n0 + k + 1}", 0, "clause", idx, toks, transcript, 0.8, 0.8)
            d.origin = "llm"
            self._annotate(d, toks, lows, utterance_id)
            extra.append(d)
        intents = list(base.intents) + extra
        relations = list(base.relations)
        merged = list(base.merged)
        for i in extra:
            self._finalize_terms(i)
        self._dedup_and_refine(intents, merged, relations)
        for order, i in enumerate(sorted([i for i in intents if i.status == "ACTIVE"], key=lambda x: x.start)):
            i.order = order
        self._score(intents, base.constraints, relations, ctx)
        dropped = list(base.dropped)
        self._budget(intents, dropped)
        active = [i for i in intents if i.status == "ACTIVE"]
        conf = sum(i.confidence for i in active) / len(active) if active else 0.0
        return Decomposition(utterance_id, transcript, "reconciled", intents, base.constraints, relations, merged,
                             dropped, base.clauses, round(conf, 3), base.external_supersessions, ctx)

    # ------------------------------------------------------------------ token classes
    def _filler_mask(self, lows: list[str]) -> list[bool]:
        """Fillers are dropped everywhere, except words that are part of a correction phrase ('i mean')."""
        lx, b = self.lx, self.lx.base
        drop = [False] * len(lows)
        protected = [False] * len(lows)
        for i in range(len(lows)):
            n = match_phrase(lows, i, lx.correction_phrases)
            for k in range(i, i + n):
                protected[k] = True
        for i in range(len(lows)):
            n = match_phrase(lows, i, b.filler_phrases)
            if n and not any(protected[i:i + n]):
                for k in range(i, i + n):
                    drop[k] = True
        for i, w in enumerate(lows):
            if w in b.fillers and not protected[i] and w not in b.negations:
                drop[i] = True
        return drop

    def _is_content(self, w: str) -> bool:
        return bool(w) and w[0].isalnum() and w not in self._non_content

    def _next_word(self, toks: list[Tok], drop: list[bool], i: int, b: int | None = None) -> int | None:
        b = len(toks) if b is None else b
        for j in range(i, b):
            if not drop[j] and not toks[j].punct:
                return j
        return None

    def _prev_word(self, toks: list[Tok], drop: list[bool], i: int, a: int = 0) -> int | None:
        for j in range(i - 1, a - 1, -1):
            if not drop[j] and not toks[j].punct:
                return j
        return None

    def _opens_clause(self, lows: list[str], j: int, after_comma: bool) -> str | None:
        """Does the word at j open a new clause? Returns the reason."""
        lx, b = self.lx, self.lx.base
        w = lows[j]
        if match_phrase(lows, j, lx.correction_phrases):
            return "correction"
        if match_phrase(lows, j, lx.focus_phrases) or match_phrase(lows, j, lx.condition_phrases):
            return "marker"
        if match_phrase(lows, j, lx.scope_all_phrases):
            return "marker"
        if match_phrase(lows, j, b.request_heads):
            return "head"
        if w in b.question_words or w in b.aux_questions or w in lx.request_verbs:
            return "question"
        if match_phrase(lows, j, lx.addition_phrases):
            return "addition"
        if after_comma and (w in lx.restriction_prepositions or w in lx.topic_prepositions):
            return "marker"
        if after_comma and w in lx.coordinators:
            k = j + 1
            if k < len(lows) and self._opens_clause(lows, k, False) in ("question", "head", "correction"):
                return "coordinator"
        return None

    # ------------------------------------------------------------------ segmentation
    def _segment(self, toks: list[Tok], lows: list[str], drop: list[bool]) -> list[Clause]:
        lx, b = self.lx, self.lx.base
        cuts: list[tuple[int, str]] = []       # (token index where a new clause starts, opened_by)
        start = 0
        marker_end = -1                       # first token after a correction / constraint marker phrase
        i = 0
        n = len(toks)
        while i < n:
            t = toks[i]
            if drop[i]:
                i += 1
                continue
            if t.punct:
                if t.text in HARD_PUNCT:
                    cuts.append((i + 1, "punct"))
                    start = i + 1
                elif t.text in (",", ":"):
                    j = self._next_word(toks, drop, i + 1)
                    if j is not None and self._opens_clause(lows, j, True):
                        cuts.append((i + 1, "comma"))
                        start = i + 1
                i += 1
                continue
            first = self._next_word(toks, drop, start)
            if i == first:
                i += 1
                continue
            prev = self._prev_word(toks, drop, i, start)
            prev_w = lows[prev] if prev is not None else ""
            only_links = all(drop[k] or toks[k].punct or lows[k] in lx.coordinators
                             or any(lows[k] in ph for ph in lx.addition_phrases) for k in range(start, i))
            after_marker = i == marker_end    # "no wait HOW ...", "specifically HOW ...": stays in that clause
            cut, skip = None, 1
            n_corr = match_phrase(lows, i, lx.correction_phrases)
            n_mark = match_phrase(lows, i, lx.focus_phrases) or match_phrase(lows, i, lx.condition_phrases)
            n_all = match_phrase(lows, i, lx.scope_all_phrases)
            n_add = match_phrase(lows, i, lx.addition_phrases)
            n_head = match_phrase(lows, i, b.request_heads)
            if n_corr:
                cut, skip = "correction", n_corr
            elif n_mark and not self._embedded_condition(lows, i, prev_w):
                cut, skip = "marker", n_mark
            elif n_all:
                cut, skip = "marker", n_all
            elif lows[i] in lx.coordinators:
                j = self._next_word(toks, drop, i + 1)
                if j is not None and self._opens_clause(lows, j, False) in ("question", "head", "correction"):
                    cut = "coordinator"
                elif n_add:
                    cut, skip = "addition", n_add
            elif n_add and not only_links:
                cut, skip = "addition", n_add
            elif n_head and not only_links and not after_marker and prev_w not in lx.embedding_verbs:
                cut, skip = "head", n_head
            elif not only_links and not after_marker and self._question_restart(lows, i, prev_w):
                cut = "question"
            if cut is not None:
                cuts.append((i, cut))
                start = i
                if cut in ("correction", "marker"):
                    marker_end = self._next_word(toks, drop, i + skip) or -1
            i += max(skip, 1)
        bounds = [(0, "start")] + [(c, why) for c, why in cuts if 0 < c <= n]
        dedup: list[tuple[int, str]] = []
        for c, why in bounds:
            if dedup and dedup[-1][0] == c:
                continue
            dedup.append((c, why))
        clauses = []
        for k, (a, why) in enumerate(dedup):
            bnd = dedup[k + 1][0] if k + 1 < len(dedup) else n
            if any(not drop[j] and not toks[j].punct for j in range(a, bnd)):
                clauses.append(Clause(len(clauses), a, bnd, why))
        return clauses

    def _embedded_condition(self, lows: list[str], i: int, prev_w: str) -> bool:
        """'ask if', 'know whether', 'wondering if' embed a question rather than opening a condition."""
        return lows[i] in ("if", "whether", "when", "once") and prev_w in self.lx.embedding_verbs

    def _question_restart(self, lows: list[str], i: int, prev_w: str) -> bool:
        w = lows[i]
        lx, b = self.lx, self.lx.base
        if w not in b.question_words or prev_w in lx.embedding_verbs or prev_w in lx.topic_prepositions \
                or prev_w in lx.restriction_prepositions or prev_w in b.question_words or prev_w in lx.coordinators \
                or (prev_w,) in lx.focus_phrases or (prev_w,) in lx.condition_phrases:
            return False                               # "specifically how high ..." stays inside the focus clause
        nxt = lows[i + 1] if i + 1 < len(lows) else ""
        if w == "how":
            return True
        return w in ("what", "when", "where", "why") and nxt in b.aux_questions

    # ------------------------------------------------------------------ roles
    def _assign_role(self, c: Clause, toks: list[Tok], lows: list[str], drop: list[bool], clauses: list[Clause]) -> None:
        lx, b = self.lx, self.lx.base
        words = [j for j in range(c.a, c.b) if not drop[j] and not toks[j].punct]
        k = 0
        # leading link words ("and also", "and", "so", "but", "then")
        while k < len(words):
            n = match_phrase(lows, words[k], lx.addition_phrases)
            if n:
                c.link = " ".join(lows[words[k]:words[k] + n]) if c.link is None else c.link
                k += _advance(words, k, n)
                continue
            if lows[words[k]] in lx.coordinators or lows[words[k]] in ("but", "then", "so", "now"):
                c.link = c.link or lows[words[k]]
                k += 1
                continue
            break
        rest = words[k:]
        if not rest:
            c.role = "IGNORE"
            return
        j0 = rest[0]
        n = match_phrase(lows, j0, lx.correction_phrases)
        has_replacement = any(match_phrase(lows, j, lx.correction_replacement_phrases) for j in rest) \
            or any(lows[j] in lx.correction_words for j in rest)
        if n or (has_replacement and c.opened_by in ("correction", "punct", "comma") and _prior_request(clauses, c)):
            c.role, c.marker = "CORRECTION", " ".join(lows[j0:j0 + n]) if n else "replacement"
            c.body = rest[_advance(rest, 0, n):] if n else rest
            return
        for phrases, kind in ((lx.focus_phrases, "focus"), (lx.condition_phrases, "condition"),
                              (lx.scope_all_phrases, "restriction")):
            n = match_phrase(lows, j0, phrases)
            if n:
                c.role, c.kind, c.marker = "CONSTRAINT", kind, " ".join(lows[j0:j0 + n])
                c.body = rest[_advance(rest, 0, n):]
                if kind == "restriction":
                    c.body = rest            # "for both of them" keeps its words; scope is decided later
                return
        n = match_phrase(lows, j0, b.request_heads) or match_phrase(lows, j0, lx.follow_up_openers)
        if n and lows[j0] not in lx.coordinators:
            c.role, c.head = "REQUEST", " ".join(lows[j0:j0 + n])
            body = rest[_advance(rest, 0, n):]
            if body and lows[body[0]] in lx.topic_prepositions:      # "i need information ABOUT x"
                body = body[1:]
            c.body = self._with_commas(body, c, toks, drop)
            return
        w0 = lows[j0]
        if w0 in b.question_words or w0 in b.aux_questions:
            c.role, c.body = "REQUEST", self._with_commas(rest, c, toks, drop)
            return
        if w0 in lx.request_verbs:
            body = rest[1:]
            while body and lows[body[0]] in ("me", "us", "about", "through", "on", "out"):
                body = body[1:]
            c.role, c.head, c.body = "REQUEST", w0, self._with_commas(body, c, toks, drop)
            return
        if (w0 in lx.restriction_prepositions or w0 in lx.topic_prepositions) and not self._has_request_cue(rest, lows):
            c.role, c.kind, c.marker, c.body = "CONSTRAINT", "restriction", None, rest
            return
        if _is_addition(c.link, lx) and _prior_request(clauses, c):
            c.role, c.body = "REQUEST", self._with_commas(rest, c, toks, drop)
            return
        if all(lows[j] in b.social or lows[j] in b.backchannel or lows[j] in self.stop for j in rest):
            c.role = "IGNORE"
            return
        c.role, c.body = "CONTEXT", rest
        subjects = {"i", "we", "you", "he", "she", "they", "my", "our", "your", "i'm", "we're", "it's"}
        if any(lows[j] in lx.aspect_nouns for j in rest) and not any(lows[j] in subjects for j in rest):
            c.role, c.implicit = "REQUEST", True          # "the requirements for x, how long ..." (facet NP)
            c.body = self._with_commas(rest, c, toks, drop)

    @staticmethod
    def _with_commas(body: list[int], c: Clause, toks: list[Tok], drop: list[bool]) -> list[int]:
        """Body word indices plus the commas between them (list separators for coordination splitting)."""
        if not body:
            return body
        return [j for j in range(body[0], c.b) if not drop[j] and (not toks[j].punct or toks[j].text == ",")]

    def _has_request_cue(self, idx: list[int], lows: list[str]) -> bool:
        b = self.lx.base
        return any(lows[j] in b.question_words or match_phrase(lows, j, b.request_heads) for j in idx)

    # ------------------------------------------------------------------ request clauses
    def _request_intents(self, c: Clause, toks: list[Tok], lows: list[str], uid: str, transcript: str,
                         constraints: list[DraftConstraint], new_key) -> list[DraftIntent]:
        body = [j for j in c.body]
        if not body or not any(self._is_content(lows[j]) for j in body):
            return []
        explicit = 0.8 if c.implicit or (c.head is None and c.link and _is_addition(c.link, self.lx)
                                         and lows[body[0]] not in self.lx.base.question_words
                                         and lows[body[0]] not in self.lx.base.aux_questions) else 1.0
        conjuncts = self._split_coordination(body, toks, lows)
        if len(conjuncts) == 1:
            return [self._draft(new_key(), c.index, "single" if c.index == 0 else "clause", conjuncts[0], toks,
                                transcript, explicit, 1.0)]
        drafts = [self._draft(new_key(), c.index, "coordination", cj, toks, transcript, min(explicit, 0.8), 0.8)
                  for cj in conjuncts]
        self._distribute_trailing_pp(drafts, conjuncts, toks, lows, uid, transcript, constraints, c.index)
        return drafts

    def _split_coordination(self, body: list[int], toks: list[Tok], lows: list[str]) -> list[list[int]]:
        lx = self.lx
        words = [j for j in body if not toks[j].punct]
        if any(match_phrase(lows, j, lx.comparison_cues) for j in words):
            return [body]
        cut_positions: list[int] = []                  # positions in `body` where a separator sits
        has_coord = any(lows[j] in lx.coordinators for j in body)
        for p, j in enumerate(body):
            w = lows[j]
            if w in lx.coordinators or toks[j].text == ",":
                prev_words = [lows[x] for x in body[:p] if not toks[x].punct]
                if any(x in lx.pair_openers for x in prev_words[-3:]):
                    continue                           # "between x and y" stays together
                cut_positions.append(p)
        if has_coord and not any(toks[j].text == "," for j in body):
            # ASR-style list without commas: "the a the b the c and the d" -> an article right after a content word
            # starts a new item (heuristic; only inside a coordinated request body)
            body = list(body)
            arts = [p for p, j in enumerate(body) if p > 0 and lows[j] in ("the", "a", "an")
                    and self._is_content(lows[body[p - 1]])]
            stops = sorted(set(arts) | set(cut_positions) | {len(body)})
            extra = []
            for p in arts:                             # an item with an auxiliary is a relative clause, not a list item
                nxt = min(q for q in stops if q > p)
                if not any(lows[body[q]] in self.lx.base.aux_questions for q in range(p, nxt)):
                    extra.append(p)
            if extra:
                cut_positions = sorted(set(cut_positions) | {p - 0.5 for p in extra})
        if not cut_positions:
            return [body]
        parts: list[list[int]] = []
        seps: list[int | None] = [None]                # separator token before each part
        last = 0
        for p in cut_positions:
            if isinstance(p, float):                   # zero-width boundary before position ceil(p)
                q = int(p + 0.5)
                parts.append(body[last:q])
                seps.append(None)
                last = q
                continue
            parts.append(body[last:p])
            seps.append(body[p])
            last = p + 1
        parts.append(body[last:])
        # merge back parts that are not substantive needs, and fixed corpus phrases ("x and y" seen in the corpus)
        out: list[list[int]] = []
        for part, sep in zip(parts, seps):
            content = [j for j in part if self._is_content(lows[j])]
            glue = [sep] if sep is not None else []
            if not content or all(lows[j].isdigit() for j in content):
                if out:
                    out[-1] = out[-1] + glue + part
                else:
                    out.append(part)
                continue
            if out:
                prev_content = [j for j in out[-1] if self._is_content(lows[j])]
                if not prev_content or (lows[prev_content[-1]], lows[content[0]]) in self.corpus_pairs:
                    out[-1] = out[-1] + glue + part
                    continue
            out.append(part)
        return [o for o in out if any(not toks[j].punct for j in o)] or [body]

    def _draft(self, key: str, clause_index: int, split: str, idx: list[int], toks: list[Tok], transcript: str,
               explicit: float, separation: float) -> DraftIntent:
        words = [j for j in idx if not toks[j].punct]
        while words and (toks[words[-1]].lower in self.lx.coordinators or toks[words[-1]].lower in self.stop
                         and toks[words[-1]].lower not in self.lx.base.negations
                         and toks[words[-1]].lower not in self.lx.base.aux_questions
                         and toks[words[-1]].lower not in self.lx.anaphora
                         and toks[words[-1]].lower not in self.lx.locative_anaphora):
            words = words[:-1]
        while words and toks[words[0]].lower in self.lx.coordinators:
            words = words[1:]
        start, end = toks[words[0]].start, toks[words[-1]].end
        text = " ".join(toks[j].text for j in words)
        return DraftIntent(key=key, clause_index=clause_index, split=split, tok_idx=words, text=text, start=start,
                           end=end, explicit=explicit, separation=separation)

    def _distribute_trailing_pp(self, drafts: list[DraftIntent], conjuncts: list[list[int]], toks: list[Tok],
                                lows: list[str], uid: str, transcript: str, constraints: list[DraftConstraint],
                                clause_index: int) -> None:
        """'requirements and timeline FOR X' -> both are about X; 'X and Y FOR APPLICANTS FROM Z' -> restriction on both."""
        lx = self.lx
        last = drafts[-1]
        pp_at = None
        for p, j in enumerate(last.tok_idx):
            if p > 0 and (lows[j] in lx.topic_prepositions or lows[j] in lx.restriction_prepositions) \
                    and any(self._is_content(lows[x]) for x in last.tok_idx[:p]):
                pp_at = p
                break
        if pp_at is None:
            return
        earlier = drafts[:-1]
        if any(any(lows[j] in lx.topic_prepositions or lows[j] in lx.restriction_prepositions for j in d.tok_idx[1:])
               for d in earlier):
            return
        pp_idx = last.tok_idx[pp_at:]
        if not any(self._is_content(lows[j]) for j in pp_idx):
            return
        pp_start, pp_end = toks[pp_idx[0]].start, toks[pp_idx[-1]].end
        pp_text = transcript[pp_start:pp_end]
        aspect_only = all(all(not self._is_content(lows[j]) or lows[j] in lx.aspect_nouns for j in d.tok_idx)
                          for d in earlier)
        if aspect_only:
            for d in earlier:
                d.inherited.append(Segment(pp_text, uid, pp_start, pp_end, "inherited", last.key, "distributed_pp"))
            return
        if lows[pp_idx[0]] in lx.restriction_prepositions:
            core = last.tok_idx[:pp_at]
            last.tok_idx = core
            last.start, last.end = toks[core[0]].start, toks[core[-1]].end
            last.text = " ".join(toks[j].text for j in core)
            k = DraftConstraint(f"k{len(constraints) + 1}", "restriction", lows[pp_idx[0]], pp_text, pp_start, pp_end,
                                clause_index, scope="global", applies_to=[d.key for d in drafts],
                                scope_reason="trailing_pp_after_coordination", scope_confidence=1.0)
            k.pinned = True
            constraints.append(k)

    # ------------------------------------------------------------------ constraints
    def _constraint(self, c: Clause, toks: list[Tok], lows: list[str], transcript: str, key: str) -> DraftConstraint:
        words = [j for j in c.body if not toks[j].punct]
        start, end = toks[words[0]].start, toks[words[-1]].end
        return DraftConstraint(key, c.kind or "restriction", c.marker, transcript[start:end], start, end, c.index,
                               scope_all=any(match_phrase(lows, j, self.lx.scope_all_phrases) for j in words),
                               has_question=any(lows[j] in self.lx.base.question_words for j in words),
                               preposition=lows[words[0]] in self.lx.restriction_prepositions)

    def _scope_constraints(self, constraints: list[DraftConstraint], intents: list[DraftIntent],
                           relations: list[DraftRelation]) -> None:
        """Scope rules (docs/multi_intent/03), first match wins:
        pinned (trailing PP after a coordination) | explicit 'for both/all' -> global | focus + question word ->
        local to the previous need | shares a term with exactly one need -> local | restriction / condition ->
        global (scope_confidence 0.6 when >= 2 needs: ambiguous) | bare focus -> nearest preceding need (0.6)."""
        active = [i for i in intents if i.status == "ACTIVE"]
        keys = {i.key for i in active}
        for k in constraints:
            k.terms = list(dict.fromkeys(self.terms_fn(k.text)))
            if k.pinned:
                k.applies_to = [a for a in k.applies_to if a in keys]
            elif not active:
                k.applies_to = []
            elif k.scope_all:
                k.applies_to, k.scope_reason, k.scope_confidence = [i.key for i in active], "explicit_all", 1.0
            else:
                before = [i for i in active if i.clause_index < k.clause_index]
                sharing = [i for i in active if set(k.terms) & set(self.terms_fn(i.text))]
                if k.kind == "focus" and k.has_question and before:
                    k.applies_to, k.scope_reason, k.scope_confidence = \
                        [before[-1].key], "focus_question_refines_previous", 0.8
                elif len(sharing) == 1 and k.kind == "focus":
                    k.applies_to, k.scope_reason, k.scope_confidence = [sharing[0].key], "shares_term_with_one_intent", 0.9
                elif k.kind in ("restriction", "condition") or k.preposition:
                    k.applies_to = [i.key for i in active]
                    k.scope_reason = "restriction_applies_to_request" if before else "fronted_restriction"
                    k.scope_confidence = 1.0 if (len(active) == 1 or not before) else 0.6
                elif before:
                    k.applies_to, k.scope_reason, k.scope_confidence = [before[-1].key], "nearest_preceding_intent", 0.6
                else:
                    k.applies_to, k.scope_reason, k.scope_confidence = [i.key for i in active], "fronted_focus", 0.6
            if len(active) == 1 and k.applies_to:
                k.scope_confidence = 1.0             # one need: no scope ambiguity
            refines = k.scope_reason in ("focus_question_refines_previous", "shares_term_with_one_intent",
                                         "nearest_preceding_intent")
            k.scope = "local" if refines or len(k.applies_to) < len(active) else "global"
            for a in k.applies_to:
                intent = next(i for i in intents if i.key == a)
                if k.key not in intent.constraint_keys:
                    intent.constraint_keys.append(k.key)
                relations.append(DraftRelation("CONSTRAINT_OF", k.key, a, k.scope_reason))

    # ------------------------------------------------------------------ corrections
    def _correction(self, c: Clause, toks: list[Tok], lows: list[str], uid: str, transcript: str,
                    intents: list[DraftIntent], ctx: DecompositionContext, new_key,
                    ext_sup: list[tuple[str, str, str]]) -> DraftIntent | None:
        """'I meant Y instead of X' / 'actually, I mean Y': the corrected need supersedes its target (the intent
        containing X, else the most recent one - in this utterance first, then earlier in the session)."""
        lx = self.lx
        body = [j for j in c.body if not toks[j].punct]
        new_part, old_part = body, []
        for p, j in enumerate(body):
            n = match_phrase(lows, j, lx.correction_replacement_phrases)
            if n:
                new_part, old_part = body[:p], body[_advance(body, p, n) + p:]
                break
            if lows[j] in lx.correction_words:
                new_part, old_part = body[:p], body[p + 1:]
                break
            if lows[j] == "not" and any(self._is_content(lows[x]) for x in body[:p]) \
                    and any(self._is_content(lows[x]) for x in body[p + 1:]):
                new_part, old_part = body[:p], body[p + 1:]      # "I meant the lamp, not the lens"
                break
        while len(new_part) > 1 and (lows[new_part[0]] in lx.request_verbs
                                     or lows[new_part[0]] in ("me", "us", "about", "to")):
            new_part = new_part[1:]
        if not new_part or not any(self._is_content(lows[j]) for j in new_part):
            return None
        old_terms = set(self.terms_fn(" ".join(lows[j] for j in old_part))) if old_part else set()
        local = [i for i in intents if i.status == "ACTIVE"]
        target_local: DraftIntent | None = None
        target_prior: PriorIntent | None = None
        if old_terms:
            scored = [(len(old_terms & set(self.terms_fn(i.text))), n, i) for n, i in enumerate(local)]
            scored = [x for x in scored if x[0] > 0]
            if scored:
                target_local = max(scored, key=lambda x: (x[0], x[1]))[2]
            else:
                pri = [(len(old_terms & set(p.terms)), n, p) for n, p in enumerate(ctx.prior)]
                pri = [x for x in pri if x[0] > 0]
                if pri:
                    target_prior = max(pri, key=lambda x: (x[0], x[1]))[2]
        elif local:
            target_local = local[-1]
        elif ctx.prior:
            target_prior = ctx.prior[-1]
        if target_local is None and target_prior is None:
            return None
        d = self._draft(new_key(), c.index, "correction", new_part, toks, transcript, 1.0, 1.0)
        d.correction_cue = c.marker
        bare = not (any(lows[j] in lx.aspect_nouns for j in new_part)
                    or any(lows[j] in lx.base.question_words for j in new_part))
        if target_local is not None:
            target_local.status, target_local.superseded_by = "SUPERSEDED", d.key
            d.supersedes = target_local.key
            if bare and old_part:
                d.replace_in, d.replace_words = target_local, [lows[j] for j in old_part]
            elif bare:
                d.aspect_from = target_local
        else:
            d.supersedes = target_prior.intent_id
            ext_sup.append((target_prior.intent_id, d.key, c.marker or "correction"))
            if bare and old_part and target_prior.segments:
                d.replace_in, d.replace_words = target_prior, [lows[j] for j in old_part]
            elif bare:
                d.aspect_from = target_prior
        return d

    # ------------------------------------------------------------------ fallback
    def _fallback(self, transcript: str, toks: list[Tok], uid: str, new_key) -> DraftIntent | None:
        q = self.fallback_builder.build(transcript)
        if q.empty or not self.terms_fn(q.text):
            return None
        s, e = q.spans[0][0], q.spans[-1][1]
        idx = [k for k, t in enumerate(toks) if not t.punct and s <= t.start and t.end <= e]
        idx = [k for k in idx if any(a <= toks[k].start and toks[k].end <= b for a, b in q.spans)]
        return DraftIntent(key=new_key(), clause_index=0, split="fallback", tok_idx=idx, text=q.text, start=s, end=e,
                           explicit=0.5, separation=0.6, origin="rule_fallback")

    # ------------------------------------------------------------------ annotation
    def _annotate(self, i: DraftIntent, toks: list[Tok], lows: list[str], uid: str) -> None:
        """Segments (verbatim pieces), type, entities, topic/aspect."""
        lx = self.lx
        i.segments = [Segment(toks[j].text, uid, toks[j].start, toks[j].end) for j in i.tok_idx]
        ws = [lows[j] for j in i.tok_idx]
        self._type(i, ws)
        cue_tok: set[int] = set()
        for k in range(len(ws)):                 # question phrases ("how long", "how many") are not topics
            if ws[k] == "how" and k + 1 < len(ws):
                cue_tok.add(i.tok_idx[k + 1])
        runs: list[list[int]] = []
        for j in i.tok_idx:
            if self._is_content(lows[j]):
                if runs and runs[-1][-1] == j - 1:
                    runs[-1].append(j)
                else:
                    runs.append([j])
        i.entities = [" ".join(toks[j].text for j in r) for r in runs]
        # aspect + topic: "requirements for X" / "X's application process" / plain topic
        content = [j for j in i.tok_idx if self._is_content(lows[j])]
        topic = None
        for p, j in enumerate(i.tok_idx):
            if lows[j] in lx.topic_prepositions and p > 0:
                before = [x for x in i.tok_idx[:p] if self._is_content(lows[x])]
                after = [x for x in i.tok_idx[p + 1:] if self._is_content(lows[x])]
                if before and after and all(lows[x] in lx.aspect_nouns for x in before):
                    i.aspect = " ".join(toks[x].text for x in before)
                    topic = after
                    break
        if topic is None:
            aspects = [j for j in content if lows[j] in lx.aspect_nouns]
            topical = [j for j in content if lows[j] not in lx.aspect_nouns and j not in cue_tok]
            if aspects:
                i.aspect = " ".join(toks[j].text for j in aspects)
            if topical:
                run = next(r for r in runs if any(j in topical for j in r))
                topic = [j for j in run if j in topical]
        if topic:
            i.topic = (" ".join(toks[j].text for j in topic), toks[topic[0]].start, toks[topic[-1]].end, uid)

    def _type(self, i: DraftIntent, ws: list[str]) -> None:
        """First intent type (lexicon order = precedence) whose cue occurs in the intent's own words."""
        i.intent_type, i.type_cues = "OTHER", []
        for t, phrases in self.lx.type_cues:
            hits = [ph for k in range(len(ws)) for ph in phrases if tuple(ws[k:k + len(ph)]) == ph]
            if hits:
                i.intent_type, i.type_cues = t, list(dict.fromkeys(" ".join(ph) for ph in hits))
                return

    # ------------------------------------------------------------------ anaphora / ellipsis
    def _resolve_references(self, intents: list[DraftIntent], toks: list[Tok], lows: list[str], uid: str,
                            transcript: str, ctx: DecompositionContext, relations: list[DraftRelation]) -> None:
        lx = self.lx
        for n, i in enumerate(intents):
            if i.replace_in is not None:          # "requirements for X" + "Y instead of X" -> "requirements for Y"
                old = set(i.replace_words)
                segs: list[Segment] = []
                placed = False
                for sgm in i.replace_in.segments:
                    words = sgm.text.lower().split()
                    if words and all(w in old for w in words):
                        if not placed:
                            segs += i.segments
                            placed = True
                        continue
                    segs.append(Segment(sgm.text, sgm.utterance_id, sgm.start, sgm.end, "inherited",
                                        i.replace_in.key, "correction_aspect"))
                if placed:
                    i.segments = segs
                    i.aspect = i.replace_in.aspect
            elif i.aspect_from is not None and i.aspect_from.aspect \
                    and i.aspect_from.aspect.lower() not in i.text.lower():
                src = i.aspect_from
                if isinstance(src, DraftIntent):
                    k = transcript.lower().find(src.aspect.lower(), src.start)
                    span = (k, k + len(src.aspect)) if k >= 0 else None
                    ref, u = src.key, uid
                else:
                    span, ref, u = src.aspect_span, src.intent_id, src.utterance_id
                if span is not None:
                    i.segments = [Segment(src.aspect, u, span[0], span[1], "inherited", ref,
                                          "correction_aspect")] + i.segments
                    i.aspect = src.aspect
            if i.status != "ACTIVE" or i.split == "correction":
                continue
            antecedent = self._antecedent(intents[:n])
            prior = ctx.recent[-1] if ctx.recent else None
            new_segs: list[Segment] = []
            for p, j in enumerate(i.tok_idx):
                w = lows[j]
                nxt = lows[i.tok_idx[p + 1]] if p + 1 < len(i.tok_idx) else None
                is_pron = w in lx.anaphora and (w in ("it", "its", "they", "them", "their", "one", "ones")
                                                or nxt is None or nxt in lx.base.aux_questions
                                                or nxt in lx.restriction_prepositions or nxt in lx.topic_prepositions)
                own = [x for q, x in enumerate(i.tok_idx[:p]) if self._is_content(lows[x])
                       and not (q > 0 and lows[i.tok_idx[q - 1]] == "how")]
                if is_pron and own:
                    is_pron = False                    # an antecedent inside the same need ("workers ... they")
                if is_pron and (antecedent is not None or prior is not None):
                    if antecedent is not None and antecedent.topic is not None:
                        txt, s, e, tu = antecedent.topic
                        new_segs.append(Segment(txt, tu, s, e, "inherited", antecedent.key, "anaphora", w))
                        relations.append(DraftRelation("DEPENDENT", i.key, antecedent.key, f"pronoun '{w}'"))
                        i.topic = antecedent.topic
                        continue
                    if antecedent is None and prior is not None and prior.topic is not None:
                        txt, s, e, tu = prior.topic
                        new_segs.append(Segment(txt, tu, s, e, "inherited", prior.intent_id, "anaphora", w))
                        relations.append(DraftRelation("FOLLOW_UP", i.key, prior.intent_id, f"pronoun '{w}'"))
                        i.topic = prior.topic
                        continue
                if w in lx.locative_anaphora and (nxt is None or not self._is_content(nxt)):
                    i.unresolved.append(w)
                    continue
                new_segs.append(Segment(toks[j].text, uid, toks[j].start, toks[j].end))
            i.segments = new_segs or i.segments
            # ellipsis: an aspect-only need inherits the topic of the previous need
            content = [lows[j] for j in i.tok_idx if self._is_content(lows[j])]
            has_inherited = any(s.source == "inherited" for s in i.segments) or i.inherited
            if content and all(w in lx.aspect_nouns for w in content) and not has_inherited:
                if antecedent is not None and antecedent.topic is not None:
                    txt, s, e, tu = antecedent.topic
                    i.inherited.append(Segment(txt, tu, s, e, "inherited", antecedent.key, "follow_up_ellipsis"))
                    i.topic = antecedent.topic
                    relations.append(DraftRelation("DEPENDENT", i.key, antecedent.key, "aspect without topic"))
                elif antecedent is None and prior is not None and prior.topic is not None:
                    txt, s, e, tu = prior.topic
                    i.inherited.append(Segment(txt, tu, s, e, "inherited", prior.intent_id, "follow_up_ellipsis"))
                    i.topic = prior.topic
                    relations.append(DraftRelation("FOLLOW_UP", i.key, prior.intent_id, "aspect without topic"))

    def _antecedent(self, earlier: list[DraftIntent]) -> DraftIntent | None:
        for d in reversed(earlier):
            if d.status == "ACTIVE" and d.topic is not None:
                return d
        return None

    # ------------------------------------------------------------------ terms, dedup, refinement
    def _finalize_terms(self, i: DraftIntent) -> None:
        if i.split == "correction":       # a corrected need keeps the target's framing ("requirements for <new>")
            self._type(i, [w.lower() for s in i.segments for w in s.text.split()])
        text = " ".join([i.resolved_text] + [s.text for s in i.inherited])
        i.terms = list(dict.fromkeys(self.terms_fn(text)))
        idfs = [self.idf_fn(t) for t in i.terms]
        best = max((x for x in idfs if x is not None and x >= self.anchor_floor), default=0.0)
        i.anchor_strength = round(min(1.0, best / self.anchor_norm), 3)

    def _dedup_and_refine(self, intents: list[DraftIntent], merged: list[tuple[str, str, str]],
                          relations: list[DraftRelation]) -> None:
        active = [i for i in intents if i.status == "ACTIVE"]
        for a_i, a in enumerate(active):
            if a.status != "ACTIVE":
                continue
            for b in active[a_i + 1:]:
                if b.status != "ACTIVE":
                    continue
                ta, tb = set(a.terms), set(b.terms)
                if not ta or not tb:
                    continue
                j = len(ta & tb) / len(ta | tb)
                linked = any({r.source, r.target} == {a.key, b.key} for r in relations
                             if r.type in ("DEPENDENT", "FOLLOW_UP"))
                own_a, own_b = set(self.terms_fn(a.text)), set(self.terms_fn(b.text))
                if j == 1.0 or j >= self.dup_j:
                    b.status = "MERGED"
                    a.constraint_keys += [k for k in b.constraint_keys if k not in a.constraint_keys]
                    merged.append((b.text, a.key, "duplicate" if j == 1.0 else "near_duplicate"))
                elif linked:
                    continue
                elif own_a and own_a < own_b:
                    relations.append(DraftRelation("REFINEMENT", b.key, a.key, "narrower need on the same terms"))
                elif own_b and own_b < own_a:
                    relations.append(DraftRelation("REFINEMENT", a.key, b.key, "narrower need on the same terms"))
        gone = {i.key for i in intents if i.status == "MERGED"}
        if gone:
            relations[:] = [r for r in relations if r.source not in gone and r.target not in gone]

    # ------------------------------------------------------------------ scoring
    def _score(self, intents: list[DraftIntent], constraints: list[DraftConstraint], relations: list[DraftRelation],
               ctx: DecompositionContext) -> None:
        active = [i for i in intents if i.status == "ACTIVE"]
        depended = {r.target for r in relations if r.type in ("DEPENDENT", "REFINEMENT")}
        for i in active:
            uses_inherited = any(s.source == "inherited" for s in i.segments) or bool(i.inherited)
            resolution = 0.7 if i.unresolved else (0.85 if uses_inherited else 1.0)
            separation = 1.0 if (len(active) == 1 and i.split != "fallback") else i.separation
            conf = 0.35 * i.explicit + 0.30 * i.anchor_strength + 0.20 * separation + 0.15 * resolution
            i.confidence = round(conf, 3)
            prior_sets = [set(t) for t in ctx.prior_query_terms if t]
            ti = set(i.terms)
            novelty = 1.0 - max((len(ti & p) / len(ti | p) for p in prior_sets if ti | p), default=0.0)
            has_k = 1.0 if i.constraint_keys else 0.0
            dep = 1.0 if i.key in depended else 0.0
            i.priority_score = round(0.35 * conf + 0.25 * i.anchor_strength + 0.15 * i.explicit + 0.10 * has_k
                                     + 0.10 * novelty + 0.05 * dep, 3)
            i.signals = {"explicit": i.explicit, "specificity": i.anchor_strength, "separation": separation,
                         "resolution": resolution, "novelty": round(novelty, 3), "has_constraints": has_k,
                         "has_dependents": dep}
        ranked = sorted(active, key=lambda x: (-x.priority_score, x.order))
        for r, i in enumerate(ranked, start=1):
            i.priority = r

    def _budget(self, intents: list[DraftIntent], dropped: list[tuple[str, str | None, str]]) -> None:
        active = [i for i in intents if i.status == "ACTIVE"]
        if len(active) <= self.max_intents:
            return
        for i in sorted(active, key=lambda x: x.priority)[self.max_intents:]:
            i.status = "DROPPED"
            dropped.append((i.text, i.key, f"max_intents_budget: priority {i.priority} of {len(active)} "
                                           f"(score {i.priority_score}) exceeds max_intents={self.max_intents}"))
        keep = [i for i in intents if i.status == "ACTIVE"]
        for r, i in enumerate(sorted(keep, key=lambda x: x.priority), start=1):
            i.priority = r


# ---------------------------------------------------------------------------------------------- helpers
def w_is_coord(w: str, lx: IntentLexicon) -> bool:
    return w in lx.coordinators


def _advance(words: list[int], k: int, n: int) -> int:
    """Number of entries of ``words`` (non-dropped word indices) covered by a phrase of n tokens at words[k]."""
    if n == 0:
        return 0
    end_tok = words[k] + n
    m = 0
    while k + m < len(words) and words[k + m] < end_tok:
        m += 1
    return max(m, 1)


def _is_addition(link: str | None, lx: IntentLexicon) -> bool:
    return bool(link) and tuple(link.split()) in lx.addition_phrases


def _prior_request(clauses: list[Clause], c: Clause) -> bool:
    return any(x.role == "REQUEST" for x in clauses if x.index < c.index)


def _find_span(text: str | None, phrase: str) -> tuple[int, int] | None:
    if not text:
        return None
    k = text.lower().find(phrase.lower())
    return (k, k + len(phrase)) if k >= 0 else None


def corpus_coordination_pairs(texts: list[str]) -> frozenset[tuple[str, str]]:
    """(a, b) for every 'a and b' / 'a & b' in the corpus: fixed phrases that must not be split into two intents."""
    rx = re.compile(r"\b([a-z][a-z'\-]*)\s+(?:and|&)\s+([a-z][a-z'\-]*)\b")
    out = set()
    for t in texts:
        out.update(rx.findall(t.lower()))
    return frozenset(out)


def anchor_norm_for(n_docs: int) -> float:
    n = max(n_docs, 1)
    return math.log1p((n - 1 + 0.5) / 1.5) or 1.0
