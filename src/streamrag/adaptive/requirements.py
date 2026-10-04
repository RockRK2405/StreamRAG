"""Claim-driven retrieval: Intent -> claim requirements -> evidence requirements (docs/retrieval/05).

A need is not "answered" because chunks came back; it is answered when every claim the answer must make has evidence.
Before searching, the need is turned into the claim slots its answer needs:

  main       the asked fact; a *value* slot when the question asks for a value kind ("how much" -> amount):
             supporting text must then state a value of that kind, not just mention the topic
  condition  one slot per active constraint ("for international applicants"): the evidence must be specific to it
             (constraint words in the text, or a matching metadata value - a document "for all" is not specific)
  item       comparisons: one slot per compared entity
  bridge     multi-hop: the statement that links the entity to what the question is about (added when a hop is
             planned)

Every slot carries the question's validity date (temporal questions). With ``claim_driven`` off (ablation) the need
is a single requirement made of all its content terms.
"""

from __future__ import annotations

from streamrag.adaptive.lexicon import QUESTION_WORDS
from streamrag.adaptive.models import EvidenceRequirement, QueryAnalysis

class RequirementBuilder:
    def __init__(self, terms_fn, claim_driven: bool = True) -> None:
        self.terms_fn, self.claim_driven = terms_fn, claim_driven
        self._q = {t for w in QUESTION_WORDS for t in terms_fn(w)}
        self._extra: set[str] = set()

    def content(self, text: str) -> list[str]:
        return [t for t in dict.fromkeys(self.terms_fn(text)) if t not in self._q and t not in self._extra]

    def build(self, a: QueryAnalysis, text: str, valid_at: str | None, valid_to: str | None = None,
              alternatives: dict[str, list[str]] | None = None) -> list[EvidenceRequirement]:
        reqs = self._build(a, text, valid_at)
        keys = [t for x in a.exact_ids for t in self.terms_fn(x)] + list(a.bridge_terms)
        for r in reqs:
            r.valid_to = valid_to
            r.alternatives = {t: v for t, v in (alternatives or {}).items() if t in r.terms}
            if r.kind != "comparison_item":
                r.key_terms = [t for t in dict.fromkeys(keys) if t in r.terms]
        return reqs

    def _build(self, a: QueryAnalysis, text: str, valid_at: str | None) -> list[EvidenceRequirement]:
        self._extra = set(a.question_terms)
        cterms = [t for c in a.constraint_texts for t in self.content(c)]
        filt_terms = {t for vals in a.metadata_filters.values() for v in vals for t in self.terms_fn(v)}
        core = [t for t in self.content(text) if t not in cterms and t not in filt_terms
                and not (a.temporal.cue and t in self.terms_fn(a.temporal.cue))]
        if not core:
            core = self.content(text) or list(a.terms)
        if not self.claim_driven:
            return [EvidenceRequirement(requirement_id="R1", claim_slot="need: " + text, kind="fact",
                                        terms=list(dict.fromkeys(self.content(text) + cterms)), valid_at=valid_at,
                                        query_text=text)]
        reqs: list[EvidenceRequirement] = []
        if a.comparison and len(a.entities) >= 2:
            shared = [t for t in core if not any(t in self.terms_fn(e) for e in a.entities)]
            for n, e in enumerate(a.entities[:4], start=1):
                reqs.append(EvidenceRequirement(
                    requirement_id=f"R{n}", claim_slot=f"comparison item: {e}", kind="comparison_item",
                    terms=list(dict.fromkeys(self.terms_fn(e) + shared)), value_kind=a.value_kind, valid_at=valid_at,
                    query_text=self._without(text, [x for x in a.entities if x != e], e)))
        else:
            kind = "value" if a.value_kind else "fact"
            slot = f"{a.value_kind} for: " if a.value_kind else "fact: "
            reqs.append(EvidenceRequirement(requirement_id="R1", claim_slot=slot + " ".join(core), kind=kind,
                                            terms=core, value_kind=a.value_kind, valid_at=valid_at, query_text=text))
        n = len(reqs)
        for c in a.constraint_texts:
            ct = self.content(c)
            if not ct:
                continue
            n += 1
            reqs.append(EvidenceRequirement(requirement_id=f"R{n}", claim_slot=f"condition: {c}", kind="condition",
                                            terms=core, constraint_terms=ct, valid_at=valid_at,
                                            value_kind=a.value_kind,
                                            query_text=text if set(ct) <= set(self.terms_fn(text)) else f"{text} {c}"))
        for f, vals in a.metadata_filters.items():
            vt = [t for v in vals for t in self.terms_fn(v)]
            if any(r.kind == "condition" and set(vt) <= set(r.constraint_terms) for r in reqs):
                for r in reqs:
                    if r.kind == "condition" and set(vt) <= set(r.constraint_terms):
                        r.metadata = {**r.metadata, f: vals}
                continue
            n += 1
            reqs.append(EvidenceRequirement(requirement_id=f"R{n}", claim_slot=f"condition: {f} = {', '.join(vals)}",
                                            kind="condition", terms=core, constraint_terms=vt, metadata={f: vals},
                                            valid_at=valid_at, query_text=text, value_kind=a.value_kind))
        return reqs

    def _without(self, text: str, others: list[str], keep: str) -> str:
        """``text`` without the words of the other compared entities, plus the item itself."""
        drop = {t for o in others for t in self.terms_fn(o)} - set(self.terms_fn(keep))
        words = [w for w in text.split() if not set(self.terms_fn(w)) & drop]
        return " ".join(([keep] if not set(self.terms_fn(keep)) <= set(self.terms_fn(" ".join(words))) else [])
                        + words)

    def bridge(self, rid: str, entity_term: str, target: str, core: list[str], valid_at: str | None,
               value_kind: str | None, hop_query: str) -> EvidenceRequirement:
        """Hop requirement: the asked aspect for the bridge target (which replaces the unreachable entity)."""
        tt = self.content(target)
        aspect = [t for t in core if t != entity_term]
        return EvidenceRequirement(requirement_id=rid, claim_slot=f"bridge: {entity_term} -> {target}: "
                                   + " ".join(aspect), kind="bridge", terms=list(dict.fromkeys(tt + aspect)),
                                   key_terms=tt, value_kind=value_kind, valid_at=valid_at, query_text=hop_query)

    def link(self, r: EvidenceRequirement, entity_term: str, target: str) -> None:
        """The original slot becomes the link statement "<entity> <relation> <target>" once a bridge is found."""
        r.kind, r.value_kind = "bridge", None
        r.terms = list(dict.fromkeys([entity_term] + self.content(target)))
        r.key_terms = [entity_term]
        r.claim_slot = f"link: {entity_term} -> {target}"
