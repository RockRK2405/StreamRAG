"""EvidenceSufficiencyEvaluator and conflict resolution (docs/retrieval/05, 09).

A requirement is MET by an evidence item when

  key terms       every key term (identifier, multi-hop entity / bridge target) occurs
  anchor term     (slots without key terms / constraint) the most specific term (highest IDF; a content word unknown
                  to the index and without equivalent counts as most specific) occurs - "cost to renew a permit" is
                  not met by a text about the *application* fee, "pay by credit card" not by a text about paying
  term coverage   |requirement terms matched in (document title + section heading + chunk text)| / |terms|
                  >= ``requirement_coverage``. A term matches itself, a validated equivalent (corpus-defined acronym
                  <-> long form, a synonym the rewrite added for it) or a morphological variant (one analyzed stem is
                  a prefix of the other, both >= 4 characters: "appli" / "applic")
  value           (value slots) a sentence of the chunk that mentions a requirement term states a value of the
                  asked kind (claims/textcheck.has_value_of)
  condition       (condition slots) the constraint words occur in the chunk, or the document's metadata names the
                  constrained value (a document "for all" is not specific evidence for the condition)
  validity        the document is valid at the question's date / period (explicit date, else the reference date for
                  "current" and undated questions; "past" questions without a date are not filtered). Documents
                  without validity dates always pass. Evidence that matched but is not valid is reported
                  (``excluded``: temporal_validity), never silently used.

Status of the need:  CONTRADICTORY  a value requirement has sources stating different values and neither an agreed
                                    supersession, a "past" question nor a strict authority difference decides
                     SUFFICIENT     every requirement MET (share >= evidence_threshold)
                     PARTIAL        some MET
                     INSUFFICIENT   none MET
A requirement is *unattainable* when its terms cannot reach the coverage threshold even in principle (terms unknown
to the index and without equivalents): no further search can meet it (stopping: NO_EXPECTED_GAIN).

Applicability: a need constrained to a metadata value cannot use evidence from a document naming a different
specific value (excluded: not_applicable); a document "for all" states the general value and meets a value
condition (not a fact / list condition, which needs condition-specific evidence).
Conflicts are detected per chunk pair from different documents (disjoint value sets of the best-matching
sentences; documents for different populations - different specific filter values - are not compared) and
resolved in this order: agreed supersession -> for a "past" question the superseded version ->
authority difference >= 0.3. Losers are excluded from the evidence handed on (with the rule).
"""

from __future__ import annotations

from streamrag.adaptive.catalog import MetadataCatalog
from streamrag.adaptive.models import EvidenceRequirement, SufficiencyAssessment
from streamrag.adaptive.values import sentences, values_of
from streamrag.claims.textcheck import has_value_of


def _variant(a: str, b: str) -> bool:
    if a == b:
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) >= 4 and l.startswith(s) and not s.isdigit()


class EvidenceSufficiencyEvaluator:
    def __init__(self, terms_fn, catalog: MetadataCatalog, coverage: float, reference_date: str,
                 vocab: dict | None = None, idf_fn=None) -> None:
        self.terms_fn, self.catalog, self.theta, self.ref = terms_fn, catalog, coverage, reference_date
        self.vocab = vocab or {}
        self.idf = idf_fn or (lambda t: None)
        self._cache: dict[str, tuple[set[str], list[tuple[str, set[str]]]]] = {}

    # ------------------------------------------------------------------ term matching
    def _expand(self, terms: set[str]) -> set[str]:
        """Add corpus-defined acronym equivalents: long form present -> acronym term, acronym -> long-form terms."""
        out = set(terms)
        for lt, short in self.catalog.long_forms.items():
            if lt and set(lt) <= terms:
                out |= set(self.terms_fn(short))
        for acr, long in self.catalog.acronyms.items():
            if acr in terms:
                out |= set(self.terms_fn(long))
        return out

    def _terms(self, e) -> tuple[set[str], list[tuple[str, set[str]]]]:
        got = self._cache.get(e.evidence_id)
        if got is None:
            head = set(self.terms_fn(" ".join(x for x in (e.document_title or "", e.section_title or "") if x)))
            sents = [(s, self._expand(set(self.terms_fn(s)) | head)) for s in sentences(e.text)]
            allt = self._expand(head | {t for _, st in sents for t in st})
            got = self._cache[e.evidence_id] = (allt, sents)
        return got

    @staticmethod
    def has(term: str, terms: set[str], alts: dict[str, list[str]]) -> bool:
        if term in terms or any(a in terms for a in alts.get(term, [])):
            return True
        return any(_variant(term, x) for x in terms)

    def coverage_of(self, r: EvidenceRequirement, e) -> float:
        if not r.terms:
            return 0.0
        allt, _ = self._terms(e)
        rt = set(r.terms)
        return sum(1 for t in rt if self.has(t, allt, r.alternatives)) / len(rt)

    def attainable(self, r: EvidenceRequirement) -> bool:
        """Can any chunk of the index reach the coverage threshold for ``r`` (vocabulary check, no search)?"""
        if not r.terms or not self.vocab:
            return True
        vocab = self.vocab

        def known(t: str) -> bool:
            if t in vocab or any(a in vocab for a in r.alternatives.get(t, [])):
                return True
            if t in self.catalog.acronyms or any(t in lt for lt in self.catalog.long_forms):
                return True
            return len(t) >= 4 and any(_variant(t, v) for v in self._prefix_candidates(t))
        rt = set(r.terms)
        anchor = self.anchor(r)
        if any(not known(t) for t in r.key_terms) or (anchor is not None and not known(anchor)):
            return False
        return sum(1 for t in rt if known(t)) / len(rt) >= self.theta

    def _prefix_candidates(self, t: str):
        p = t[:4]
        if not hasattr(self, "_by_prefix"):
            self._by_prefix: dict[str, list[str]] = {}
            for v in self.vocab:
                self._by_prefix.setdefault(v[:4], []).append(v)
        return self._by_prefix.get(p, [])

    def _weight(self, t: str, r: EvidenceRequirement) -> float:
        w = self.idf(t)
        if w is not None:
            return w
        alts = [x for x in (self.idf(a) for a in r.alternatives.get(t, [])) if x is not None]
        if alts:
            return max(alts)
        var = [x for x in (self.idf(v) for v in self._prefix_candidates(t) if _variant(t, v)) if x is not None]
        return max(var) if var else float("inf")

    def anchor(self, r: EvidenceRequirement) -> str | None:
        """The most specific non-key term (ties: first in the question). A content word the corpus never uses and
        that has no equivalent is the most specific of all: the question asks about something the corpus does not
        mention ("pay by credit card"), so the slot is unattainable. Question words and fillers are not terms."""
        best, bw = None, -1.0
        for t in r.terms:
            if t in r.key_terms or t.isdigit():
                continue
            w = self._weight(t, r)
            if w > bw:
                best, bw = t, w
        # key terms (identifiers, bridge target) or a condition's constraint already make the slot specific
        return best if len(r.terms) > 1 and not r.key_terms and r.kind != "condition" else None

    # ------------------------------------------------------------------ support
    def supports(self, r: EvidenceRequirement, e) -> tuple[bool, list[str], str | None]:
        """(meets the requirement, values of the asked kind in its best sentence(s), rejection reason)."""
        allt, sents = self._terms(e)
        if any(not self.has(t, allt, r.alternatives) for t in r.key_terms):
            return False, [], None
        if self.coverage_of(r, e) < self.theta:
            return False, [], None
        anchor = self.anchor(r)
        if anchor is not None and not self.has(anchor, allt, r.alternatives):
            return False, [], None
        if r.kind == "condition":
            meta = self.catalog.docs.get(e.document_id)
            fv = {f: {x.strip().lower() for x in str(meta.meta.get(f, "")).split(",")} for f in r.metadata} \
                if meta is not None else {}
            meta_ok = bool(r.metadata) and any(fv.get(f, set()) & {v.lower() for v in vals}
                                               for f, vals in r.metadata.items())
            # a document that applies to everyone ("all") states the general *value*, which holds for the condition
            # (a fee for all applicants is the fee for international ones); a list of requirements for everyone is
            # not the condition-specific part of the answer, so fact slots still need specific evidence
            meta_all = bool(r.metadata) and r.value_kind is not None and any("all" in fv.get(f, set())
                                                                             for f in r.metadata)
            if not (all(self.has(t, allt, r.alternatives) for t in r.constraint_terms) or meta_ok or meta_all):
                return False, [], None
        vals: list[str] = []
        if r.value_kind:
            rt = set(r.terms)
            cand = [(sum(1 for t in rt if self.has(t, st, r.alternatives)), s) for s, st in sents]
            cand = [(c, s) for c, s in cand if c and has_value_of(s, r.value_kind)]
            if not cand:
                return False, [], None
            best = max(c for c, _ in cand)
            for c, s in cand:
                if c == best:
                    vals += values_of(s, r.value_kind)
        if r.valid_at is not None and self.catalog.valid_at(e.document_id, r.valid_at, r.valid_to) is False:
            return False, [], "temporal_validity"
        return True, list(dict.fromkeys(vals)), None

    def assess(self, reqs: list[EvidenceRequirement], pool: list, temporal_kind: str = "none",
               threshold: float = 1.0) -> tuple[SufficiencyAssessment, dict[str, str]]:
        """Updates ``reqs`` in place; returns the assessment and {excluded evidence id: reason}."""
        excluded: dict[str, str] = {}
        qualities = []
        # applicability: a need constrained to a metadata value (applicant_type = domestic) cannot use evidence from
        # a document that names a different specific value (international); "all" / no value applies to everyone
        conds: dict[str, set[str]] = {}
        for r in reqs:
            for f, vals in r.metadata.items():
                conds.setdefault(f, set()).update(v.lower() for v in vals)
        if conds:
            kept = []
            for e in pool:
                meta = self.catalog.docs.get(e.document_id)
                bad = meta is not None and any(
                    meta.meta.get(f) is not None and not ({x.strip().lower() for x in str(meta.meta[f]).split(",")}
                                                          & (vals | {"all"})) for f, vals in conds.items())
                if bad:
                    excluded[e.evidence_id] = "not_applicable:" + ",".join(sorted(conds))
                else:
                    kept.append(e)
            pool = kept
        ev_doc = {e.evidence_id: e.document_id for e in pool}
        conds = [r for r in reqs if r.kind == "condition"]
        for r in conds:                                    # condition slots first: they decide applicability
            r.evidence_ids = [e.evidence_id for e in pool if self.supports(r, e)[0]]
        applicable = None if not conds else set.intersection(*[set(r.evidence_ids) for r in conds])
        for r in reqs:
            r.evidence_ids, r.values, r.resolution = [], {}, None
            best = 0.0
            for e in pool:
                best = max(best, self.coverage_of(r, e))
                ok, vals, why = self.supports(r, e)
                if ok:
                    r.evidence_ids.append(e.evidence_id)
                    if vals:
                        r.values[e.evidence_id] = vals
                elif why is not None:
                    excluded.setdefault(e.evidence_id, f"{why}:{r.requirement_id}")
            qualities.append(best)
            r.status = "MET" if r.evidence_ids and len({ev_doc[i] for i in r.evidence_ids}) >= r.min_sources \
                else "UNMET"
            if r.status == "MET" and r.value_kind and len(set(r.terms)) >= 2:   # one word: subject unspecified
                losers, how, unresolved = self._conflict(r, ev_doc, temporal_kind, applicable)
                if unresolved:
                    r.status, r.resolution = "CONFLICT", "unresolved"
                elif losers:
                    r.resolution = how
                    for i in losers:
                        excluded[i] = f"conflict_resolution:{how}:{r.requirement_id}"
                    r.evidence_ids = [i for i in r.evidence_ids if i not in losers]
                    r.values = {k: v for k, v in r.values.items() if k not in losers}
        excluded = {i: w for i, w in excluded.items() if not any(i in r.evidence_ids for r in reqs)}
        met = [r.requirement_id for r in reqs if r.status == "MET"]
        conflicts = [r.requirement_id for r in reqs if r.status == "CONFLICT"]
        unmet = [r.requirement_id for r in reqs if r.status == "UNMET"]
        cov = len(met) / len(reqs) if reqs else 0.0
        if conflicts:
            status = "CONTRADICTORY"
        elif reqs and cov >= threshold:
            status = "SUFFICIENT"
        elif met:
            status = "PARTIAL"
        else:
            status = "INSUFFICIENT"
        reasons = [f"{r.requirement_id}:{r.status}" + (f"({r.resolution})" if r.resolution else "") for r in reqs]
        return SufficiencyAssessment(
            status=status, coverage=round(cov, 4),
            quality=round(sum(qualities) / len(qualities), 4) if qualities else 0.0, met=met, unmet=unmet,
            conflicts=conflicts, unattainable=[r.requirement_id for r in reqs if r.status == "UNMET"
                                               and not self.attainable(r)],
            resolved_conflicts=[r.requirement_id for r in reqs if r.resolution and r.resolution != "unresolved"],
            reasons=reasons), excluded

    def _comparable(self, d1: str, d2: str) -> bool:
        """Two documents whose metadata name different specific values of a filter field (domestic vs
        international) state rules for different populations: different values are not a contradiction."""
        m1, m2 = self.catalog.docs.get(d1), self.catalog.docs.get(d2)
        if m1 is None or m2 is None:
            return True
        for f in self.catalog.filter_values:
            v1, v2 = m1.meta.get(f), m2.meta.get(f)
            if v1 is None or v2 is None:
                continue
            s1 = {x.strip().lower() for x in str(v1).split(",")}
            s2 = {x.strip().lower() for x in str(v2).split(",")}
            if "all" not in s1 | s2 and not s1 & s2:
                return False
        return True

    def _conflict(self, r: EvidenceRequirement, ev_doc: dict[str, str], temporal_kind: str,
                  applicable: set[str] | None = None) -> tuple[set[str], str | None, bool]:
        """Different values for one requirement in chunks of different documents -> (losers, rule, unresolved).
        With condition slots, only evidence that meets every condition is compared (values for another population
        or subject are not a contradiction)."""
        items = [(eid, ev_doc[eid], set(v)) for eid, v in r.values.items()
                 if applicable is None or eid in applicable]
        clash = [(a, b) for i, a in enumerate(items) for b in items[i + 1:]
                 if a[1] != b[1] and not a[2] & b[2] and self._comparable(a[1], b[1])]
        if not clash:
            return set(), None, False
        involved = sorted({x[1] for p in clash for x in p})
        keep, how = None, None
        sup = {d for d in involved if d in self.catalog.superseded and self.catalog.superseded[d] in involved}
        if sup:
            keep = sup if temporal_kind == "past" else {d for d in involved if d not in sup}
            how = "past_version" if temporal_kind == "past" else "version_supersession"
        if keep is None:
            auth = {d: self.catalog.authority(d) for d in involved}
            top = max(auth.values())
            if all(top - v >= 0.3 for v in auth.values() if v != top) and len(set(auth.values())) > 1:
                keep, how = {d for d, v in auth.items() if v == top}, "authority"
        if keep is None:
            return set(), None, True
        if any(a[1] in keep and b[1] in keep for a, b in clash):
            return set(), None, True
        kept_vals = set().union(*[v for _, d, v in items if d in keep])
        # losers: chunks of the other documents whose values disagree with every kept value (a chunk that also
        # states the kept value - "raised from 40 to 55" - is not a loser)
        return {eid for eid, d, v in items if d in involved and d not in keep and not v & kept_vals}, how, False
