"""GroundedAnswerEngine: evidence -> claims -> grounded answer -> validated claims -> citations -> answer state
(Phase 7; docs/answer/, ADR-017).

Per answer version of a topic frame (the Phase 6 ``AnswerVersion`` decides *what changed*):

  1 ClaimPlanner        evidence-derived planned facts per need, gaps, conflicts            CLAIM_PLAN_CREATED
  2 AnswerPlanner       sections, detail, order, what to reuse
  3 reuse               sections Phase 6 marks unchanged are reused verbatim; inside a regenerated section, sentences
                        whose facts are still planned and whose evidence is still usable are kept (re-verified)
  4 generation          one LLM call for the facts that still need a sentence (structured), or extractive
                                                                ANSWER_GENERATION_STARTED / LLM_CALL / _COMPLETED
  5 extraction          one candidate claim per generated sentence                          CLAIMS_EXTRACTED
  6 verification        entailment + rules against the need's usable evidence  CLAIM_VERIFICATION_STARTED, CLAIM_VERIFIED
  7 policy / repair     keep / conflict / keep atoms / restore facts / retrieval fallback / remove
                        (``generation.repair = false``: validation only - anything not supported is removed)
                                                          CLAIM_REJECTED, CLAIM_REPAIRED, VALIDATION_RETRIEVAL
  8 revision            strict: a section that produced unsupported content is regenerated once with feedback,
                        then rendered extractively; both modes: missing critical facts are added verbatim
  9 consistency         new claims must not contradict kept claims
 10 citations           from the verification, to the supporting sentence; validated against the index
                                                                      CITATION_CREATED, CITATION_VALIDATED
 11 coverage / status   every need covered by facts or explicit uncertainty; DRAFT / VALIDATED_FINAL / BLOCKED
                                                                      ANSWER_VALIDATED (+ ANSWER_REVISED)
 12 streaming contract  ANSWER_STARTED, ANSWER_SECTION_STARTED, ANSWER_CLAIM_READY, ANSWER_CITATION_READY,
                        ANSWER_SECTION_COMPLETED, ANSWER_COMPLETED (+ ANSWER_FINALIZED)

Nothing unverified is ever rendered as a fact: a factual claim reaches the answer only with a SUPPORTED verification
and a valid citation (uncertainty sentences are typed and never cite). Drafts (``draft=True``) use the extractive
generator, are verified the same way and are marked DRAFT.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from streamrag.answer_state.models import (
    AnswerClaim,
    AnswerClaimDiff,
    GroundedAnswer,
    GroundedSection,
    RejectedClaim,
    RepairRecord,
)
from streamrag.answer_state.render import render_answer
from streamrag.citations.mapper import ChunkCatalog, CitationMapper
from streamrag.citations.validator import CitationValidator
from streamrag.claims.aligner import ClaimEvidenceAligner
from streamrag.claims.decomposer import ClaimDecomposer, ClaimLexicon
from streamrag.claims.models import ClaimVerification, IntentGap
from streamrag.claims.planner import ClaimPlanner
from streamrag.claims.verifier import ClaimVerifier
from streamrag.generation.answer_planner import AnswerPlanner
from streamrag.generation.extraction import ExtractedClaim, GeneratedClaimExtractor, fact_key
from streamrag.generation.generator import GroundedAnswerGenerator
from streamrag.generation.llm import LLMResponse
from streamrag.models.events import EventType as E
from streamrag.validation.consistency import ConsistencyChecker
from streamrag.validation.coverage import AnswerCoverageValidator
from streamrag.validation.metrics import grounding_metrics
from streamrag.validation.policy import decide
from streamrag.validation.repair import ClaimRepairer, gap_sentence

LEXICON = Path(__file__).resolve().parents[3] / "configs" / "claim_lexicon.yaml"
SCOPE_FIELDS = ("applicant_type", "country", "region", "product", "language")   # = adaptive_retrieval.filter_fields
_STAGES = ("claim_planning", "generation", "claim_extraction", "claim_verification", "repair", "retrieval",
           "citation_mapping", "validation", "total")


def _noop(*_a, **_k):
    return None


def _norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def claim_id_for(section_id: str, text: str) -> str:
    norm = " ".join(text.lower().split())
    return "AC-" + hashlib.sha1(f"{section_id}|{norm}".encode()).hexdigest()[:10]


@dataclass
class GroundingContext:
    """What the engine reads from the session (Phase 6 state) for one answer."""

    graph: object
    store: object
    tracker: object
    conflicts: dict
    utterance_id: str
    retrieve_fn: Callable[[str, str], list[str]] | None = None   # (intent id, query) -> new evidence ids (stored)


class GroundedAnswerEngine:
    def __init__(self, gcfg, terms_fn: Callable[[str], list[str]], catalog: ChunkCatalog, nli=None, backend=None,
                 emit: Callable | None = None, lexicon_path: Path = LEXICON) -> None:
        self.cfg = gcfg
        mode = "nli" if (nli is not None and gcfg.verifier == "nli") else "rules"
        self.aligner = ClaimEvidenceAligner(terms_fn, nli if mode == "nli" else None, mode)
        self.decomposer = ClaimDecomposer(ClaimLexicon.load(lexicon_path))
        self.verifier = ClaimVerifier(self.aligner, self.decomposer)
        self.catalog = catalog
        self.claim_planner = ClaimPlanner(self.decomposer, self.aligner, catalog.is_boilerplate)
        self.answer_planner = AnswerPlanner(gcfg.detail, gcfg.max_claims_per_section)
        self.backend = backend
        self.generator = GroundedAnswerGenerator(backend, gcfg.max_structured_retries, on_call=self._on_llm_call,
                                                 answerability=gcfg.answerability)
        self.extractor = GeneratedClaimExtractor(terms_fn)
        self.repairer = ClaimRepairer(backend if gcfg.llm_repair else None)
        self.mapper = CitationMapper(catalog)
        self.citation_validator = CitationValidator(catalog)
        self.coverage = AnswerCoverageValidator()
        self.consistency = ConsistencyChecker(terms_fn, nli if mode == "nli" else None)
        self.terms_fn = terms_fn
        self.emit = emit or _noop
        self.versions: list[GroundedAnswer] = []
        self.current: dict[str, GroundedAnswer] = {}           # latest FINAL answer per frame (reuse + diff base)
        self.drafts: dict[str, GroundedAnswer] = {}            # latest draft per frame since that final
        self.llm_records: list[dict] = []
        self._uid: str | None = None
        self._version = 1
        self.cancel_check = None                       # Phase 8: called between stages; raises when cancelled

    # ------------------------------------------------------------------ Phase 8: runtime support
    def checkpoint(self) -> tuple:
        """The engine's own mutable state (answers are immutable models; containers are copied)."""
        return (list(self.versions), dict(self.current), dict(self.drafts), list(self.llm_records), self._version,
                self._uid)

    def restore(self, cp: tuple) -> None:
        """Undo an answer that was cancelled or discarded as stale (it never happened for the session)."""
        versions, current, drafts, llm_records, version, uid = cp
        self.versions, self.current, self.drafts = list(versions), dict(current), dict(drafts)
        self.llm_records, self._version, self._uid = list(llm_records), version, uid

    def _check(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    # ------------------------------------------------------------------ telemetry helpers
    def _on_llm_call(self, purpose: str, r: LLMResponse, attempt: int) -> None:
        rec = {"purpose": purpose, "attempt": attempt, "backend": r.backend, "model": r.model, "ok": r.ok,
               "request_sha1": r.request_sha1, "output": r.text, "error": r.error, "prompt_tokens": r.prompt_tokens,
               "output_tokens": r.output_tokens}
        self.llm_records.append(rec)
        self.emit(E.LLM_CALL, "llm_gateway", {**rec, "wall": {"ttft_ms": r.ttft_ms, "total_ms": round(r.total_ms, 3),
                                                              **{k: v for k, v in r.meta.items()}}}, self._uid)

    # ------------------------------------------------------------------ main entry
    def answer(self, p6, ctx: GroundingContext, now: float, draft: bool = False) -> GroundedAnswer:
        T = {k: 0.0 for k in _STAGES}
        t_all = time.perf_counter()
        self._uid = ctx.utterance_id
        prev = (self.drafts.get(p6.topic_id) or self.current.get(p6.topic_id)) if draft else self.current.get(p6.topic_id)
        last = self.drafts.get(p6.topic_id) or self.current.get(p6.topic_id)
        n = len(self.versions) + 1
        aid = f"GA{n}"
        version = (last.version + 1) if last else 1
        self._version = version
        self._edges: list[tuple[str, str]] = []
        emit = self.emit
        # 1-2 plans ----------------------------------------------------------------------------------------------
        t = time.perf_counter()
        cp = self.claim_planner.plan(p6, ctx.graph, ctx.store, ctx.tracker, ctx.conflicts)
        pools: dict[str, dict[str, str]] = {}
        order: list[str] = []
        for ip in cp.intents:
            ids = [a.evidence_id for a in ctx.store.usable(ip.intent_id)]
            ids += [e for f in ip.claims for e in f.evidence_ids if e not in ids]
            pools[ip.section_id] = {e: ctx.store.text(e) for e in ids}
            order += [e for e in ids if e not in order]
        label_of = {e: f"E{k + 1}" for k, e in enumerate(order)}
        ap = self.answer_planner.plan(cp, label_of)
        T["claim_planning"] += (time.perf_counter() - t) * 1000.0
        emit(E.CLAIM_PLAN_CREATED, "claim_planner", {
            "answer_id": aid, "version": version, "plan_id": cp.plan_id, "answer_plan_id": ap.answer_plan_id,
            "phase6_answer_id": p6.answer_id, "draft": draft,
            "sections": [{"section_id": s.section_id, "intent_id": s.intent_id, "regenerate": s.regenerate,
                          "facts": [{"id": f.plan_claim_id, "importance": f.importance, "evidence_ids": f.evidence_ids,
                                     "text": f.text} for f in s.facts],
                          "omitted": s.omitted_facts, "gaps": [g.model_dump(mode="json") for g in s.uncertainties]}
                         for s in ap.sections], "conflicts": [list(c) for c in cp.conflicts]}, ctx.utterance_id)
        # 3 reuse ---------------------------------------------------------------------------------------------------
        claims: dict[str, AnswerClaim] = {}
        verifs: dict[str, ClaimVerification] = {}
        sec_claims: dict[str, list[str]] = {s.section_id: [] for s in ap.sections}
        reused_sections, regenerated = [], []
        need_facts: dict[str, list[str]] = {}
        kept_text: dict[str, list[str]] = {}
        t = time.perf_counter()
        for s in ap.sections:
            pool = pools[s.section_id]
            prev_sec = next((x for x in prev.sections if x.section_id == s.section_id), None) if prev else None
            planned_keys = {fact_key(f): f for f in s.facts}
            kept = []
            if prev_sec is not None:
                for cid in prev_sec.claim_ids:
                    c = prev.claim(cid)
                    if c is None or c.kind != "fact":
                        continue
                    if c.fact_keys and not set(c.fact_keys) <= set(planned_keys):
                        continue                              # its facts are no longer planned
                    if not c.fact_keys and s.regenerate:
                        continue
                    self._check()
                    v = self.verifier.verify(cid, c.text, c.evidence_ids, pool, decompose=False)
                    if v.status != "SUPPORTED":
                        continue                              # evidence no longer usable / changed
                    kept.append((c, v))
            if prev_sec is not None and not s.regenerate and len(kept) == sum(
                    1 for cid in prev_sec.claim_ids if prev.claim(cid) and prev.claim(cid).kind == "fact"):
                reused_sections.append(s.section_id)
            else:
                regenerated.append(s.section_id)
            covered = set()
            rank = {"critical": 0, "important": 1, "supplementary": 2}
            for c, v in kept:
                fs = [planned_keys[k] for k in c.fact_keys]
                imp = min((f.importance for f in fs), key=lambda x: rank[x]) if fs else c.importance
                claims[c.claim_id] = c.model_copy(update={"origin": "kept", "importance": imp,
                                                          "facts": [f.plan_claim_id for f in fs],
                                                          "evidence_ids": list(v.supporting_evidence)})
                verifs[c.claim_id] = v
                sec_claims[s.section_id].append(c.claim_id)
                covered |= set(c.fact_keys)
            kept_text[s.section_id] = [c.text for c, _ in kept]
            if s.section_id in regenerated:
                need_facts[s.section_id] = [f.plan_claim_id for f in s.facts if fact_key(f) not in covered]
        T["claim_verification"] += (time.perf_counter() - t) * 1000.0
        self._check()
        # 4 generation ------------------------------------------------------------------------------------------------
        gen_backend = None if draft else self.backend
        stats = {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "ttft": None, "fallback": None,
                 "backend": "extractive" if gen_backend is None else self.backend.name}
        todo = {k: v for k, v in need_facts.items() if v}
        s_by_id = {x.section_id: x for x in ap.sections}
        raw_statuses: list[str] = []
        rejected: list[RejectedClaim] = []
        repairs: list[RepairRecord] = []
        budget = {"retrievals": 0 if draft else self.cfg.max_validation_retrievals, "used": 0}
        revisions = 0
        if todo:
            emit(E.ANSWER_GENERATION_STARTED, "answer_generator", {
                "answer_id": aid, "version": version, "backend": stats["backend"], "sections": sorted(todo),
                "facts": todo, "kept_sentences": {k: len(v) for k, v in kept_text.items() if v}}, ctx.utterance_id)
            extracted = self._generate(ap, todo, kept_text, None, gen_backend, stats, T)
            self._check()
            emit(E.ANSWER_GENERATION_COMPLETED, "answer_generator", {
                "answer_id": aid, "version": version, "backend": stats["backend"], "fallback": stats["fallback"],
                "sentences": len(extracted), "llm_calls": stats["llm_calls"],
                "tokens": {"prompt": stats["prompt_tokens"], "output": stats["output_tokens"]},
                "wall": {"generation_ms": round(T["generation"], 3), "ttft_raw_ms": stats["ttft"]}}, ctx.utterance_id)
            removed_by_sec = self._process(extracted, ap, pools, ctx, claims, verifs, sec_claims, rejected, repairs,
                                           raw_statuses, budget, T, aid, version)
            # 8 revision (strict: unsupported content; both: missing critical facts) ---------------------------------
            while gen_backend is not None and self.cfg.repair and revisions < self.cfg.max_answer_revision_attempts:
                missing = self._missing_facts(ap, claims, sec_claims)
                redo = {sid: missing.get(sid, []) for sid in todo if sid not in stats.get("unanswered", ())
                        and ((self.cfg.validation_mode == "strict" and removed_by_sec.get(sid))
                        or any(f.importance == "critical" for f in self._facts(ap, sid, missing.get(sid, []))))}
                redo = {k: v for k, v in redo.items() if v}
                if not redo:
                    break
                revisions += 1
                avoid = {sid: removed_by_sec.get(sid, []) for sid in redo}
                kept_now = {sid: [claims[c].text for c in sec_claims[sid] if claims[c].kind == "fact"] for sid in redo}
                self._check()
                extracted = self._generate(ap, redo, kept_now, avoid, gen_backend, stats, T)
                self._check()
                removed_by_sec = self._process(extracted, ap, pools, ctx, claims, verifs, sec_claims, rejected,
                                               repairs, raw_statuses, budget, T, aid, version)
            # final completion: planned critical facts (and, strict, every planned fact of a section that produced
            # unsupported content) that are still missing are added verbatim
            missing = self._missing_facts(ap, claims, sec_claims) if (self.cfg.repair or gen_backend is None) else {}
            for sid in sorted(stats.get("unanswered", ())):
                missing.pop(sid, None)
                self._drop_unanswered(sid, s_by_id.get(sid), claims, sec_claims, rejected, aid, version, ctx)
            for sid, fids in missing.items():
                strict_fill = self.cfg.validation_mode == "strict" and any(r.section_id == sid for r in rejected)
                fill = [f for f in self._facts(ap, sid, fids) if f.importance == "critical" or strict_fill
                        or gen_backend is None]
                for f in fill:
                    self._add_fact(f, sid, ap, pools, claims, verifs, sec_claims, "extractive", None, T)
        self._check()
        # 9 consistency ---------------------------------------------------------------------------------------------
        t = time.perf_counter()
        consistency_pairs: list[tuple[str, str]] = []
        for s in ap.sections:
            ids = sec_claims[s.section_id]
            new = [(c, claims[c].text) for c in ids if claims[c].origin != "kept" and claims[c].kind == "fact"]
            old = [(c, claims[c].text) for c in ids if claims[c].origin == "kept" and claims[c].kind == "fact"]
            for a, b in self.consistency.conflicts(new, old):
                if a not in sec_claims[s.section_id]:
                    continue                            # already removed: one claim can contradict several others
                consistency_pairs.append((a, b))
                ea, eb = set(claims[a].evidence_ids), set(claims[b].evidence_ids)
                if ea and eb and not ea & eb:                    # different sources disagree: present both
                    self._edges.append((a, b))
                else:
                    rejected.append(RejectedClaim(claim_id=a, section_id=s.section_id, text=claims[a].text,
                                                  status="CONSISTENCY_CONFLICT", action="remove",
                                                  reasons=[f"contradicts_retained_claim:{b}"],
                                                  importance=claims[a].importance, origin=claims[a].origin))
                    sec_claims[s.section_id].remove(a)
                    emit(E.CLAIM_REJECTED, "answer_validator", {"answer_id": aid, "version": version, "claim_id": a,
                                                               "reason": "consistency_conflict", "with": b},
                         ctx.utterance_id, intent_id=s.intent_id)
        T["validation"] += (time.perf_counter() - t) * 1000.0
        # conflict groups (verifier, Phase 6 numeric conflicts, consistency) -> both sides presented
        self._mark_conflicts(cp, ap, claims, sec_claims)
        # uncertainty statements (deterministic) ------------------------------------------------------------------
        for s in ap.sections:
            facts_here = [c for c in sec_claims[s.section_id] if claims[c].kind in ("fact", "conflict")]
            gaps = [g for g in s.uncertainties if g.kind != "conflict"]
            if not facts_here and not gaps:
                gaps = [IntentGap(intent_id=s.intent_id, kind="no_evidence", aspect=s.title)]
            for g in gaps:
                text = gap_sentence(g)
                cid = claim_id_for(s.section_id, text)
                claims[cid] = AnswerClaim(claim_id=cid, section_id=s.section_id, intent_id=s.intent_id, text=text,
                                          kind="uncertainty", origin="template", importance="important",
                                          status="UNCERTAINTY", introduced_in=prev.claim(cid).introduced_in
                                          if prev and prev.claim(cid) else version)
                sec_claims[s.section_id].append(cid)
        # 10 citations ----------------------------------------------------------------------------------------------
        t = time.perf_counter()
        in_answer = {c: verifs[c] for sid in sec_claims for c in sec_claims[sid]
                     if claims[c].kind in ("fact", "conflict") and c in verifs}
        cmap = self.mapper.map(aid, in_answer, label_of)
        T["citation_mapping"] += (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        ev_text = {e: x for p in pools.values() for e, x in p.items()}
        section_ev = [e for s in ap.sections for e in pools[s.section_id]]
        cmap, crep = self.citation_validator.validate(cmap, in_answer, list(in_answer), ev_text, section_ev)
        for cid in crep.orphan_claims:                       # a fact without a valid citation never stays
            sid = claims[cid].section_id
            rejected.append(RejectedClaim(claim_id=cid, section_id=sid, text=claims[cid].text, status="NO_CITATION",
                                          action="remove", reasons=["no_valid_citation"],
                                          importance=claims[cid].importance, origin=claims[cid].origin))
            sec_claims[sid] = [c for c in sec_claims[sid] if c != cid]
        for cid, cits in cmap.claim_to_citations.items():
            if cid in claims:
                claims[cid] = claims[cid].model_copy(update={"citation_ids": cits})
        T["validation"] += (time.perf_counter() - t) * 1000.0
        # 11 coverage / status -------------------------------------------------------------------------------------
        t = time.perf_counter()
        sections = []
        for s in ap.sections:
            ids = list(dict.fromkeys(sec_claims[s.section_id]))
            ids.sort(key=lambda c: (claims[c].kind == "uncertainty", self._rank(claims[c], s)))
            sections.append(GroundedSection(section_id=s.section_id, intent_id=s.intent_id, title=s.title,
                                            order=s.order, claim_ids=ids, regenerated=s.section_id in regenerated,
                                            uncertainty=s.uncertainties))
        final_claims = {c: claims[c] for s in sections for c in s.claim_ids}
        unanswered = stats.get("unanswered", set())          # handled by an explicit uncertainty statement instead
        critical = {s.section_id: [] if s.section_id in unanswered else
                    [f.plan_claim_id for f in s.facts if f.importance == "critical"] for s in ap.sections}
        expressed = {s.section_id: {f for c in s.claim_ids for f in final_claims[c].facts} for s in sections}
        cov = self.coverage.validate(sections, final_claims, critical, expressed)
        blocked = [f"intent_not_handled:{i}" for i in cov.failures]
        if cov.critical_facts_missing and not draft:
            blocked += [f"critical_fact_missing:{k}" for k in cov.critical_facts_missing]
        status = "DRAFT" if draft else ("BLOCKED" if blocked else "VALIDATED_FINAL")
        partial = bool(cov.uncertain_only) or any(c.kind == "uncertainty" for c in final_claims.values())
        T["validation"] += (time.perf_counter() - t) * 1000.0
        # render, diff, version ---------------------------------------------------------------------------------
        sections, text = render_answer(sections, final_claims, cmap, self._doc_info)
        diff = self._diff(prev, sections, final_claims, regenerated, reused_sections)
        prev_ids = {x.section_id: set(x.claim_ids) for x in prev.sections} if prev else {}
        sections = [s.model_copy(update={"status": "new" if s.section_id not in prev_ids else
                                         ("unchanged" if set(s.claim_ids) == prev_ids[s.section_id] else "changed")})
                    for s in sections]
        T["total"] = (time.perf_counter() - t_all) * 1000.0
        ga = GroundedAnswer(
            answer_id=aid, version=version, parent_answer_id=prev.answer_id if prev else None, frame_id=p6.topic_id,
            phase6_answer_id=p6.answer_id, utterance_id=ctx.utterance_id, status=status, partial=partial,
            mode=self.cfg.validation_mode, sections=sections, claims=list(final_claims.values()),
            verifications={c: verifs[c] for c in final_claims if c in verifs}, rejected=rejected, repairs=repairs,
            citations=cmap, citation_report=crep, coverage=cov, consistency_conflicts=consistency_pairs,
            blocked_reasons=blocked, diff=diff, text=text, backend=stats["backend"],
            model=getattr(self.backend, "model", None) if stats["backend"] != "extractive" else None,
            fallback=stats["fallback"], llm_calls=stats["llm_calls"], prompt_tokens=stats["prompt_tokens"],
            output_tokens=stats["output_tokens"], ttft_raw_ms=stats["ttft"], validation_retrievals=budget["used"],
            revision_attempts=revisions, verifier=self.aligner.name,
            timings_ms={k: round(v, 3) for k, v in T.items()}, created_at_ms=now)
        ga = ga.model_copy(update={"metrics": grounding_metrics(raw_statuses, ga)})
        self.versions.append(ga)
        if draft:
            self.drafts[p6.topic_id] = ga
        else:
            self.current[p6.topic_id] = ga
            self.drafts.pop(p6.topic_id, None)
        self._emit_answer(ga, cmap, crep, cov, ctx.utterance_id)
        return ga

    # ------------------------------------------------------------------ generation + processing
    def _facts(self, ap, sid: str, ids: list[str] | None):
        sec = next(s for s in ap.sections if s.section_id == sid)
        return [f for f in sec.facts if ids is None or f.plan_claim_id in ids]

    def _drop_unanswered(self, sid, sec, claims, sec_claims, rejected, aid, version, ctx) -> None:
        """The model judged that the section's facts do not answer the user's need (answerability, Phase 11): its
        fact claims - including ones kept from a draft - are withdrawn, so the section ends with the deterministic
        'not in the retrieved documents' statement instead of adjacent facts presented as the answer."""
        for cid in [c for c in sec_claims[sid] if claims[c].kind in ("fact", "conflict")]:
            sec_claims[sid].remove(cid)
            rejected.append(RejectedClaim(claim_id=cid, section_id=sid, text=claims[cid].text, status="NOT_ANSWERING",
                                          action="remove", reasons=["facts_do_not_answer_the_need"],
                                          importance=claims[cid].importance, origin=claims[cid].origin))
            self.emit(E.CLAIM_REJECTED, "answer_validator", {"answer_id": aid, "version": version, "claim_id": cid,
                                                           "reason": "facts_do_not_answer_the_need"},
                      ctx.utterance_id, intent_id=sec.intent_id if sec else None)

    def _missing_facts(self, ap, claims, sec_claims) -> dict[str, list[str]]:
        out = {}
        for s in ap.sections:
            have = {k for c in sec_claims[s.section_id] for k in claims[c].fact_keys
                    if claims[c].kind in ("fact", "conflict")}
            miss = [f.plan_claim_id for f in s.facts if fact_key(f) not in have]
            if miss:
                out[s.section_id] = miss
        return out

    def _generate(self, ap, todo, kept, avoid, backend, stats, T) -> list[ExtractedClaim]:
        t = time.perf_counter()
        if backend is None:
            cand = self.generator.extractive.generate(ap.model_copy(update={"sections": [
                s.model_copy(update={"facts": [f for f in s.facts if f.plan_claim_id in todo[s.section_id]]})
                for s in ap.sections if s.section_id in todo]}))
        else:
            cand = self.generator.generate(ap, sections=list(todo), kept=kept, avoid=avoid, facts=todo)
            stats["llm_calls"] += cand.llm_calls
            stats["prompt_tokens"] += cand.prompt_tokens
            stats["output_tokens"] += cand.output_tokens
            if stats["ttft"] is None:
                stats["ttft"] = cand.ttft_ms
            if cand.fallback:
                stats["fallback"] = cand.fallback
                stats["backend"] = f"{self.backend.name}->extractive"
            stats.setdefault("unanswered", set()).update(cand.unanswered_sections)
        T["generation"] += (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        ex = self.extractor.extract(cand, ap)
        T["claim_extraction"] += (time.perf_counter() - t) * 1000.0
        self.emit(E.CLAIMS_EXTRACTED, "claim_extractor", {
            "claims": [{"key": c.claim_key, "section_id": c.section_id, "text": c.text, "kind": c.kind,
                        "facts": c.facts, "cited_evidence": c.cited_evidence, "invalid_labels": c.invalid_labels,
                        "unknown_facts": c.unknown_facts, "importance": c.importance, "origin": c.origin}
                       for c in ex]}, self._uid)
        return ex

    def _process(self, extracted, ap, pools, ctx, claims, verifs, sec_claims, rejected, repairs, raw_statuses,
                 budget, T, aid, version) -> dict[str, list[str]]:
        removed: dict[str, list[str]] = {}
        sec_by = {s.section_id: s for s in ap.sections}
        self.emit(E.CLAIM_VERIFICATION_STARTED, "claim_verifier", {
            "answer_id": aid, "version": version, "claims": len([e for e in extracted if e.kind == "fact"]),
            "verifier": self.aligner.name}, self._uid)
        for ec in extracted:
            self._check()                                  # cooperative cancellation: per claim
            if ec.kind == "connective":
                continue
            s = sec_by[ec.section_id]
            pool = pools[ec.section_id]
            t = time.perf_counter()
            cid = claim_id_for(ec.section_id, ec.text)
            v = self.verifier.verify(cid, ec.text, ec.cited_evidence, pool)
            T["claim_verification"] += (time.perf_counter() - t) * 1000.0
            raw_statuses.append("CONFLICT" if v.status == "CONTRADICTED" and v.supporting_evidence else v.status)
            action = decide(v, ec.text, ec.facts, budget["retrievals"] - budget["used"])
            if not self.cfg.repair and action not in ("keep",):
                action = "remove"                             # validation only (ablation): no repair of any kind
            ev_payload = {"answer_id": aid, "version": version, "claim_id": cid, "text": ec.text, "status": v.status,
                          "supporting_evidence": v.supporting_evidence,
                          "contradicting_evidence": v.contradicting_evidence, "cited_evidence": v.cited_evidence,
                          "invalid_labels": ec.invalid_labels, "reasons": v.reasons, "action": action,
                          "importance": ec.importance, "atoms": v.atom_texts}
            self.emit(E.CLAIM_VERIFIED if v.status == "SUPPORTED" else E.CLAIM_REJECTED, "claim_verifier", ev_payload,
                      self._uid, intent_id=s.intent_id)
            t = time.perf_counter()
            if action == "keep":
                self._put(cid, ec.text, ec, s, v, claims, verifs, sec_claims, "llm" if ec.origin == "llm" else ec.origin)
            elif action == "present_conflict":
                pair = ClaimRepairer.conflict_pair(v, pool)
                sides = []
                for text, eid in pair:
                    vv = self.verifier.verify(claim_id_for(s.section_id, text), text, [eid], {eid: pool[eid]},
                                              decompose=False)
                    if vv.supported:
                        # a side states planned facts verbatim: they count as expressed (no revision for them)
                        same = [f for f in self._facts(ap, s.section_id, None) if _norm(f.text) == _norm(text)]
                        self._put(vv.claim_id, text, ec, s, vv, claims, verifs, sec_claims, "repair",
                                  kind="conflict", status="CONTRADICTED", repaired_from=cid,
                                  facts=[f.plan_claim_id for f in same], keys=[fact_key(f) for f in same])
                        sides.append(vv.claim_id)
                if len(sides) == 2:
                    self._edges.append((sides[0], sides[1]))
                repairs.append(RepairRecord(claim_id=cid, action="present_conflict", from_text=ec.text,
                                            to_texts=[p[0] for p in pair], ok=len(pair) == 2))
            elif action == "keep_atoms":
                atoms = ClaimRepairer.keep_atoms(v)
                for text, av in atoms:
                    self._put(claim_id_for(s.section_id, text), text, ec, s, av, claims, verifs, sec_claims, "repair",
                              repaired_from=cid, facts=[])
                dropped = [t2 for t2, a in zip(v.atom_texts, v.atoms) if not a.supported]
                for t2 in dropped:
                    rejected.append(RejectedClaim(claim_id=claim_id_for(s.section_id, t2), section_id=s.section_id,
                                                  text=t2, status="UNSUPPORTED", action="remove",
                                                  reasons=["atom_not_supported"], importance=ec.importance,
                                                  origin=ec.origin))
                removed.setdefault(s.section_id, []).extend(dropped)
                repairs.append(RepairRecord(claim_id=cid, action="keep_atoms", from_text=ec.text,
                                            to_texts=[a for a, _ in atoms], ok=bool(atoms),
                                            detail=f"dropped: {dropped}"))
            elif action == "restore_facts":
                fs = self._facts(ap, s.section_id, ec.facts)
                done = [self._add_fact(f, s.section_id, ap, pools, claims, verifs, sec_claims, "repair", cid, T)
                        for f in fs]
                rejected.append(RejectedClaim(claim_id=cid, section_id=s.section_id, text=ec.text, status=v.status,
                                              action="restore_facts", reasons=v.reasons, importance=ec.importance,
                                              origin=ec.origin))
                removed.setdefault(s.section_id, []).append(ec.text)
                repairs.append(RepairRecord(claim_id=cid, action="restore_facts", from_text=ec.text,
                                            to_texts=[f.text for f in fs], ok=all(done)))
            elif action == "retrieve":
                t_r = time.perf_counter()
                new = ctx.retrieve_fn(s.intent_id, ec.text) if ctx.retrieve_fn else []
                budget["used"] += 1
                T["retrieval"] += (time.perf_counter() - t_r) * 1000.0
                for e in new:
                    pool[e] = ctx.store.text(e)
                vv = self.verifier.verify(cid, ec.text, ec.cited_evidence + new, pool) if new else v
                self.emit(E.VALIDATION_RETRIEVAL, "answer_validator", {
                    "answer_id": aid, "version": version, "claim_id": cid, "query": ec.text, "new_evidence": new,
                    "status_after": vv.status}, self._uid, intent_id=s.intent_id)
                if vv.status == "SUPPORTED":
                    self._put(cid, ec.text, ec, s, vv, claims, verifs, sec_claims, "retrieval")
                    repairs.append(RepairRecord(claim_id=cid, action="retrieve", from_text=ec.text,
                                                to_texts=[ec.text], ok=True, detail=f"new evidence {new}"))
                else:
                    rejected.append(RejectedClaim(claim_id=cid, section_id=s.section_id, text=ec.text,
                                                  status=vv.status, action="retrieve_then_remove", reasons=vv.reasons,
                                                  importance=ec.importance, origin=ec.origin))
                    removed.setdefault(s.section_id, []).append(ec.text)
                    repairs.append(RepairRecord(claim_id=cid, action="retrieve", from_text=ec.text, ok=False,
                                                detail="still unsupported after retrieval"))
            else:
                rewritten = self.repairer.llm_rewrite(ec.text, v, pool) if self.repairer.llm else None
                if rewritten:
                    vv = self.verifier.verify(claim_id_for(s.section_id, rewritten), rewritten, ec.cited_evidence, pool)
                    if vv.status == "SUPPORTED":
                        self._put(vv.claim_id, rewritten, ec, s, vv, claims, verifs, sec_claims, "repair",
                                  repaired_from=cid)
                        repairs.append(RepairRecord(claim_id=cid, action="llm_rewrite", from_text=ec.text,
                                                    to_texts=[rewritten], ok=True))
                        T["repair"] += (time.perf_counter() - t) * 1000.0
                        continue
                rejected.append(RejectedClaim(claim_id=cid, section_id=s.section_id, text=ec.text, status=v.status,
                                              action="remove", reasons=v.reasons, importance=ec.importance,
                                              origin=ec.origin))
                removed.setdefault(s.section_id, []).append(ec.text)
            T["repair"] += (time.perf_counter() - t) * 1000.0 if action != "keep" else 0.0
            if action not in ("keep", "remove"):
                self.emit(E.CLAIM_REPAIRED, "claim_repairer", {"answer_id": aid, "version": version,
                                                               "claim_id": cid, "action": action,
                                                               "repair": repairs[-1].model_dump(mode="json")},
                          self._uid, intent_id=s.intent_id)
        return removed

    def _put(self, cid, text, ec, s, v, claims, verifs, sec_claims, origin, kind="fact", group=None,
             status="SUPPORTED", repaired_from=None, facts=None, keys=None):
        if cid in claims and cid in sec_claims[s.section_id]:
            return
        fs = ec.facts if facts is None else facts
        keys = (ec.fact_keys if facts is None else []) if keys is None else keys
        claims[cid] = AnswerClaim(claim_id=cid, section_id=s.section_id, intent_id=s.intent_id, text=text, kind=kind,
                                  origin=origin, importance=ec.importance, facts=fs, fact_keys=keys, status=status,
                                  evidence_ids=list(v.supporting_evidence), repaired_from=repaired_from,
                                  conflict_group=group, introduced_in=self._version)
        verifs[cid] = v
        sec_claims[s.section_id].append(cid)

    def _add_fact(self, f, sid, ap, pools, claims, verifs, sec_claims, origin, repaired_from, T) -> bool:
        s = next(x for x in ap.sections if x.section_id == sid)
        cid = claim_id_for(sid, f.text)
        if cid in claims and cid in sec_claims[sid]:
            claims[cid] = claims[cid].model_copy(update={"facts": list(dict.fromkeys(claims[cid].facts + [f.plan_claim_id])),
                                                         "fact_keys": list(dict.fromkeys(claims[cid].fact_keys +
                                                                                         [fact_key(f)]))})
            return True
        self._check()
        t = time.perf_counter()
        v = self.verifier.verify(cid, f.text, f.evidence_ids, pools[sid], decompose=False)
        T["claim_verification"] += (time.perf_counter() - t) * 1000.0
        if not v.supported:
            return False
        claims[cid] = AnswerClaim(claim_id=cid, section_id=sid, intent_id=s.intent_id, text=f.text, kind="fact",
                                  origin=origin, importance=f.importance, facts=[f.plan_claim_id],
                                  fact_keys=[fact_key(f)], status="SUPPORTED", evidence_ids=list(v.supporting_evidence),
                                  repaired_from=repaired_from, introduced_in=self._version)
        verifs[cid] = v
        sec_claims[sid].append(cid)
        return True

    def _mark_conflicts(self, cp, ap, claims, sec_claims) -> None:
        """Evidence conflicts -> conflict groups: edges from the verifier (present_conflict), from Phase 6 numeric
        conflicts between planned facts and from the consistency check; groups = connected components."""
        live = {c for s in ap.sections for c in sec_claims[s.section_id]}
        for a, b in cp.conflicts:
            ca = [c for c in live if a in claims[c].facts]
            cb = [c for c in live if b in claims[c].facts]
            if ca and cb:
                self._edges.append((ca[0], cb[0]))
        # documents scoped to different groups (applicant type, country, ...) apply to different people and cannot
        # contradict each other: "domestic: 3.0" vs "international: 3.3" is not a conflict (Phase 11)
        self._edges = [(a, b) for a, b in self._edges
                       if a not in claims or b not in claims or not self._different_scope(claims[a], claims[b])]
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in self._edges:
            if a in live and b in live and a != b:
                parent[find(a)] = find(b)
        groups: dict[str, list[str]] = {}
        for x in list(parent):
            if x in live:
                groups.setdefault(find(x), []).append(x)
        for members in groups.values():
            if len(members) < 2:
                continue
            g = "K-" + min(members)
            for c in members:
                claims[c] = claims[c].model_copy(update={"kind": "conflict", "conflict_group": g,
                                                         "status": "CONTRADICTED"})
        for c in live:                                   # a lone "conflict" side (its partner was removed)
            if claims[c].kind == "conflict" and not any(c in m for m in groups.values() if len(m) > 1):
                claims[c] = claims[c].model_copy(update={"kind": "fact", "conflict_group": None,
                                                         "status": "SUPPORTED"})

    def _doc_info(self, claim: AnswerClaim) -> tuple[str, str, list[str]] | None:
        docs = {self.catalog.get(e).document_id for e in claim.evidence_ids if self.catalog.get(e) is not None}
        if len(docs) != 1:
            return None
        d = self.catalog.documents.get(next(iter(docs)))
        if d is None:
            return None
        m = d.metadata or {}
        sup = [x.strip() for x in str(m.get("supersedes", "") or "").split(",") if x.strip()]
        return d.document_id, str(m.get("status", "current") or "current").lower(), sup

    def _scope(self, claim: AnswerClaim) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for eid in claim.evidence_ids:
            ch = self.catalog.get(eid)
            doc = self.catalog.documents.get(ch.document_id) if ch is not None else None
            for f in SCOPE_FIELDS:
                v = str((doc.metadata if doc is not None else {}).get(f, "") or "").lower()
                if v and v != "all":
                    out.setdefault(f, set()).add(v)
        return out

    def _different_scope(self, a: AnswerClaim, b: AnswerClaim) -> bool:
        sa, sb = self._scope(a), self._scope(b)
        return any(f in sb and not (sa[f] & sb[f]) for f in sa)

    @staticmethod
    def _rank(c: AnswerClaim, s) -> int:
        pos = {f.plan_claim_id: k for k, f in enumerate(s.facts)}
        return min((pos.get(f, 99) for f in c.facts), default=50)

    # ------------------------------------------------------------------ diff + events
    def _diff(self, prev, sections, claims, regenerated, reused) -> AnswerClaimDiff:
        new_ids = [c for s in sections for c in s.claim_ids if claims[c].kind != "uncertainty"]
        old_ids = [c for s in prev.sections for c in s.claim_ids
                   if prev.claim(c) and prev.claim(c).kind != "uncertainty"] if prev else []
        unchanged = [c for c in new_ids if c in old_ids]
        added = [c for c in new_ids if c not in old_ids]
        removed = [c for c in old_ids if c not in new_ids]
        modified = []
        for r in list(removed):
            rk = set(prev.claim(r).fact_keys)
            m = next((a for a in added if rk and rk & set(claims[a].fact_keys)), None)
            if m is not None:
                modified.append((r, m))
                removed.remove(r)
                added.remove(m)
        return AnswerClaimDiff(added=added, modified=modified, removed=removed, unchanged=unchanged,
                               sections_regenerated=list(regenerated), sections_reused=list(reused))

    def _emit_answer(self, ga: GroundedAnswer, cmap, crep, cov, uid) -> None:
        emit = self.emit
        base = {"answer_id": ga.answer_id, "version": ga.version, "status": ga.status}
        for c in cmap.citations:
            emit(E.CITATION_CREATED, "citation_mapper", {**base, "citation_id": c.citation_id, "claim_id": c.claim_id,
                                                         "evidence_id": c.evidence_id, "chunk_id": c.chunk_id,
                                                         "source_id": c.source_id, "key": c.display_metadata.get("key"),
                                                         "span": [c.location.char_start, c.location.char_end],
                                                         "strength": c.strength, "status": c.status}, uid)
        emit(E.CITATION_VALIDATED, "citation_validator", {**base, **crep.model_dump(mode="json")}, uid)
        emit(E.ANSWER_VALIDATED, "answer_validator", {
            **base, "partial": ga.partial, "blocked_reasons": ga.blocked_reasons, "metrics": ga.metrics,
            "coverage": {"covered": cov.covered, "uncertain_only": cov.uncertain_only, "failures": cov.failures,
                         "intent_coverage": cov.intent_coverage},
            "rejected": [r.model_dump(mode="json") for r in ga.rejected], "verifier": ga.verifier}, uid)
        if ga.parent_answer_id:
            emit(E.ANSWER_REVISED, "answer_state", {**base, "parent_answer_id": ga.parent_answer_id,
                                                    "diff": ga.diff.model_dump(mode="json")}, uid)
        emit(E.ANSWER_STARTED, "answer_stream", {**base, "frame_id": ga.frame_id, "sections": len(ga.sections)}, uid)
        by_id = {c.claim_id: c for c in ga.claims}
        cits = {c.citation_id: c for c in cmap.citations if c.status == "valid"}
        for s in ga.sections:
            emit(E.ANSWER_SECTION_STARTED, "answer_stream", {**base, "section_id": s.section_id,
                                                             "title": s.title, "section_status": s.status}, uid,
                 intent_id=s.intent_id)
            for cid in s.claim_ids:
                c = by_id[cid]
                emit(E.ANSWER_CLAIM_READY, "answer_stream", {**base, "section_id": s.section_id, "claim_id": cid,
                                                             "text": c.text, "kind": c.kind, "claim_status": c.status,
                                                             "origin": c.origin}, uid, intent_id=s.intent_id)
                for cit in c.citation_ids:
                    if cit in cits:
                        emit(E.ANSWER_CITATION_READY, "answer_stream", {
                            **base, "section_id": s.section_id, "claim_id": cid, "citation_id": cit,
                            "key": cits[cit].display_metadata.get("key")}, uid, intent_id=s.intent_id)
            emit(E.ANSWER_SECTION_COMPLETED, "answer_stream", {**base, "section_id": s.section_id,
                                                               "claims": len(s.claim_ids)}, uid, intent_id=s.intent_id)
        emit(E.ANSWER_COMPLETED, "answer_stream", {**base, "text": ga.text, "partial": ga.partial,
                                                   "diff": ga.diff.model_dump(mode="json"),
                                                   "wall": {"timings_ms": ga.timings_ms}}, uid)
        if ga.status == "VALIDATED_FINAL":
            emit(E.ANSWER_FINALIZED, "answer_state", {**base, "citations": cmap.keys()}, uid)
