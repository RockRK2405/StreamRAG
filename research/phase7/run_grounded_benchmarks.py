"""Phase 7 measurements on the DEV grounded-answer suite (fixture domain; NOT REPORTABLE; NOT held-out).

Every number is measured by running the implementation with the local LLM (Ollama ``generation.model``) and the
entailment verifier; nothing is estimated or copied.

Outputs (research/phase7/results/):
  arms.json            per case / turn / arm: answer, raw and final claims, gold checks, verifier-judged metrics, cost
  summary.json         aggregates per arm (ablation, brief §53) and per category (benchmark, brief §51-52)
  hallucination.json   temptation cases grouped by tag (brief §54)
  revision.json        multi-turn cases: incremental (F) vs full regeneration every turn (F_restart) - stability,
                       LLM calls, latency, revision accuracy (brief §41, §48)
  latency.json         per-stage latency of the full system, generation vs validation per arm (brief §55)
  verifier_eval.json   claim-verifier accuracy on perturbations with labels by construction (nli vs rules)
  claims_for_labeling.jsonl  raw LLM claims with their evidence, for blind hand-labelling (labels/)

Usage: .venv/bin/python research/phase7/run_grounded_benchmarks.py --index-root /tmp/idx7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

from streamrag.answer_state.resources import GroundingResources  # noqa: E402
from streamrag.bench.grounded import (  # noqa: E402
    ABSTAIN,
    BANNER,
    abstains,
    conflict_presented,
    forbidden_hits,
    gold_recall,
    run_free_arm,
)
from streamrag.bench.multi_intent import pct, write_json  # noqa: E402
from streamrag.citations.mapper import ChunkCatalog, CitationMapper  # noqa: E402
from streamrag.claims.graph import sentences  # noqa: E402
from streamrag.claims.nli import NliModel  # noqa: E402
from streamrag.claims.textcheck import instruction_like  # noqa: E402
from streamrag.config import load_config  # noqa: E402
from streamrag.generation.generator import GroundedAnswerGenerator  # noqa: E402
from streamrag.generation.llm import OllamaBackend  # noqa: E402
from streamrag.retrieval import build_index  # noqa: E402
from streamrag.session import AdaptivePipeline, FullRestartPipeline  # noqa: E402
from streamrag.streaming.factory import build_stack  # noqa: E402

CASES = REPO / "eval" / "dev_grounded"
CORPORA = {"fixture": "tests/fixtures/corpus", "grounding": "tests/fixtures/corpus_grounding",
           "conflict": "tests/fixtures/corpus_conflict", "injection": "tests/fixtures/corpus_injection"}
GAP_MS = 3000.0


class CachingBackend:
    """Wraps the LLM so that arms D and E reuse arm C's exact output (same prompt): the ablation then isolates
    validation and repair. The first call's measured timings are returned for every reuse."""

    def __init__(self, inner) -> None:
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.cache: dict[str, object] = {}

    def complete(self, messages, schema=None):
        from streamrag.generation.llm import request_sha1
        key = request_sha1(self.model, messages, schema)
        if key not in self.cache:
            self.cache[key] = self.inner.complete(messages, schema)
        return self.cache[key]


def stack_for(name: str, index_root: Path, nli, backend):
    cfg = load_config(REPO / "configs" / "default.yaml",
                      {"paths.corpus": str(REPO / CORPORA[name]), "paths.index_root": str(index_root / name),
                       "telemetry.log_level": "ERROR", "multi_intent.enabled": True, "session.enabled": True,
                       "generation.enabled": True}, base_dir=REPO)
    st = build_stack(cfg, build_index(cfg).path)
    an = st.bundle.analyzer
    st._grounding = GroundingResources(ChunkCatalog.from_bundle(st.bundle), nli, backend,
                                       lambda t: list(dict.fromkeys(an.tokens(t))))
    return st


def facts_of(g):
    return [c.text for c in g.claims if c.kind in ("fact", "conflict")]


def score_system(g, gold, nli) -> dict:
    facts = facts_of(g)
    found, total, missing = gold_recall(facts, gold["required_facts"], nli)
    unc = [c.text for c in g.claims if c.kind == "uncertainty"]
    return {"answer_id": g.answer_id, "status": g.status, "partial": g.partial, "text": g.text,
            "fact_claims": len(facts), "required_found": found, "required_total": total, "missing": missing,
            "forbidden_hits": forbidden_hits(facts, gold["forbidden"]),
            "conflict_presented": conflict_presented(g.text, gold["conflict_values"]),
            "uncertainty_expressed": bool(unc), "intent_coverage": g.coverage.intent_coverage,
            "sections": len(g.sections), "metrics": g.metrics, "llm_calls": g.llm_calls,
            "prompt_tokens": g.prompt_tokens, "output_tokens": g.output_tokens, "ttft_raw_ms": g.ttft_raw_ms,
            "timings_ms": g.timings_ms, "rejected": [r.model_dump(mode="json") for r in g.rejected],
            "repairs": [r.action for r in g.repairs], "backend": g.backend, "fallback": g.fallback,
            "revision_attempts": g.revision_attempts, "validation_retrievals": g.validation_retrievals,
            "citations_valid": g.citation_report.valid, "citations_checked": g.citation_report.checked,
            "claims": [{"claim_id": c.claim_id, "text": c.text, "kind": c.kind, "status": c.status, "origin": c.origin,
                        "evidence_ids": c.evidence_ids} for c in g.claims],
            "diff": g.diff.model_dump(mode="json")}


def score_free(r, gold, nli) -> dict:
    facts = [f["text"] for f in r["final"]]
    found, total, missing = gold_recall(facts, gold["required_facts"], nli)
    return {**{k: v for k, v in r.items() if k not in ("final", "raw")}, "raw": r["raw"],
            "final": r["final"], "required_found": found, "required_total": total, "missing": missing,
            "forbidden_hits": forbidden_hits(facts, gold["forbidden"]),
            "conflict_presented": conflict_presented(r["text"], gold["conflict_values"]),
            "uncertainty_expressed": r["abstained"]}


def question_of(p6) -> str:
    parts = []
    for s in p6.sections:
        t = s.title.rstrip("?.")
        parts.append(t + (f" ({'; '.join(s.constraints)})" if s.constraints else ""))
    return "? ".join(parts) + "?"


def run_cases(stacks, cases, nli, llm, gen_free, verifier, mappers) -> list[dict]:
    rows = []
    for case in cases:
        st = stacks[case["corpus"]]
        sysF = AdaptivePipeline(st, grounding=st.grounding)
        sysX = AdaptivePipeline(st, grounding=st.grounding.with_backend(None))
        for n, turn in enumerate(case["turns"], start=1):
            uid, text, gold = turn["utterance_id"], turn["utterance_text"], turn["gold"]
            rF = sysF.process(uid, text, n * GAP_MS)
            rX = sysX.process(uid, text, n * GAP_MS)
            gF = sysF.grounding.current.get(sysF.engine.frames.active.frame_id) if sysF.engine.frames.active else None
            gX = sysX.grounding.current.get(sysX.engine.frames.active.frame_id) if sysX.engine.frames.active else None
            row = {"case_id": case["case_id"], "category": case["category"], "turn": n, "text": text, "gold": gold,
                   "changes": [c.change_type for c in rF.changes], "arms": {}}
            if gF is not None:
                row["arms"]["F"] = score_system(gF, gold, nli)
                row["arms"]["F"]["new_version"] = rF.grounded is not None
            if gX is not None:
                row["arms"]["X"] = score_system(gX, gold, nli)
            p6 = sysF.engine.answers.get_current_answer_state()
            if p6 is not None and gF is not None:
                eng = sysF.engine
                pool = {}
                for s in p6.sections:
                    for a in eng.store.usable(s.intent_id):
                        pool.setdefault(a.evidence_id, eng.store.text(a.evidence_id))
                pool = {e: t for e, t in pool.items()}
                cites = {e: eng.store.records[e].citation for e in pool}
                row["pool"] = {e: {"citation": cites[e], "text": t} for e, t in pool.items()}
                q = question_of(p6)

                def retrieve(query, _st=st):
                    es = _st.service.retrieve(query, sysF.options.model_copy(update={"top_k": 3}))
                    return {e.evidence_id: e.text for e in es.items}
                for arm in ("A", "B", "C", "D", "E"):
                    r = run_free_arm(arm, q, pool, cites, gen_free, verifier, mappers[case["corpus"]],
                                     retrieve if arm == "E" else None)
                    row["arms"][arm] = score_free(r, gold, nli)
            rows.append(row)
            print(f"  {case['case_id']} u{n}: " + " ".join(
                f"{a}:{row['arms'][a]['required_found']}/{row['arms'][a]['required_total']}"
                f"{'!' if row['arms'][a]['forbidden_hits'] else ''}" for a in row["arms"]), flush=True)
        sysF.close()
        sysX.close()
    return rows


def _rate(a, b):
    return round(a / b, 4) if b else None


def summarize(rows, arms=("A", "B", "C", "D", "E", "F", "X"), key=None) -> dict:
    out = {}
    for arm in arms:
        rs = [r["arms"][arm] for r in rows if arm in r["arms"] and (key is None or key(r))]
        if not rs:
            continue
        sysarm = arm in ("F", "X")
        raw_claims = sum(r["metrics"]["raw_claims"] or 0 for r in rs) if sysarm else sum(r["raw_claims"] for r in rs)
        raw_sup = (sum((r["metrics"]["raw_support_rate"] or 0) * (r["metrics"]["raw_claims"] or 0) for r in rs)
                   if sysarm else sum(r["raw_supported"] for r in rs))
        raw_uns = (sum((r["metrics"]["raw_unsupported_rate"] or 0) * (r["metrics"]["raw_claims"] or 0) for r in rs)
                   if sysarm else sum(r["raw_unsupported"] for r in rs))
        final = sum(r["fact_claims"] for r in rs) if sysarm else sum(r["final_claims"] for r in rs)
        final_unsup = (sum(round((r["metrics"]["final_unsupported_rate"] or 0) * r["fact_claims"]) for r in rs)
                       if sysarm else sum(r["final_unsupported"] for r in rs))
        cit_e = sum(r["citations_checked"] for r in rs) if sysarm else sum(r["citations_emitted"] for r in rs)
        cit_v = sum(r["citations_valid"] for r in rs) if sysarm else sum(r["citations_valid"] for r in rs)
        cited = (sum(round((r["metrics"]["citation_coverage"] or 0) * r["fact_claims"]) for r in rs) if sysarm
                 else sum(r["final_cited"] for r in rs))
        req_f, req_t = sum(r["required_found"] for r in rs), sum(r["required_total"] for r in rs)
        unc_rows = [r for row, r in zip([x for x in rows if arm in x["arms"] and (key is None or key(x))], rs)
                    if row["gold"]["expect_uncertainty"]]
        conf_rows = [r for r in rs if r["conflict_presented"] is not None]
        out[arm] = {
            "turns": len(rs),
            "gold_required_fact_recall": _rate(req_f, req_t),
            "forbidden_assertions": sum(len(r["forbidden_hits"]) for r in rs),
            "turns_with_forbidden_assertion": sum(1 for r in rs if r["forbidden_hits"]),
            "uncertainty_expressed_when_expected": _rate(sum(1 for r in unc_rows if r["uncertainty_expressed"]),
                                                         len(unc_rows)),
            "conflict_presented_rate": _rate(sum(1 for r in conf_rows if r["conflict_presented"]), len(conf_rows)),
            "raw_claims": raw_claims, "raw_claim_support_rate_verifier": _rate(raw_sup, raw_claims),
            "raw_claim_unsupported_rate_verifier": _rate(raw_uns, raw_claims),
            "final_claims": final, "final_unsupported_rate_verifier": _rate(final_unsup, final),
            "citation_precision_verifier": _rate(cit_v, cit_e) if arm not in ("A", "B") else None,
            "citation_coverage": _rate(cited, final) if arm not in ("A", "B") else None,
            "intent_coverage": (round(statistics.fmean(r["intent_coverage"] for r in rs if r["intent_coverage"]
                                                       is not None), 4) if sysarm else None),
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "latency_ms": pct([(r["timings_ms"]["total"] if sysarm else r["total_ms"]) for r in rs]),
            "generation_ms": pct([(r["timings_ms"]["generation"] if sysarm else r["generation_ms"]) for r in rs]),
            "validation_ms": pct([((r["timings_ms"]["total"] - r["timings_ms"]["generation"]) if sysarm
                                   else r["validation_ms"]) for r in rs]),
        }
    return out


def revision(stacks, cases, nli) -> dict:
    multi = [c for c in cases if len(c["turns"]) > 1]
    rows = []
    for case in multi:
        st = stacks[case["corpus"]]
        for mode in ("incremental", "full_restart"):
            p = (AdaptivePipeline(st, grounding=st.grounding) if mode == "incremental"
                 else FullRestartPipeline(st, grounding=st.grounding))
            prev = None
            for n, turn in enumerate(case["turns"], start=1):
                t0 = time.perf_counter()
                r = p.process(turn["utterance_id"], turn["utterance_text"], n * GAP_MS)
                wall = (time.perf_counter() - t0) * 1000.0
                g = r.grounded
                eng = p.last if mode == "full_restart" else p
                cur = g or (eng.grounding.current.get(eng.engine.frames.active.frame_id)
                            if eng.engine.frames.active else None)
                facts = facts_of(cur) if cur else []
                found, total, _ = gold_recall(facts, turn["gold"]["required_facts"], nli)
                ids_now = {c.claim_id for c in cur.claims} if cur else set()
                ids_prev = {c.claim_id for c in prev.claims} if prev else set()
                rows.append({"case_id": case["case_id"], "mode": mode, "turn": n, "wall_ms": round(wall, 2),
                             "llm_calls": g.llm_calls if g else 0, "new_version": g is not None,
                             "claims": len(ids_now), "kept_from_previous": len(ids_now & ids_prev),
                             "previous_claims": len(ids_prev), "required_found": found, "required_total": total,
                             "forbidden_hits": forbidden_hits(facts, turn["gold"]["forbidden"]),
                             "diff": g.diff.model_dump(mode="json") if g else None})
                prev = cur or prev
            p.close()
    out = {}
    for mode in ("incremental", "full_restart"):
        rs = [r for r in rows if r["mode"] == mode and r["turn"] > 1]
        out[mode] = {"later_turns": len(rs), "llm_calls": sum(r["llm_calls"] for r in rs),
                     "wall_ms": pct([r["wall_ms"] for r in rs]),
                     "claims_kept_from_previous_version": _rate(sum(r["kept_from_previous"] for r in rs),
                                                                sum(r["previous_claims"] for r in rs)),
                     "revision_accuracy_required_recall": _rate(sum(r["required_found"] for r in rs),
                                                                sum(r["required_total"] for r in rs)),
                     "forbidden_after_revision": sum(len(r["forbidden_hits"]) for r in rs)}
    return {"summary": out, "rows": rows}


# ------------------------------------------------------------------------------------------------ verifier eval
_SWAP = {"must": "may", "may": "must", "not": "", "cannot": "can", "only": "always", "never": "always"}


def perturbations(sentence: str, other: str, rng: random.Random) -> list[tuple[str, str, bool]]:
    """(kind, text, supported?) - labels by construction (docs/answer/04 caveats)."""
    out = [("original", sentence, True)]
    nums = re.findall(r"\b\d+\b", sentence)
    if nums:
        n = nums[0]
        out.append(("number_changed", re.sub(rf"\b{n}\b", str(int(n) + 7), sentence, count=1), False))
    words = sentence.split()
    for k, w in enumerate(words):
        lw = w.lower().strip(".,;")
        if lw in _SWAP and k > 0:
            rep = _SWAP[lw]
            new = " ".join(words[:k] + ([rep] if rep else []) + words[k + 1:])
            out.append(("modality_or_negation_changed", new, False))
            break
    else:
        verb = next((k for k, w in enumerate(words) if w.lower() in ("is", "are", "was", "were")), None)
        if verb is not None:
            out.append(("negation_inserted", " ".join(words[:verb + 1] + ["not"] + words[verb + 1:]), False))
    tail = other.rstrip(".")
    out.append(("unsupported_conjunct", sentence.rstrip(".") + " and " + tail[0].lower() + tail[1:] + ".", False))
    return out


def verifier_eval(stacks, nli) -> dict:
    from streamrag.claims.aligner import ClaimEvidenceAligner
    from streamrag.claims.decomposer import ClaimDecomposer, ClaimLexicon
    from streamrag.claims.verifier import ClaimVerifier
    rng = random.Random(7)
    items = []
    for name in ("fixture", "grounding"):
        st = stacks[name]
        cat = st.grounding.catalog
        chunks = [c for c in st.bundle.chunks]
        sents = [(c, c.text[s:e]) for c in chunks for s, e in sentences(c.text)
                 if not cat.is_boilerplate(c.text[s:e]) and not instruction_like(c.text[s:e])
                 and len(c.text[s:e].split()) >= 5]
        for c, s in sents:
            pool = {c.chunk_id: c.text}
            for d in rng.sample([x for x in chunks if x.document_id != c.document_id], k=min(2, len(chunks) - 1)):
                pool[d.chunk_id] = d.text
            pool_docs = {x.document_id for x in chunks if x.chunk_id in pool}
            # the appended clause must come from a document outside the pool, else it may really be supported
            other = rng.choice([x for ch, x in sents if ch.document_id not in pool_docs] or [x for _, x in sents
                                                                                             if x != s])
            for kind, text, label in perturbations(s, other, rng):
                items.append((name, kind, text, label, pool))
    out = {}
    lex = ClaimLexicon.load(REPO / "configs" / "claim_lexicon.yaml")
    for mode in ("nli", "rules"):
        terms = stacks["fixture"].grounding.terms_fn
        v = ClaimVerifier(ClaimEvidenceAligner(terms, nli if mode == "nli" else None, mode), ClaimDecomposer(lex))
        res = []
        t0 = time.perf_counter()
        for k, (name, kind, text, label, pool) in enumerate(items):
            vv = v.verify(f"p{k}", text, [], pool)
            res.append((kind, label, vv.status == "SUPPORTED"))
        ms = (time.perf_counter() - t0) * 1000.0
        tp = sum(1 for _, lab, pred in res if lab and pred)
        fp = sum(1 for _, lab, pred in res if not lab and pred)
        fn = sum(1 for _, lab, pred in res if lab and not pred)
        tn = sum(1 for _, lab, pred in res if not lab and not pred)
        by = {}
        for kind in sorted({k for k, _, _ in res}):
            rs = [(lab, pred) for k, lab, pred in res if k == kind]
            by[kind] = {"n": len(rs), "accepted_as_supported": sum(p for _, p in rs)}
        out[mode] = {"items": len(res), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "support_precision": _rate(tp, tp + fp), "support_recall": _rate(tp, tp + fn),
                     "unsupported_detection_recall": _rate(tn, tn + fp), "accuracy": _rate(tp + tn, len(res)),
                     "by_kind": by, "ms_per_claim": round(ms / max(1, len(res)), 3)}
    return {"note": "labels by construction: original sentence = supported by its own chunk; a changed number, "
                    "modality / negation change or an appended unrelated clause = not supported. Caveat: a few "
                    "modality swaps can remain true (e.g. 'may' for 'must'); counted as not supported.", **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t_start = time.time()
    cfg0 = load_config(REPO / "configs" / "default.yaml", {}, base_dir=REPO)
    g = cfg0.generation
    if not OllamaBackend.reachable(g.ollama_url, g.model):
        raise SystemExit(f"LLM {g.model} not reachable at {g.ollama_url}: start Ollama first")
    nli = NliModel.load(cfg0.paths.models_dir, g.nli_model)
    llm = CachingBackend(OllamaBackend(g.ollama_url, g.model, g.temperature, g.seed, g.num_ctx, g.max_output_tokens,
                                       g.timeout_s))
    stacks = {k: stack_for(k, a.index_root, nli, OllamaBackend(g.ollama_url, g.model, g.temperature, g.seed,
                                                               g.num_ctx, g.max_output_tokens, g.timeout_s))
              for k in CORPORA}
    cases = [json.loads(p.read_text()) for p in sorted(CASES.glob("*.json"))]
    if a.only:
        cases = [c for c in cases if c["case_id"] in a.only.split(",")]
    from streamrag.claims.aligner import ClaimEvidenceAligner
    from streamrag.claims.decomposer import ClaimDecomposer, ClaimLexicon
    from streamrag.claims.verifier import ClaimVerifier
    verifier = ClaimVerifier(ClaimEvidenceAligner(stacks["fixture"].grounding.terms_fn, nli, "nli"),
                             ClaimDecomposer(ClaimLexicon.load(REPO / "configs" / "claim_lexicon.yaml")))
    mappers = {k: CitationMapper(s.grounding.catalog) for k, s in stacks.items()}
    gen_free = GroundedAnswerGenerator(llm, g.max_structured_retries)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"REPORTABLE": False, "banner": BANNER, "cases": len(cases), "llm": g.model, "llm_backend": "ollama",
            "temperature": g.temperature, "seed": g.seed, "verifier": f"nli:{g.nli_model}+rules",
            "index_content_hash": {k: s.index_hash for k, s in stacks.items()},
            "embedder": stacks["fixture"].bundle.dense.info.name if stacks["fixture"].bundle.dense else None,
            "arms": {"A": "plain LLM (no evidence)", "B": "RAG (evidence, no citation requirement)",
                     "C": "RAG + citations (labels trusted)", "D": "C + claim validation (unsupported removed)",
                     "E": "D + repair (supported atoms kept, retrieval fallback for material claims)",
                     "F": "full Phase 7: claim planning + validation + repair + incremental refinement",
                     "X": "extractive generator (no LLM), same validation"},
            "note": "D and E reuse C's exact LLM output (cached by request hash); their generation latency is C's."}
    print("cases ...", flush=True)
    rows = run_cases(stacks, cases, nli, llm, gen_free, verifier, mappers)
    write_json(OUT / "arms.json", {"meta": meta, "rows": rows})
    cats = sorted({r["category"] for r in rows})
    summary = {"meta": meta, "by_arm": summarize(rows),
               "by_category": {c: summarize(rows, key=lambda r, c=c: r["category"] == c) for c in cats}}
    write_json(OUT / "summary.json", summary)
    tags = sorted({t for r in rows for t in r["gold"]["tempt"]})
    write_json(OUT / "hallucination.json", {"meta": meta, "by_tag": {
        t: summarize(rows, key=lambda r, t=t: t in r["gold"]["tempt"]) for t in tags}})
    print("revision ...", flush=True)
    write_json(OUT / "revision.json", {"meta": meta, **revision(stacks, cases, nli)})
    lat = {"F_stages_ms": {s: pct([r["arms"]["F"]["timings_ms"].get(s, 0.0) for r in rows if "F" in r["arms"]
                                   and r["arms"]["F"]["new_version"]])
                           for s in ("claim_planning", "generation", "claim_extraction", "claim_verification",
                                     "repair", "retrieval", "citation_mapping", "validation", "total")},
           "F_ttft_raw_ms": pct([r["arms"]["F"]["ttft_raw_ms"] for r in rows if "F" in r["arms"]
                                 and r["arms"]["F"]["new_version"] and r["arms"]["F"]["ttft_raw_ms"]]),
           "F_tokens": {"prompt": sum(r["arms"]["F"]["prompt_tokens"] for r in rows if "F" in r["arms"]
                                      and r["arms"]["F"]["new_version"]),
                        "output": sum(r["arms"]["F"]["output_tokens"] for r in rows if "F" in r["arms"]
                                      and r["arms"]["F"]["new_version"])}}
    write_json(OUT / "latency.json", {"meta": meta, **lat})
    print("verifier eval ...", flush=True)
    write_json(OUT / "verifier_eval.json", {"meta": meta, **verifier_eval(stacks, nli)})
    # blind labelling sheet: claim text + the exact evidence pool it was verified against, no verifier verdicts
    # (the verdicts stay in arms.json; research/phase7/label_agreement.py joins them after labelling)
    with (OUT / "claims_for_labeling.jsonl").open("w") as f:
        k = 0
        for r in rows:
            items = [(arm, c["text"]) for arm in ("B", "C") for c in r["arms"].get(arm, {}).get("raw", [])
                     if not c["abstention"]]
            fa = r["arms"].get("F", {})
            if fa.get("new_version"):
                items += [("F_released", c["text"]) for c in fa["claims"] if c["kind"] == "fact"]
                items += [("F_rejected", x["text"]) for x in fa["rejected"]]
            uniq = {}
            for arm, text in items:                      # one item per distinct claim text of the turn
                uniq.setdefault(text, []).append(arm)
            for text, arms in uniq.items():
                k += 1
                f.write(json.dumps({"item": f"L{k:03d}", "case_id": r["case_id"], "turn": r["turn"], "sources": arms,
                                    "claim": text, "pool": [v["text"] for v in r.get("pool", {}).values()]}) + "\n")
    print(f"done in {time.time() - t_start:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
