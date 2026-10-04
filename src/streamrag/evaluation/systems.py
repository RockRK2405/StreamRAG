"""System variants under evaluation (Phase 10; docs/evaluation/experimental_setup.md).

Families
  free       batch, single turn, no session: utterance -> fixed retrieval (mode, k, rerank) -> LLM answer with its own
             evidence labels (Phase 7 free-form generator), NO verification / repair / citation validation.
             Baselines A (naive: dense), B (hybrid RRF), C (hybrid + cross-encoder).
  pipeline   batch session pipeline (Phase 5-7 + optional Phase 9): interpretation, delta planning, evidence
             lifecycle, retrieval (fixed or adaptive), grounded answer. Answer modes: grounded (Phase 7 engine),
             unverified (free-form generation over the pipeline's evidence, nothing verified),
             no_citation_validation (claims verified and unsupported ones dropped, but the LLM's own citation labels
             kept). ``memoryless``: every turn in a fresh session; ``full_restart``: Phase 6 full-restart baseline.
  runtime    the Phase 8 streaming runtime (realtime, real worker threads): chunks streamed at ``interval_ms``,
             early retrieval, drafts, cancellation; Baseline D (fixed retrieval) and the full system (adaptive).
             ``batch_mode``: utterance-end retrieval only, no drafts (streaming responsiveness experiment).

Every adapter returns one ``TurnRecord`` per evaluated turn. Numbers are measured (wall clock / counters); nothing
is estimated.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from streamrag.evaluation.dataset import EvalSample


@dataclass
class TurnRecord:
    sample_id: str
    system: str
    status: str = "ok"                       # ok | failed
    error: str | None = None
    query_used: str = ""
    keys: list[str] = field(default_factory=list)            # ranked evidence citations handed to the answer stage
    evidence: list[dict] = field(default_factory=list)       # [{chunk_id, citation, document_id, text}]
    answer: str = ""
    claims: list[tuple[str, list[str]]] = field(default_factory=list)
    ops: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    adaptive: dict = field(default_factory=dict)
    stream: dict = field(default_factory=dict)
    session_stream: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    trace: str | None = None
    events_digest: dict = field(default_factory=dict)


def chunks_for(text: str, words: int = 3) -> list[str]:
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words)] or [text]


# ------------------------------------------------------------------------------------------------ stacks
class Stacks:
    """One Phase 5-7 stack per corpus (index, embedder, reranker, NLI), shared by every variant (cached)."""

    def __init__(self, repo: Path, corpora: dict[str, str], index_root: Path, llm=None, rerank: bool = True) -> None:
        self.repo, self.corpora, self.index_root, self.llm, self.rerank = repo, corpora, index_root, llm, rerank
        self._st: dict = {}
        self._nli = None

    def nli(self):
        if self._nli is None:
            from streamrag.claims.nli import NliModel
            self._nli = NliModel.load(self.repo / "models", "nli-deberta-v3-xsmall")
        return self._nli

    def get(self, corpus: str, generation: str = "llm"):
        """``generation``: "llm" (the local model; requires ``llm``) or "extractive" (deterministic, no LLM)."""
        key = (corpus, generation)
        if key in self._st:
            return self._st[key]
        use_llm = generation == "llm"
        if use_llm and self.llm is None:
            raise RuntimeError("variant needs the LLM but no LLM backend was given (start `ollama serve`)")
        from streamrag.answer_state.resources import GroundingResources
        from streamrag.citations.mapper import ChunkCatalog
        from streamrag.config import load_config
        from streamrag.retrieval import build_index
        from streamrag.streaming.factory import build_stack
        ov = {"paths.corpus": str(self.repo / self.corpora[corpus]),
              "paths.index_root": str(self.index_root / corpus), "telemetry.log_level": "ERROR",
              "multi_intent.enabled": True, "session.enabled": True, "generation.enabled": True,
              "generation.backend": "ollama" if use_llm else "extractive",
              "streaming.rerank": self.rerank, "adaptive_retrieval.reference_date": "2026-10-03"}
        cfg = load_config(self.repo / "configs" / "default.yaml", ov, base_dir=self.repo)
        st = build_stack(cfg, build_index(cfg).path)
        an = st.bundle.analyzer
        st._grounding = GroundingResources(ChunkCatalog.from_bundle(st.bundle), self.nli(),
                                           self.llm if use_llm else None, lambda t: list(dict.fromkeys(an.tokens(t))))
        self._st[key] = st
        return st


def with_cfg(st, overrides: dict):
    cfg = st.cfg
    for key, val in overrides.items():
        parts = key.split(".")
        objs = [cfg]
        for p in parts[:-1]:
            objs.append(getattr(objs[-1], p))
        new = objs[-1].model_copy(update={parts[-1]: val})
        for obj, p in zip(reversed(objs[:-1]), reversed(parts[:-1])):
            new = obj.model_copy(update={p: new})
        cfg = new
    s2 = st.with_config(cfg)
    s2._intent_stack = st.intent_stack
    s2._grounding = st.grounding
    return s2


def _ev(st, chunk_ids: list[str]) -> list[dict]:
    by = {c.chunk_id: c for c in st.bundle.chunks}
    out = []
    for i in chunk_ids:
        c = by.get(i)
        if c is not None:
            out.append({"chunk_id": i, "citation": c.citation, "document_id": c.document_id, "text": c.text})
    return out


def _keys(ev: list[dict]) -> list[str]:
    return list(dict.fromkeys(e["citation"] for e in ev))


# ------------------------------------------------------------------------------------------------ free (A, B, C)
def run_free(spec: dict, st, session: list[EvalSample], llm) -> list[TurnRecord]:
    from streamrag.generation.generator import GroundedAnswerGenerator
    from streamrag.models.retrieval import RetrievalOptions
    gen = GroundedAnswerGenerator(llm, 1)
    out = []
    for s in session:
        r = TurnRecord(s.sample_id, spec["id"], query_used=s.query)
        try:
            t0 = time.perf_counter()
            es = st.service.retrieve(s.query, RetrievalOptions(mode=spec.get("mode", "hybrid"), top_k=spec.get("k", 5),
                                                               rerank=spec.get("rerank", False)))
            t_ret = (time.perf_counter() - t0) * 1000.0
            ids = [e.chunk_id for e in es.items]
            r.evidence = _ev(st, ids)
            r.keys = _keys(r.evidence)
            labels = {f"E{k + 1}": e.chunk_id for k, e in enumerate(es.items)}
            lab = {v: k for k, v in labels.items()}
            cand = gen.generate_free(s.query, [(lab[e.chunk_id], e.citation, e.text) for e in es.items],
                                     "evidence_labels")
            t_all = (time.perf_counter() - t0) * 1000.0
            from streamrag.evaluation.instrument import AnswerInstrument
            r.claims = AnswerInstrument.split_free(cand.sentences, labels)
            r.answer = " ".join(t for t, _ in r.claims)
            mode = spec.get("mode", "hybrid")
            r.ops = {"retrieval_calls": 1, "embeddings": int(mode in ("dense", "hybrid")),
                     "lexical_searches": int(mode in ("bm25", "hybrid")),
                     "reranker_calls": int(bool(es.trace.rerank_applied)), "chunks_retrieved": len(es.items),
                     "k_values": [spec.get("k", 5)], "iterations": 1, "expansions": 0,
                     "documents_retrieved": len({e.document_id for e in es.items}), "cache_hit": 0,
                     "evidence_reused": 0, "llm_calls": cand.llm_calls, "prompt_tokens": cand.prompt_tokens,
                     "output_tokens": cand.output_tokens}
            r.latency = {"ttfe_after_end": round(t_ret, 3), "ttfa_after_end": round(t_all, 3),
                         "ttva_after_end": round(t_all, 3), "total_after_end": round(t_all, 3),
                         "generation_ms": round(cand.generation_ms or 0.0, 3), "llm_ttft_ms": cand.ttft_ms,
                         "verification_ms": 0.0}
            if cand.fallback:
                r.error = f"generation_fallback: {cand.fallback}"
        except Exception as exc:     # noqa: BLE001 - recorded as a failed sample, never dropped
            r.status, r.error = "failed", f"{exc.__class__.__name__}: {exc}"
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ pipeline
def _turn_evidence(p, res, st, uid: str) -> list[str]:
    needs = []
    for ch in res.changes:
        for i in list(ch.new_intents) + list(ch.affected_intents):
            it = p.tracker.intents.get(i)
            if it is not None and it.status == "ACTIVE" and i not in needs:
                needs.append(i)
    lists = []
    for i in needs:
        rec = p.ledger.active_for_intent(i)
        if rec is not None and rec.evidence_ids:
            lists.append(list(rec.evidence_ids))
    ids: list[str] = []
    for n in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if n < len(x) and x[n] not in ids:
                ids.append(x[n])
    return ids


def run_pipeline(spec: dict, st, session: list[EvalSample], llm) -> list[TurnRecord]:
    from streamrag.session.pipeline import AdaptivePipeline, FullRestartPipeline
    s2 = with_cfg(st, {"streaming.rerank": False, **spec.get("overrides", {})})
    mode = spec.get("answer_mode", "grounded")
    out = []
    pipe = None
    for n, s in enumerate(session, start=1):
        r = TurnRecord(s.sample_id, spec["id"], query_used=s.query)
        try:
            if pipe is None or spec.get("memoryless"):
                if pipe is not None:
                    pipe.close()
                pipe = FullRestartPipeline(s2) if spec.get("full_restart_pipeline") else AdaptivePipeline(s2)
            t0 = time.perf_counter()
            res = pipe.process(f"u{n}", s.query, n * 5000.0)
            wall = (time.perf_counter() - t0) * 1000.0
            p = pipe.last if isinstance(pipe, FullRestartPipeline) else pipe
            ids = _turn_evidence(p, res, st, f"u{n}")
            r.evidence = _ev(st, ids)
            r.keys = _keys(r.evidence)
            ga = res.grounded
            ver_ms = 0.0
            if mode == "grounded":
                from streamrag.evaluation.instrument import AnswerInstrument
                r.claims = AnswerInstrument.split_grounded(ga)
                r.answer = ga.text if ga is not None else ""
                gen_ms = (ga.timings_ms or {}).get("generation", 0.0) if ga is not None else 0.0
                ver_ms = sum((ga.timings_ms or {}).get(k, 0.0) for k in ("claim_verification", "repair", "validation",
                                                                         "citation_mapping")) if ga is not None else 0.0
                llm_calls = ga.llm_calls if ga is not None else 0
                pt, ot = (ga.prompt_tokens, ga.output_tokens) if ga is not None else (0, 0)
                gtime = res.timings_ms.get("grounded_answer", 0.0)
            else:
                r.claims, gen_ms, ver_ms, llm_calls, pt, ot, gtime = _free_answer(spec, st, s, r.evidence, llm, mode)
                r.answer = " ".join(t for t, _ in r.claims)
            ad = res.adaptive
            retr = res.timings_ms.get("retrieval", 0.0)
            interp = res.timings_ms.get("interpretation", 0.0)
            if ad:
                ops = {"retrieval_calls": sum(a.ops.searches for a in ad),
                       "embeddings": sum(a.ops.dense_searches for a in ad),
                       "lexical_searches": sum(a.ops.lexical_searches for a in ad),
                       "reranker_calls": sum(a.ops.reranker_calls for a in ad),
                       "chunks_retrieved": sum(a.ops.chunks_returned for a in ad),
                       "k_values": [x["k"] for a in ad for x in a.searches],
                       "iterations": sum(a.state.iteration for a in ad),
                       "expansions": sum(max(0, len(a.searches) - 1) for a in ad),
                       "cache_hit": int(all(a.ops.searches == 0 for a in ad)),
                       "documents_retrieved": len({i.split("§")[0] for a in ad for x in a.searches
                                                   for i in x["items"]})}
                r.adaptive = {"strategies": [a.plan.strategy.value for a in ad],
                              "complexity": [a.analysis.complexity.value for a in ad],
                              "stops": [a.state.stop_reason.value for a in ad if a.state.stop_reason],
                              "assessments": [a.assessment.status for a in ad],
                              "hops": [h.bridge for a in ad for h in a.hops if h.bridge]}
            else:
                nq = res.retrievals
                cm = s2.cfg.streaming.retrieval_mode
                created = [x for x in p.ledger.for_utterance(f"u{n}") if x.status == "completed"
                           and x.retrieval_status != "cache_hit"]
                ops = {"retrieval_calls": nq, "embeddings": nq if cm in ("dense", "hybrid") else 0,
                       "lexical_searches": nq if cm in ("bm25", "hybrid") else 0,
                       "reranker_calls": nq if s2.cfg.streaming.rerank else 0,
                       "chunks_retrieved": sum(len(x.evidence_ids) for x in created),
                       "k_values": [s2.cfg.multi_intent.max_candidates_per_intent] * nq, "iterations": nq,
                       "expansions": 0, "cache_hit": int(nq == 0 and bool(ids)),
                       "documents_retrieved": len({e["document_id"] for e in _ev(st, [i for x in created
                                                                                      for i in x.evidence_ids])})}
            reused = 0
            store = p.engine.store
            for i in ids:
                rec = store.records.get(i)
                if rec is not None:
                    q = p.ledger.get(rec.first_seen_query) if rec.first_seen_query in {x.query_id for x in
                                                                                       p.ledger.all()} else None
                    if q is not None and q.utterance_id != f"u{n}":
                        reused += 1
            ops.update({"evidence_reused": reused, "llm_calls": llm_calls, "prompt_tokens": pt, "output_tokens": ot,
                        "phase6_cache_hits": res.cache_hits})
            r.ops = ops
            r.latency = {"ttfe_after_end": round(interp + retr, 3), "ttfa_after_end": round(wall, 3),
                         "ttva_after_end": round(wall, 3), "total_after_end": round(wall, 3),
                         "retrieval_ms": round(retr, 3), "generation_ms": round(gen_ms or 0.0, 3),
                         "verification_ms": round(ver_ms, 3), "answer_stage_ms": round(gtime, 3)}
        except Exception as exc:     # noqa: BLE001
            r.status, r.error = "failed", f"{exc.__class__.__name__}: {exc}"
        out.append(r)
    if pipe is not None:
        pipe.close()
    return out


def _free_answer(spec, st, s: EvalSample, evidence: list[dict], llm, mode: str):
    """Answer-stage ablations over the pipeline's evidence: unverified free-form answer, or verified claims that keep
    the LLM's own (unvalidated) citation labels."""
    from streamrag.evaluation.instrument import AnswerInstrument
    from streamrag.generation.generator import GroundedAnswerGenerator
    gen = GroundedAnswerGenerator(llm, 1)
    labels = {f"E{k + 1}": e["chunk_id"] for k, e in enumerate(evidence)}
    t0 = time.perf_counter()
    cand = gen.generate_free(s.query, [(f"E{k + 1}", e["citation"], e["text"]) for k, e in enumerate(evidence)],
                             "evidence_labels")
    gen_ms = (time.perf_counter() - t0) * 1000.0
    claims = AnswerInstrument.split_free(cand.sentences, labels)
    ver_ms = 0.0
    if mode == "no_citation_validation":
        t1 = time.perf_counter()
        v = spec["_verifier"]
        pool = {e["chunk_id"]: e["text"] for e in evidence}
        kept = []
        for k, (text, cited) in enumerate(claims):
            if pool and v.verify(f"a{k}", text, [c for c in cited if c in pool], pool).supported:
                kept.append((text, cited))           # verified, but citations NOT validated / remapped
        claims = kept
        ver_ms = (time.perf_counter() - t1) * 1000.0
    return claims, gen_ms, ver_ms, cand.llm_calls, cand.prompt_tokens, cand.output_tokens, gen_ms + ver_ms


# ------------------------------------------------------------------------------------------------ runtime
async def _drive_session(spec: dict, st, session: list[EvalSample], llm, faults=None, trace_dir: Path | None = None,
                         name: str | None = None):
    from streamrag.runtime import StreamingRuntime
    rt = await StreamingRuntime(st.cfg, st, llm=llm, faults=faults).start()
    sid = (name or session[0].session_id).replace(".", "-")      # memoryless: one runtime (and trace) per turn
    rt.start_session(sid)
    interval = spec.get("interval_ms", 250) / 1000.0
    first: dict[str, float] = {}
    for n, s in enumerate(session, start=1):
        uid = f"u{n}"
        chunks = s.stream or [type("C", (), {"text": c, "replaces": None})() for c in chunks_for(s.query)]
        for c in chunks:
            if c.replaces is not None:
                rt.push_transcript_delta(sid, uid, c.text, replaces=c.replaces)
            else:
                rt.push_transcript_delta(sid, uid, c.text)
            await asyncio.sleep(interval)
        rt.end_utterance(sid, uid)
        if not spec.get("overlap_turns"):          # a conversation: the next question after the answer
            await rt.wait_idle(spec.get("turn_timeout_s", 120))
        await asyncio.sleep(spec.get("gap_ms", 300) / 1000.0)
    await rt.complete_session(sid, spec.get("turn_timeout_s", 120))
    evs = rt.events(sid)
    rs = rt.sessions[sid]
    summ = rt.summary()
    await rt.shutdown()
    for e in evs:
        if e.type.value == "CHUNK_RECEIVED" and e.utterance_id not in first:
            first[e.utterance_id] = e.t_wall_ms
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{spec['id']}__{sid}.jsonl").write_text("".join(e.canonical_json() + "\n" for e in evs))
    return evs, rs, first, summ


def turn_evidence(events, uid: str) -> list[str] | None:
    """Evidence handed to the final answer of turn ``uid``: the items of the turn's last EVIDENCE_FUSED event before
    its last ANSWER_COMMITTED (rank order), plus items of earlier fusions of the same turn that the committed answer
    still cites (answer sections reused from a draft). Draft-time evidence the final answer does not use is left out,
    which matches the batch pipeline (final retrieval state of the turn). None if the turn fused no evidence. Read
    from the events, not from the session ledger, because later turns supersede the ledger state. Accepts
    TelemetryEvent objects or trace dicts."""
    def get(e, k):
        if isinstance(e, dict):
            return e[k]
        v = getattr(e, k)
        return v.value if k == "type" else v
    mine = [e for e in events if get(e, "utterance_id") == uid]
    commits = [e for e in mine if get(e, "type") == "ANSWER_COMMITTED"]
    last_seq = get(commits[-1], "seq") if commits else None
    fused = [e for e in mine if get(e, "type") == "EVIDENCE_FUSED" and (last_seq is None or get(e, "seq") <= last_seq)]
    if not fused:
        return None
    ids = [it["evidence_id"] for it in get(fused[-1], "payload").get("items") or [] if it.get("evidence_id")]
    cited = set(get(commits[-1], "payload").get("citations") or []) if commits else set()
    for e in reversed(fused[:-1]):
        for it in get(e, "payload").get("items") or []:
            if it.get("evidence_id") and it.get("citation") in cited and it["evidence_id"] not in ids:
                ids.append(it["evidence_id"])
    return list(dict.fromkeys(ids))


def run_runtime(spec: dict, st, session: list[EvalSample], llm, trace_dir: Path | None = None,
                faults_fn=None) -> list[TurnRecord]:
    from streamrag.evaluation.instrument import AnswerInstrument
    from streamrag.evaluation.metrics import latency as L
    from streamrag.evaluation.metrics import streaming as SM
    s2 = with_cfg(st, {"streaming.rerank": False, **spec.get("overrides", {})})
    groups = [[x] for x in session] if spec.get("memoryless") else [session]
    out = []
    for g in groups:
        t_cpu = time.process_time()
        try:
            evs, rs, first, summ = asyncio.run(_drive_session(spec, s2, g, llm, faults_fn() if faults_fn else None,
                                                              trace_dir, g[0].sample_id if spec.get("memoryless")
                                                              else None))
        except Exception as exc:     # noqa: BLE001
            for s in g:
                out.append(TurnRecord(s.sample_id, spec["id"], status="failed", error=f"{exc.__class__.__name__}: {exc}"))
            continue
        cpu_ms = (time.process_time() - t_cpu) * 1000.0
        sess_stream = SM.session(evs)
        for n, s in enumerate(g, start=1):
            uid = f"u{n}"
            r = TurnRecord(s.sample_id, spec["id"], query_used=s.query, session_stream=sess_stream)
            mine = [e for e in evs if e.utterance_id == uid]
            try:
                led = rs.session.ledger
                r.evidence = _ev(st, turn_evidence(evs, uid) or [])
                r.keys = _keys(r.evidence)
                ga = rs.lane.finals.get(uid) if rs.lane is not None else None
                r.claims = AnswerInstrument.split_grounded(ga)
                r.answer = ga.text if ga is not None else ""
                if uid in first:
                    r.latency = L.milestones(evs, first[uid], uid)
                    keys = [c.key for c in s.expected_claims if c.key]
                    r.stream = SM.turn(evs, first[uid], uid, keys)
                ad_starts = [e for e in mine if e.type.value == "RETRIEVAL_STARTED" and e.payload.get("adaptive_search")]
                fixed_starts = [e for e in mine if e.type.value == "RETRIEVAL_STARTED"
                                and not e.payload.get("adaptive_search")]
                adaptive_on = s2.cfg.adaptive_retrieval.enabled
                searches = len(ad_starts) if adaptive_on else len(fixed_starts)
                emb = sum(1 for e in ad_starts if "dense" in (e.payload.get("retrievers") or [])) if adaptive_on \
                    else len(fixed_starts)
                llm_ev = [e for e in mine if e.type.value == "LLM_CALL"]
                cancelled = [e for e in mine if e.type.value == "TASK_CANCELLED"]
                stops = [e.payload for e in mine if e.type.value == "RETRIEVAL_STOPPED"]
                pols = [e.payload for e in mine if e.type.value == "RETRIEVAL_POLICY_SELECTED"]
                r.ops = {"retrieval_calls": searches, "embeddings": emb,
                         "lexical_searches": sum(1 for e in ad_starts if "bm25" in (e.payload.get("retrievers") or []))
                         if adaptive_on else len(fixed_starts), "reranker_calls": 0,
                         "chunks_retrieved": sum(int(e.payload.get("n_results") or 0) for e in mine
                                                 if e.type.value == "RETRIEVAL_COMPLETED"
                                                 and e.payload.get("adaptive_search")) if adaptive_on else None,
                         "k_values": [e.payload.get("k") for e in ad_starts] if adaptive_on else [],
                         "iterations": sum(int(p.get("iterations") or 0) for p in stops) if adaptive_on else searches,
                         "expansions": sum(1 for e in mine if e.type.value == "RETRIEVAL_EXPANDED"),
                         "cache_hit": int(any(e.type.value in ("CACHE_HIT", "QUERY_REUSED") for e in mine)
                                          and searches == 0),
                         "evidence_reused": sum(1 for e in mine if e.type.value == "QUERY_REUSED"),
                         "documents_retrieved": len({e["document_id"] for e in r.evidence}),
                         "llm_calls": len(llm_ev),
                         "prompt_tokens": sum(int(e.payload.get("prompt_tokens") or 0) for e in llm_ev),
                         "output_tokens": sum(int(e.payload.get("output_tokens") or 0) for e in llm_ev),
                         "cancelled_tasks": len(cancelled),
                         "cancelled_by_type": {t: sum(1 for e in cancelled if e.payload.get("task_type") == t)
                                               for t in {e.payload.get("task_type") for e in cancelled}},
                         "queries": len([e for e in mine if e.type.value == "QUERY_GENERATED"]),
                         "superseded_queries": len([e for e in mine if e.type.value == "QUERY_SUPERSEDED"])}
                r.adaptive = {"strategies": [p.get("strategy") for p in pols],
                              "complexity": [p.get("complexity") for p in pols],
                              "stops": [p.get("stop_reason") for p in stops],
                              "assessments": [p.get("assessment") for p in stops],
                              "hops": [e.payload.get("bridge") for e in mine if e.type.value == "HOP_CREATED"]}
                r.events_digest = _waste(mine, led)
                r.resources = {"session_cpu_ms": round(cpu_ms, 1), "turns_in_session": len(g)}
                if not any(e.type.value == "TURN_COMPLETED" for e in mine):
                    r.error = "turn_not_completed"
                if ga is not None and ga.fallback:
                    r.error = (r.error or "") + f" generation_fallback: {ga.fallback}"
            except Exception as exc:     # noqa: BLE001
                r.status, r.error = "failed", f"{exc.__class__.__name__}: {exc}"
            out.append(r)
    return out


def _waste(mine, ledger) -> dict:
    """Worker time of tasks whose query / answer ended superseded, cancelled or stale (measured exec_ms)."""
    wasted = useful = 0.0
    gen_wasted = 0.0
    for e in mine:
        if e.type.value not in ("TASK_COMPLETED", "TASK_CANCELLED", "TASK_FAILED", "TASK_TIMED_OUT"):
            continue
        p = e.payload
        ms = float((p.get("wall") or {}).get("exec_ms") or 0.0)
        tt = p.get("task_type")
        dead = e.type.value == "TASK_CANCELLED"
        if p.get("query_id"):
            rec = ledger.get(p["query_id"])
            dead = dead or rec is None or rec.status == "cancelled" or rec.stale_reason is not None
        if dead:
            wasted += ms
            if tt in ("generation", "draft", "answer_extractive"):
                gen_wasted += ms
        else:
            useful += ms
    stale = sum(1 for e in mine if e.type.value == "STALE_RESULT_DISCARDED")
    return {"wasted_exec_ms": round(wasted, 3), "useful_exec_ms": round(useful, 3),
            "wasted_generation_ms": round(gen_wasted, 3), "stale_discarded": stale}
