"""Human-readable debug stream for the CLI (development aid; the JSONL trace is the source of truth)."""

from __future__ import annotations

from streamrag.models.events import TelemetryEvent


def _ts(ms: float | None) -> str:
    ms = ms or 0.0
    return f"{int(ms // 60000):02d}:{(ms % 60000) / 1000:06.3f}"


def format_event(ev: TelemetryEvent) -> str | None:
    p, t = ev.payload, ev.type.value
    head = f"[{_ts(ev.t_session_ms)}] {ev.utterance_id or '--':4s}"
    if t == "CHUNK_RECEIVED":
        status = p.get("status")
        extra = f"  ({status})" if status != "accepted" else ""
        return f"{head} USER       #{p['input']['chunk_index']}: \"{p['input']['text']}\"{extra}"
    if t == "RETRIEVAL_DECISION":
        sg = p.get("signals", {})
        return (f"{head} CONTROLLER {p['decision']:8s} {p['reason']}  conf={p['confidence']:.2f} "
                f"stab={sg.get('semantic_stability')} worth={sg.get('retrieval_worthiness')} "
                f"nov={sg.get('novelty')} act={sg.get('act')}")
    if t == "QUERY_UPDATED":
        sup = f" supersedes {p['supersedes']} ({p['relation']})" if p.get("supersedes") else ""
        return f"{head} QUERY      {p['query_id']} v{p['version']} \"{p['query_text']}\"{sup}"
    who = f"{p.get('query_id')}{'/' + ev.intent_id if ev.intent_id else ''}"
    if t == "RETRIEVAL_STARTED":
        return f"{head} RETRIEVAL  START {who} [{p['trigger']}] queue_wait={p['queue_wait_ms']}ms"
    if t == "RETRIEVAL_COMPLETED":
        stale = " STALE(superseded by " + str(p.get("superseded_by")) + ")" if p.get("stale") else ""
        cites = ", ".join(p.get("citations", [])[:3])
        return f"{head} RETRIEVAL  DONE  {who} status={p['status']} {p['latency_ms']}ms{stale} top: {cites}"
    if t == "RETRIEVAL_CANCELLED":
        return f"{head} RETRIEVAL  CANCEL {p['query_id']} ({p['reason']})"
    if t == "RETRIEVAL_SKIPPED":
        if ev.component == "multi_query":
            return f"{head} RETRIEVAL  SKIP {ev.intent_id} ({p['reason']})"
        return None   # already visible in the decision line
    if t == "UTTERANCE_FINALIZED":
        return f"{head} UTTERANCE  FINALIZED reason={p['reason']} transcript=\"{p['transcript']}\""
    if t == "INTENTS_UPDATED":
        d = p["delta"]
        ks = [f"{k['constraint_id']}:{k['text']!r}->{','.join(k['applies_to'])}" for k in
              p["global_constraints"] + p["local_constraints"]]
        parts = [f"+{','.join(d['added'])}" if d["added"] else "",
                 f"~{','.join(m['intent_id'] for m in d['modified'])}" if d["modified"] else "",
                 f"-{','.join(d['removed'])}" if d["removed"] else "",
                 f"superseded {','.join(x['old'] + '->' + x['new'] for x in d['superseded'])}" if d["superseded"] else ""]
        return (f"{head} INTENTS    v{p['version']} {' '.join(x for x in parts if x)} | "
                + "; ".join(f"{i['intent_id']}[{i['type']}] {i['text']!r}" for i in p["intents"])
                + (f" | constraints {'; '.join(ks)}" if ks else ""))
    if t in ("INTENT_DETECTED", "INTENT_UPDATED"):
        return None                                   # summarised by the INTENTS line
    if t == "INTENT_SUPERSEDED":
        return f"{head} INTENT     {p['old']} SUPERSEDED by {p['new']} ({p['cue']})"
    if t == "QUERY_GENERATED":
        rel = f" supersedes {p['supersedes']} ({p['relation']})" if p.get("supersedes") else ""
        return f"{head} QUERY      {ev.query_id} for {ev.intent_id} v{p['intent_version']} \"{p['query_text']}\"{rel}"
    if t == "MULTI_QUERY_STARTED":
        reuse = f" reuse {','.join(p['reused_intents'])}" if p["reused_intents"] else ""
        return f"{head} MULTI      {p['batch_id']} start {','.join(p['query_ids'])} ({p['dispatch']}){reuse}"
    if t == "MULTI_QUERY_COMPLETED":
        return f"{head} MULTI      {p['batch_id']} done makespan={p['makespan_ms']}ms"
    if t == "EVIDENCE_DEDUPLICATED":
        return None
    if t == "EVIDENCE_FUSED":
        items = ", ".join(f"{i['label']}={i['citation']}<{'+'.join(i['supporting_intents'])}>" for i in p["items"][:6])
        return (f"{head} FUSED      {'final' if p['final'] else 'provisional'} {p['strategy']}/{p['rerank']} "
                f"coverage={p['intent_coverage']} items={len(p['items'])}: {items}")
    if t in ("RERANK_STARTED", "RERANK_COMPLETED"):
        return f"{head} RERANK     {t.split('_')[1].lower()} {p['mode']}"
    # ---------------------------------------------------------------- Phase 6 (adaptive session)
    if t == "CONTEXT_CHANGE_DETECTED":
        ks = (f" +{p['added_constraints']}" if p["added_constraints"] else "") + \
             (f" -{p['removed_constraints']}" if p["removed_constraints"] else "")
        return (f"{head} CHANGE     {p['change_id']} {p['change_type']} affected={p['affected_intents']} "
                f"new={p['new_intents']}{ks} frame={p['frame_action']} conf={p['confidence']} ({p['cue']})")
    if t == "DELTA_PLAN_CREATED":
        return (f"{head} PLAN       {p['plan_id']} create={[q['intent_id'] + ':' + q['query']['text'] for q in p['queries_to_create']]}"
                f" reuse={[q['intent_id'] + ':' + q['action'] for q in p['queries_to_reuse']]}"
                f" supersede={p['queries_to_supersede']} evidence retain={len(p['evidence_to_retain'])}"
                f" revalidate={len(p['evidence_to_revalidate'])} discard={len(p['evidence_to_discard'])}"
                f" claims={p['claims_to_revalidate']}")
    if t == "QUERY_REUSED":
        return f"{head} REUSE      {p['intent_id']} {p['action']} of {p['reused_query_id']} -> {p['query_id']}"
    if t in ("EVIDENCE_RETAINED", "EVIDENCE_INVALIDATED", "EVIDENCE_REVALIDATED"):
        return (f"{head} EVIDENCE   {p['evidence_id']}@{p['intent_id']} {p['from_status']}->{p['to_status']} "
                f"({p['rule']})")
    if t == "CLAIM_CREATED":
        return f"{head} CLAIM      {p['claim_id']} new for {p['intent_id']}: \"{p['text'][:70]}\""
    if t in ("CLAIM_INVALIDATED", "CLAIM_REVALIDATED"):
        return f"{head} CLAIM      {p['claim_id']} {p['from_status']}->{p['to_status']} ({p['reason']})"
    if t in ("ANSWER_VERSION_CREATED", "ANSWER_VERSION_UPDATED"):
        regen = [s["section_id"] for s in p["sections"] if s["needs_regeneration"]]
        return (f"{head} ANSWER     {p['answer_id']} v{p['version']} {p['kind']} frame={p['frame_id']} "
                f"claims={p['claim_ids']} re-render={regen} | {p['change_summary']}")
    if t in ("SESSION_VERSION_CREATED", "QUERY_SUPERSEDED"):
        return None
    # Phase 7: grounded answers (other ANSWER_* / CLAIM_* / CITATION_* events are telemetry only)
    if t == "LLM_CALL":
        return (f"{head} LLM        {p['model']} {p['purpose']} ok={p['ok']} tokens {p['prompt_tokens']}->"
                f"{p['output_tokens']} total={round(p['wall']['total_ms'])}ms")
    if t == "CLAIM_REJECTED":
        return f"{head} VERIFY     {p['status']} -> {p['action']}: \"{p['text'][:70]}\""
    if t == "VALIDATION_RETRIEVAL":
        return f"{head} VERIFY     retrieval for {p['claim_id']}: {len(p['new_evidence'])} new -> {p['status_after']}"
    if t == "ANSWER_COMPLETED":
        tag = f"{head} GROUNDED   {p['answer_id']} v{p['version']} {p['status']}" + (" partial" if p["partial"] else "")
        if p["status"] == "DRAFT":
            return f"{tag}: \"{' '.join(p['text'].split())[:90]}\""
        return tag + "\n" + "\n".join(("               " + ln) if ln else "" for ln in p["text"].splitlines())
    if t == "TURN_COMPLETED" and "session" in p:
        s = p["session"]
        return (f"{head} TURN       net={s['net_change_types']} turn_needs={s['turn_intents']} "
                f"sub_queries={p['sub_queries']} answer={s['answer_id']}{' (updated)' if s['answer_changed'] else ''}"
                f" session_v={s['session_version']}" + (f" deferred={s['deferred']}" if s["deferred"] else ""))
    if t == "TURN_COMPLETED" and "intent_set" in p:
        m = p.get("metrics", {})

        def ms2(k: str) -> str:
            return "n/a" if m.get(k) is None else f"{m[k]}ms"
        return (f"{head} TURN       intents={m.get('intent_count')} retrievals={m.get('retrieval_count')} "
                f"versions={m.get('intent_set_versions')} coverage={m.get('intent_coverage')} "
                f"lead_time={ms2('lead_time_ms')} post_final={ms2('post_final_retrieval_latency_ms')} "
                f"sub_queries={p['sub_queries']}")
    if t == "TURN_COMPLETED":
        m = p.get("metrics", {})

        def ms(k: str) -> str:
            return "n/a" if m.get(k) is None else f"{m[k]}ms"
        return (f"{head} TURN       retrievals={m.get('retrieval_count')} lead_time={ms('lead_time_ms')} "
                f"ttfr={ms('ttfr_ms')} post_final={ms('post_final_retrieval_latency_ms')} "
                f"final_query={p.get('final_query_id')}")
    if t == "ERROR":
        return f"{head} ERROR      {p['component']}: {p['error_class']} - {p['detail']} -> {p['action']}"
    return None
