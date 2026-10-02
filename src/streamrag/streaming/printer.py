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
