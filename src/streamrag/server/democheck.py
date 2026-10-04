"""`streamrag demo-check`: runs every demo scenario headlessly through the frozen final pipeline (same DemoApp,
same StreamingRuntime, no HTTP) and checks the expectations declared in demo/scenarios.yaml. It is the automated
end-to-end demo test (brief §41 / §54) and the one-command replay of the container (spec G1).

Checks per scenario: every turn ends with a VALIDATED_FINAL answer, every citation names an existing corpus section,
no ERROR event, plus the scenario's own `expect` keys. Writes a JSON report; exit code 1 if any scenario fails."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import yaml

from streamrag.server.ui_events import UiMapper

ABSTAIN = re.compile(r"do(?:es)? not (?:contain|state|say|mention)|not (?:established|stated)|no information", re.I)


async def _run(app, sc: dict, timeout_s: float) -> dict:
    s = app.new_session()
    rs = app.rt.sessions[s.sid]
    t0 = time.perf_counter()
    await app._drive(s, sc)
    uids = [f"u{k}" for k in range(1, len(sc.get("turns", [])) + 1)]
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:                       # until every turn has its validated answer
        if all(any(e.utterance_id == u and e.type.value in ("ANSWER_COMMITTED", "ERROR") for e in rs.bus.events)
               for u in uids) and not rs.lane.busy():
            break
        await asyncio.sleep(0.05)
    mapper = UiMapper(lambda uid: rs.lane.finals.get(uid) if rs.lane is not None else None)
    ui = [x for e in list(rs.bus.events) for x in mapper.map(e)]
    errors = [e.payload for e in rs.bus.events if e.type.value == "ERROR"]
    app._close(s)
    return {"ui": ui, "uids": uids, "errors": errors, "wall_s": round(time.perf_counter() - t0, 2)}


def _check(sc: dict, run: dict, sections: set[str]) -> list[str]:
    fails = []
    ui = run["ui"]
    if run["errors"]:
        fails.append(f"ERROR events: {[e.get('error_class') for e in run['errors']]}")
    expects = sc.get("expect") or []
    for k, uid in enumerate(run["uids"]):
        mine = [x for x in ui if x.get("utterance") == uid]
        finals = [x for x in mine if x["type"] == "answer" and x["status"] != "DRAFT"]
        if not finals or finals[-1]["status"] != "VALIDATED_FINAL":
            fails.append(f"{uid}: no VALIDATED_FINAL answer")
            continue
        fin = finals[-1]
        cites = {c for cl in fin["claims"] for c in cl["citations"]}
        bad = sorted(c for c in cites if c not in sections)
        if bad:
            fails.append(f"{uid}: citations to unknown sections {bad}")
        exp = expects[k] if k < len(expects) else None
        if not exp:
            continue
        text = " ".join(cl["text"] for cl in fin["claims"]) or fin["text"]
        low = text.lower()
        plans = {x["strategy"] for x in mine if x["type"] == "plan"}
        changes = {x["change_type"] for x in mine if x["type"] == "change"}
        needs = max((len(x["items"]) for x in mine if x["type"] == "intents"), default=0)
        if exp.get("plans_any") and not plans & set(exp["plans_any"]):
            fails.append(f"{uid}: plan {sorted(plans)} not in {exp['plans_any']}")
        if exp.get("plans_none") and plans & set(exp["plans_none"]):
            fails.append(f"{uid}: unexpected plan {sorted(plans & set(exp['plans_none']))}")
        if exp.get("changes_any") and not changes & set(exp["changes_any"]):
            fails.append(f"{uid}: context change {sorted(changes)} not in {exp['changes_any']}")
        if exp.get("cancelled") and not any(x["type"] == "cancelled" for x in ui):
            fails.append(f"{uid}: no outdated work dropped")
        if exp.get("needs_min") and needs < exp["needs_min"]:
            fails.append(f"{uid}: {needs} needs detected (< {exp['needs_min']})")
        for c in exp.get("cites_all", []):
            if c not in cites:
                fails.append(f"{uid}: missing citation {c}")
        for c in exp.get("cites_none", []):
            if c in cites:
                fails.append(f"{uid}: unexpected citation {c}")
        for t in exp.get("text_all", []):
            if t.lower() not in low:
                fails.append(f"{uid}: answer lacks '{t}'")
        for t in exp.get("text_none", []):
            if t.lower() in low:
                fails.append(f"{uid}: answer contains '{t}'")
        if exp.get("uncertainty") and not ABSTAIN.search(text):
            fails.append(f"{uid}: no 'not in the documents' statement")
    return fails


async def demo_check(app, out: Path | None, only: list[str] | None = None, timeout_s: float = 120.0) -> dict:
    await app.start()
    if app.rt is None:
        raise SystemExit(f"demo-check: pipeline not ready: {app.error}")
    scenarios = yaml.safe_load(app.scenarios_path.read_text()) or []
    sections = {c.citation for c in app.stack.bundle.chunks}
    report = {"ready": app.ready()[1], "scenarios": []}
    try:
        for sc in scenarios:
            if only and sc["id"] not in only:
                continue
            run = await _run(app, sc, timeout_s)
            fails = _check(sc, run, sections)
            fin = {}
            for x in run["ui"]:
                if x["type"] == "answer" and x["status"] != "DRAFT":
                    fin.setdefault(x["utterance"], {}).update({"status": x["status"], "mode": x.get("mode"),
                                                               "claims": x["claims"]})
                if x["type"] == "metrics":
                    fin.setdefault(x["utterance"], {})["metrics"] = {k: v for k, v in x.items() if k.endswith("_ms")}
            report["scenarios"].append({"id": sc["id"], "title": sc.get("title"), "passed": not fails,
                                        "failures": fails, "wall_s": run["wall_s"], "turns": fin,
                                        "plans": sorted({x["strategy"] for x in run["ui"] if x["type"] == "plan"})})
            print(f"{'PASS' if not fails else 'FAIL'}  {sc['id']:18s} {run['wall_s']:6.1f}s  "
                  + ("; ".join(fails) if fails else ""), flush=True)
    finally:
        await app.shutdown()
    n = len(report["scenarios"])
    report["summary"] = {"passed": sum(s["passed"] for s in report["scenarios"]), "total": n,
                         "llm": report["ready"]["checks"].get("llm")}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"demo-check: {report['summary']['passed']}/{n} scenarios passed", flush=True)
    return report
