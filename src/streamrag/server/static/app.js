"use strict";
// StreamRAG demo UI. All server text is inserted with textContent (retrieved documents are untrusted data).

const STAGES = [["query", "Query"], ["understanding", "Understanding"], ["retrieval", "Retrieval"],
  ["evidence", "Evidence"], ["generating", "Generating"], ["verifying", "Verifying"], ["final", "Final"]];
const CITE_RE = /\[([A-Z0-9][A-Z0-9-]* §\d+(?:\s*;\s*[A-Z0-9][A-Z0-9-]* §\d+)*)\]/g;
const FAST = new Set(["LEXICAL", "FAST_VECTOR", "CACHE_REUSE", "SESSION_REUSE"]);
const $ = (id) => document.getElementById(id);
let session = null, source = null, busy = false;
const turns = new Map();

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

async function api(method, path, body) {
  const r = await fetch(path, {method, headers: body ? {"Content-Type": "application/json"} : {},
    body: body ? JSON.stringify(body) : undefined});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function logLine(text) {
  const li = el("li");
  li.append(el("time", null, new Date().toLocaleTimeString([], {hour12: false})), el("span", null, text));
  $("log").prepend(li);
  while ($("log").children.length > 200) $("log").lastChild.remove();
}

async function checkReady() {
  try {
    const r = await fetch("/ready");
    const d = await r.json();
    const llm = d.checks && d.checks.llm;
    $("b-ready").textContent = d.ready ? (d.degraded ? "Ready (degraded)" : "Ready") : "Not ready";
    $("b-ready").className = "badge " + (d.ready ? (d.degraded ? "warn" : "ok") : "bad");
    if (llm) {
      $("b-llm").textContent = llm.ok ? `LLM: ${llm.model} (local)` : "LLM unavailable: verified extractive answers";
      $("b-llm").className = "badge " + (llm.ok ? "ok" : "warn");
    }
    const idx = d.checks && d.checks.index;
    $("b-data").hidden = !(idx && idx.test_fixture);
  } catch (e) {
    $("b-ready").textContent = "Server unreachable";
    $("b-ready").className = "badge bad";
  }
}

async function loadScenarios() {
  const {scenarios} = await api("GET", "/api/scenarios");
  const box = $("scenarios");
  box.replaceChildren();
  for (const s of scenarios) {
    const b = el("button", "scenario");
    b.type = "button";
    b.append(el("strong", null, s.title), el("span", null, s.shows));
    b.title = s.turns.map((t) => t.join(" ")).join("  /  ");
    b.addEventListener("click", () => runScenario(s.id));
    box.append(b);
  }
}

async function newSession() {
  if (source) source.close();
  if (session) fetch(`/api/sessions/${session}`, {method: "DELETE"}).catch(() => {});
  turns.clear();
  $("turns").replaceChildren();
  $("notices").replaceChildren();
  const d = await api("POST", "/api/sessions");
  session = d.session_id;
  source = new EventSource(`/api/sessions/${session}/events`);
  source.onmessage = (m) => handle(JSON.parse(m.data));
  source.onerror = () => logLine("Event stream interrupted");
  logLine(`New conversation ${session}`);
}

async function runScenario(id) {
  if (busy) return;
  setBusy(true);
  try {
    await newSession();
    await api("POST", `/api/sessions/${session}/scenario`, {id});
    logLine(`Scenario ${id} started`);
  } catch (e) {
    notice("error", e.message);
  } finally {
    setBusy(false);
  }
}

function setBusy(v) {
  busy = v;
  $("send").disabled = v;
  for (const b of document.querySelectorAll(".scenario")) b.disabled = v;
}

async function ask(text) {
  if (!text.trim() || busy) return;
  setBusy(true);
  try {
    if (!session) await newSession();
    const words = text.trim().split(/\s+/);
    for (let i = 0; i < words.length; i += 3) {
      await api("POST", `/api/sessions/${session}/chunks`, {text: words.slice(i, i + 3).join(" ")});
      await new Promise((r) => setTimeout(r, 450));
    }
    await api("POST", `/api/sessions/${session}/end`);
  } catch (e) {
    notice("error", e.message);
  } finally {
    setBusy(false);
  }
}

function notice(level, text) {
  $("notices").append(el("div", `notice ${level}`, text));
}

function turnCard(uid) {
  if (turns.has(uid)) return turns.get(uid);
  $("empty")?.remove();
  const card = el("article", "turn");
  const you = el("div", "you");
  const transcript = el("div", "transcript live");
  you.append(el("span", "who", "You"), transcript);
  const stages = el("div", "stages");
  const stageEls = {};
  for (const [k, label] of STAGES) {
    stageEls[k] = el("span", `stage ${k}`, label);
    stages.append(stageEls[k]);
  }
  const meta = el("div", "meta");
  const needs = el("div", "meta-row");
  needs.append(el("span", "label", "Needs"));
  const plans = el("div", "meta-row");
  plans.append(el("span", "label", "Retrieval"));
  const evidence = el("div", "meta-row");
  evidence.append(el("span", "label", "Evidence"));
  meta.append(needs, plans, evidence);
  needs.hidden = plans.hidden = evidence.hidden = true;
  const notes = el("ul", "notes");
  const answer = el("div", "answer");
  answer.hidden = true;
  const metrics = el("div", "metrics");
  card.append(you, stages, meta, notes, answer, metrics);
  $("turns").append(card);
  if (window.innerWidth < 760) card.scrollIntoView({behavior: "smooth", block: "start"});
  const t = {card, transcript, stageEls, needs, plans, evidence, notes, answer, metrics, planSeen: new Set(),
    final: false};
  turns.set(uid, t);
  return t;
}

function setStage(t, stage) {
  let reached = true;
  for (const [k] of STAGES) {
    const e = t.stageEls[k];
    e.classList.toggle("now", k === stage);
    e.classList.toggle("done", reached && k !== stage);
    if (k === stage) reached = false;
  }
}

function renderWithCitations(parent, text) {
  let last = 0;
  for (const m of text.matchAll(CITE_RE)) {
    parent.append(document.createTextNode(text.slice(last, m.index)));
    for (const key of m[1].split(/\s*;\s*/)) parent.append(citeButton(key));
    last = m.index + m[0].length;
  }
  parent.append(document.createTextNode(text.slice(last)));
}

function citeButton(key) {
  const b = el("button", "cite", key);
  b.type = "button";
  b.title = "Show the source section";
  b.addEventListener("click", () => showSource(key));
  return b;
}

async function showSource(key) {
  const box = $("source");
  try {
    const s = await api("GET", `/api/sources/${encodeURIComponent(key)}`);
    box.replaceChildren(el("h3", null, s.document_title), el("p", "loc", `${s.citation} · ${s.section_title || ""}`),
      el("blockquote", null, s.text), el("p", "hint", s.note));
  } catch (e) {
    box.replaceChildren(el("p", "hint", `Source not available: ${e.message}`));
  }
}

function renderAnswer(t, ev) {
  const a = t.answer;
  a.hidden = false;
  a.replaceChildren();
  const head = el("div", "answer-head");
  const final = ev.status === "VALIDATED_FINAL";
  head.append(el("span", `tag ${final ? "final" : ev.status === "BLOCKED" ? "blocked" : "draft"}`,
    final ? "Verified" : ev.status === "BLOCKED" ? "Blocked" : "Draft"));
  if (ev.mode === "extractive" || (ev.backend || "").includes("extractive"))
    head.append(el("span", "hint", "extractive answer (no LLM)"));
  a.append(head);
  if (final && ev.text) {
    // the validated answer as rendered by the system: section headings, conflict / version labels, citations
    const p = el("div", "answer-text final-text");
    renderWithCitations(p, ev.text);
    a.append(p);
  } else if (ev.claims && ev.claims.length) {
    const ul = el("ul", "claims");
    let lastSection = null;
    const multi = new Set(ev.claims.map((c) => c.section)).size > 1;
    for (const c of ev.claims) {
      const li = el("li", c.kind === "uncertainty" ? "uncertainty" : "");
      if (multi && c.section !== lastSection) li.append(el("span", "section", c.section));
      lastSection = c.section;
      li.append(document.createTextNode(c.text));
      for (const k of c.citations) li.append(citeButton(k));
      ul.append(li);
    }
    a.append(ul);
  } else {
    const p = el("div", `answer-text ${final ? "" : "draft"}`);
    renderWithCitations(p, ev.text || "");
    a.append(p);
  }
}

function note(t, cls, text) {
  const last = t.notes.lastElementChild;
  if (last && last.textContent === text) return;
  t.notes.append(el("li", cls, text));
}

function handle(ev) {
  if (ev.type === "hello") return;
  if (ev.type === "notice" && !ev.utterance) { notice(ev.level, ev.text); logLine(ev.text); return; }
  const uid = ev.utterance;
  if (!uid) return;
  const t = turnCard(uid);
  switch (ev.type) {
    case "transcript": t.transcript.textContent = ev.text; break;
    case "status": setStage(t, ev.stage); logLine(`${uid}: ${ev.text}`); break;
    case "intents":
      if (!ev.items.length) break;                   // a correction refines the existing need: nothing new to show
      t.needs.hidden = false;
      t.needs.replaceChildren(el("span", "label", "Needs"));
      ev.items.forEach((i, n) => t.needs.append(el("span", "chip", `${n + 1} · ${i.text}`)));
      break;
    case "plan": {
      const key = `${ev.intent}:${ev.strategy}`;
      if (t.planSeen.has(key)) break;
      t.planSeen.add(key);
      t.plans.hidden = false;
      const c = el("span", `chip plan ${FAST.has(ev.strategy) ? "fast" : ""}`, ev.label);
      c.title = `complexity ${ev.complexity}; retrievers ${ev.retrievers.join(" + ") || "-"}; top-k ${ev.top_k}`;
      t.plans.append(c);
      logLine(`${uid}: plan ${ev.strategy} (${ev.complexity})`);
      break;
    }
    case "evidence":
      t.evidence.hidden = false;
      t.evidence.replaceChildren(el("span", "label", "Evidence"));
      for (const k of ev.citations) t.evidence.append(citeButton(k));
      break;
    case "change": note(t, "change", ev.text); logLine(`${uid}: ${ev.text}`); break;
    case "cancelled": note(t, "cancelled", ev.text); logLine(`${uid}: ${ev.text}`); break;
    case "reuse": note(t, "reuse", ev.text); break;
    case "answer":
      if (t.final && ev.status === "DRAFT") break;
      t.final = ev.status === "VALIDATED_FINAL";
      renderAnswer(t, ev);
      break;
    case "metrics": {
      t.metrics.replaceChildren();
      for (const [k, label] of [["ttfe_ms", "first evidence"], ["ttfa_ms", "first answer"],
        ["ttva_ms", "verified answer"]]) {
        if (ev[k] !== undefined) {
          const s = el("span");
          s.append(el("b", null, `${Math.round(ev[k])} ms`), document.createTextNode(` ${label}`));
          t.metrics.append(s);
        }
      }
      break;
    }
    case "notice": note(t, "change", ev.text); notice(ev.level, ev.text); break;
    case "turn_done": t.transcript.classList.remove("live"); break;
  }
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("q").value;
  $("q").value = "";
  ask(text);
});
$("newconv").addEventListener("click", () => newSession().catch((e) => notice("error", e.message)));
checkReady();
setInterval(checkReady, 15000);
loadScenarios().catch((e) => notice("error", `Scenarios unavailable: ${e.message}`));
