"""Deterministic structural validation of intent output (docs/multi_intent/02 "Validation").

Applied to every IntentSet (rule or LLM) before it may drive retrieval. Checks: empty output, duplicate ids,
duplicate intents, identical queries, unsupported types, dangling references, and source spans that do not match
the transcript they claim to quote. Rule output passing these checks is an invariant enforced by tests; LLM output
failing them is rejected (retry once, then fall back to the rule decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass

from streamrag.models.intents import INTENT_TYPES, IntentQuery, IntentSet, SourceSpan


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    detail: str


def _span_issue(sp: SourceSpan, transcripts: dict[str, str], where: str) -> ValidationIssue | None:
    tr = transcripts.get(sp.utterance_id)
    if tr is None:
        return ValidationIssue("unknown_utterance", f"{where}: span refers to unknown utterance {sp.utterance_id}")
    if sp.end > len(tr) or tr[sp.start:sp.end] != sp.text:
        return ValidationIssue("span_mismatch", f"{where}: span {sp.start}-{sp.end} != {sp.text!r}")
    return None


def validate_intent_set(iset: IntentSet, transcripts: dict[str, str], known_intents: set[str] | None = None,
                        terms_fn=None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    known = set(known_intents or set())
    ids = [i.intent_id for i in iset.intents]
    if not iset.intents and iset.original_text.strip():
        issues.append(ValidationIssue("empty_output", "no active intent for a non-empty utterance"))
    if len(ids) != len(set(ids)):
        issues.append(ValidationIssue("duplicate_intent_id", f"{ids}"))
    known |= set(ids)
    seen_text: dict[str, str] = {}
    for i in iset.intents:
        if i.status != "ACTIVE":
            issues.append(ValidationIssue("inactive_in_active_list", i.intent_id))
        if i.intent_type not in INTENT_TYPES:
            issues.append(ValidationIssue("unsupported_type", f"{i.intent_id}: {i.intent_type}"))
        if not i.text.strip() or not i.resolved_text.strip():
            issues.append(ValidationIssue("empty_intent", i.intent_id))
        for sp, where in [(i.provenance.span, f"{i.intent_id}.provenance")] + \
                [(c.span, f"{i.intent_id}.component") for c in i.components] + \
                [(c.source_span, f"{i.intent_id}.inherited") for c in i.inherited_context]:
            bad = _span_issue(sp, transcripts, where)
            if bad:
                issues.append(bad)
        key = " ".join(sorted(terms_fn(i.resolved_text))) if terms_fn else i.resolved_text.lower().strip()
        if key in seen_text:
            issues.append(ValidationIssue("duplicate_intent", f"{i.intent_id} duplicates {seen_text[key]}"))
        seen_text[key] = i.intent_id
    cids = [c.constraint_id for c in iset.global_constraints + iset.local_constraints]
    if len(cids) != len(set(cids)):
        issues.append(ValidationIssue("duplicate_constraint_id", f"{cids}"))
    for c in iset.global_constraints + iset.local_constraints:
        bad = _span_issue(c.source_span, transcripts, c.constraint_id)
        if bad:
            issues.append(bad)
        missing = [a for a in c.applies_to if a not in known]
        if missing:
            issues.append(ValidationIssue("dangling_constraint_scope", f"{c.constraint_id} -> {missing}"))
    for r in iset.relationships:
        if r.type != "CONSTRAINT_OF" and (r.source not in known or r.target not in known):
            issues.append(ValidationIssue("dangling_relationship", f"{r.type} {r.source}->{r.target}"))
    return issues


def validate_queries(queries: list[IntentQuery]) -> list[ValidationIssue]:
    issues = []
    seen: dict[str, str] = {}
    for q in queries:
        norm = " ".join(q.text.lower().split())
        if not norm:
            issues.append(ValidationIssue("empty_query", q.intent_id))
        elif norm in seen:
            issues.append(ValidationIssue("identical_query", f"{q.intent_id} == {seen[norm]}: {q.text!r}"))
        seen[norm] = q.intent_id
    return issues
