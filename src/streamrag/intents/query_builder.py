"""IntentQueryBuilder: one retrieval query per intent version (docs/multi_intent/04).

query = resolved intent words (anaphora replaced by the antecedent's words)
      + inherited context not already present (distributed "for X", follow-up topic)
      + constraint text in scope (local + global) not already present

Every component is verbatim speech with a source span (this utterance or an earlier one), so nothing is injected
that the user did not say. Entities, numbers, negations and terminology survive because nothing inside a component
is rewritten; only fillers and request framing were removed upstream.
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.models.intents import Constraint, Intent, IntentQuery, QueryComponent


class IntentQueryBuilder:
    def __init__(self, terms_fn: Callable[[str], list[str]]) -> None:
        self.terms_fn = terms_fn

    def build(self, intent: Intent, constraints: list[Constraint]) -> IntentQuery:
        comps: list[QueryComponent] = list(intent.components)
        terms: list[str] = list(dict.fromkeys(self.terms_fn(intent.resolved_text)))
        for ic in intent.inherited_context:
            if ic.reason == "anaphora" or ic.reason == "correction_aspect":
                continue                                   # already inside resolved_text
            new = [t for t in self.terms_fn(ic.text) if t not in terms]
            if new:
                comps.append(QueryComponent(text=ic.text, source="inherited", ref=ic.from_intent, span=ic.source_span))
                terms += new
        for c in constraints:
            if intent.intent_id not in c.applies_to:
                continue
            new = [t for t in self.terms_fn(c.text) if t not in terms]
            if new:
                comps.append(QueryComponent(text=c.text, source="constraint", ref=c.constraint_id, span=c.source_span))
                terms += new
        text = " ".join(c.text for c in comps).strip()
        return IntentQuery(intent_id=intent.intent_id, intent_version=intent.version, utterance_id=intent.utterance_id,
                           text=text, components=comps, terms=list(dict.fromkeys(terms)))
