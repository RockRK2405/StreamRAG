"""Answer consistency check (Phase 7; docs/answer/08).

When an answer is revised, a new claim must not contradict a claim the answer keeps (V1 "Applicants need A." /
V2 "Applicants do not need A."). New claims are paired with the retained claims of the same section and with each
other; a pair the entailment model labels a contradiction in either direction is a consistency conflict. Pairs are
only scored when they share at least half of the shorter claim's content terms: entailment models label two
different statements about the same topic as "contradiction", so only claims about the same proposition are compared.

Both claims of a conflicting pair that are each supported by *different* evidence are an evidence conflict and are
presented as such; otherwise the new claim is withdrawn (the retained one was already verified in the answer).
In ``rules`` mode (no entailment model) the check is limited to same-content claims with different numbers or
opposite negation.
"""

from __future__ import annotations

from collections.abc import Callable

from streamrag.claims.textcheck import negated, numbers


class ConsistencyChecker:
    def __init__(self, terms_fn: Callable[[str], list[str]], nli=None) -> None:
        self.terms_fn, self.nli = terms_fn, nli

    def conflicts(self, new: list[tuple[str, str]], kept: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """new / kept: [(claim id, text)] -> conflicting (new id, other id) pairs."""
        pairs = []
        for i, (a, ta) in enumerate(new):
            for b, tb in kept + new[i + 1:]:
                sa, sb = set(self.terms_fn(ta)), set(self.terms_fn(tb))
                if a != b and sa and sb and len(sa & sb) / min(len(sa), len(sb)) >= 0.5:
                    pairs.append((a, ta, b, tb))
        if not pairs:
            return []
        out = []
        if self.nli is not None:
            preds = self.nli.predict([(ta, tb) for _, ta, _, tb in pairs] + [(tb, ta) for _, ta, _, tb in pairs])
            n = len(pairs)
            for k, (a, _, b, _) in enumerate(pairs):
                if preds[k]["label"] == "contradiction" or preds[n + k]["label"] == "contradiction":
                    out.append((a, b))
            return out
        for a, ta, b, tb in pairs:
            same = set(self.terms_fn(ta)) == set(self.terms_fn(tb))
            if same and (numbers(ta) != numbers(tb) or negated(ta) != negated(tb)):
                out.append((a, b))
        return out
