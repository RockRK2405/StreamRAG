"""Answer validation (Phase 7): unsupported-claim policy, repair, coverage, consistency, grounding metrics."""

from streamrag.validation.consistency import ConsistencyChecker
from streamrag.validation.coverage import AnswerCoverageValidator, CoverageReport
from streamrag.validation.metrics import grounding_metrics
from streamrag.validation.policy import decide, material, needs_revision
from streamrag.validation.repair import ClaimRepairer, gap_sentence

__all__ = ["AnswerCoverageValidator", "ClaimRepairer", "ConsistencyChecker", "CoverageReport", "decide",
           "gap_sentence", "grounding_metrics", "material", "needs_revision"]
