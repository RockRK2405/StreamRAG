"""Data contracts (spec §5, §6, §9, §11, §17, §22). Pydantic v2; strict; deterministic JSON."""

from streamrag.models.answers import AnswerDiff, AnswerVersion, Citation, Claim, ClaimChange, UncertaintyItem
from streamrag.models.base import SCHEMA_VERSION, Contract, canonical_dumps
from streamrag.models.benchmark import BenchmarkCase, MultiIntentBenchmarkCase, RetrievalEvalItem
from streamrag.models.corpus import (
    CorpusChunk,
    CorpusDocument,
    CorpusManifest,
    CorpusSection,
    EmbeddingInfo,
    Paragraph,
    SourceFile,
)
from streamrag.models.events import (
    EventType,
    InputEvent,
    SessionEnd,
    SessionStart,
    TelemetryEvent,
    TranscriptChunk,
    UtteranceEnd,
)
from streamrag.models.evidence import Evidence, EvidenceSet, RetrievalTrace
from streamrag.models.intents import (
    Constraint,
    InheritedContext,
    Intent,
    IntentProvenance,
    IntentQuery,
    IntentRelationship,
    IntentSet,
    IntentSetDelta,
    SourceSpan,
)
from streamrag.models.retrieval import (
    RetrievalFilters,
    RetrievalHit,
    RetrievalOptions,
    RetrievalRequest,
    RetrievalResult,
)

# Models exported as JSON Schemas to docs/schemas/ (see streamrag.models.schemas).
SCHEMA_MODELS = {
    "SessionStart": SessionStart,
    "TranscriptChunk": TranscriptChunk,
    "UtteranceEnd": UtteranceEnd,
    "SessionEnd": SessionEnd,
    "TelemetryEvent": TelemetryEvent,
    "Intent": Intent,
    "IntentSet": IntentSet,
    "IntentSetDelta": IntentSetDelta,
    "IntentQuery": IntentQuery,
    "Evidence": Evidence,
    "EvidenceSet": EvidenceSet,
    "Claim": Claim,
    "Citation": Citation,
    "AnswerVersion": AnswerVersion,
    "RetrievalRequest": RetrievalRequest,
    "RetrievalResult": RetrievalResult,
    "BenchmarkCase": BenchmarkCase,
    "MultiIntentBenchmarkCase": MultiIntentBenchmarkCase,
    "RetrievalEvalItem": RetrievalEvalItem,
    "CorpusChunk": CorpusChunk,
    "CorpusManifest": CorpusManifest,
}


def _phase4_models() -> dict:
    # imported lazily to avoid a circular import (controller/ledger/fusion depend on streamrag.models)
    from streamrag.controller.models import RetrievalDecision
    from streamrag.fusion.models import UnifiedEvidenceSet
    from streamrag.ledger.models import QueryRecord
    from streamrag.context.models import ContextChange, TopicFrame
    from streamrag.delta.models import DeltaPlan, EvidenceAssignment
    from streamrag.session.models import SessionArchive, SessionState, SessionStateVersion
    from streamrag.answer_state.models import GroundedAnswer
    from streamrag.citations.models import Citation as GroundedCitation
    from streamrag.claims.models import ClaimPlan, ClaimVerification
    return {"RetrievalDecision": RetrievalDecision, "QueryRecord": QueryRecord, "UnifiedEvidenceSet": UnifiedEvidenceSet,
            # Phase 6
            "SessionState": SessionState, "SessionStateVersion": SessionStateVersion, "SessionArchive": SessionArchive,
            "ContextChange": ContextChange, "TopicFrame": TopicFrame, "DeltaPlan": DeltaPlan,
            "EvidenceAssignment": EvidenceAssignment,
            # Phase 7
            "GroundedAnswer": GroundedAnswer, "GroundedCitation": GroundedCitation, "ClaimPlan": ClaimPlan,
            "ClaimVerification": ClaimVerification}

__all__ = [n for n in dir() if not n.startswith("_")]
