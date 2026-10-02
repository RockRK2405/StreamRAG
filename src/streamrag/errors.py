"""Explicit, typed errors. Retrieval never silently returns wrong evidence: it raises one of these
or returns an EvidenceSet whose ``status``/``warnings`` make the degradation observable."""


class StreamRagError(Exception):
    """Base class for all project errors."""


class ConfigError(StreamRagError):
    pass


class CorpusNotFoundError(StreamRagError):
    pass


class EmptyCorpusError(StreamRagError):
    pass


class DocumentLoadError(StreamRagError):
    """A single document could not be loaded (invalid, empty, unreadable, image-only PDF, ...)."""


class CorpusIntegrityError(StreamRagError):
    """Corpus-level inconsistency, e.g. two documents claiming the same native document ID."""


class InvalidQueryError(StreamRagError):
    pass


class ModelNotAvailableError(StreamRagError):
    pass


class IndexNotFoundError(StreamRagError):
    pass


class IndexCorruptError(StreamRagError):
    pass


class IndexVersionMismatchError(StreamRagError):
    pass


class EmbeddingDimensionMismatchError(StreamRagError):
    pass


class RetrieverTimeoutError(StreamRagError):
    pass


class RerankerTimeoutError(StreamRagError):
    pass
