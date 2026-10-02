"""Assemble the streaming stack (index + Phase 3 retrieval service + controller policy) from configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from streamrag.config.settings import StreamRagConfig
from streamrag.controller import make_policy
from streamrag.retrieval.service import RetrievalService
from streamrag.retrieval.store import IndexBundle, load_index, resolve_index


@dataclass
class StreamingStack:
    cfg: StreamRagConfig
    bundle: IndexBundle
    service: RetrievalService
    policy: object
    _intent_stack: object = None

    @property
    def intent_stack(self):
        """Decomposer + query builder + fusion engine over this index (built lazily; Phase 5)."""
        if self._intent_stack is None:
            from streamrag.fusion import make_fusion_engine
            from streamrag.intents import make_decomposer
            from streamrag.intents.query_builder import IntentQueryBuilder
            from streamrag.multi_retrieval.coordinator import IntentStack
            an = self.bundle.analyzer
            self._intent_stack = IntentStack(make_decomposer(self.cfg, self.bundle),
                                             IntentQueryBuilder(lambda t: list(dict.fromkeys(an.tokens(t)))),
                                             make_fusion_engine(self.cfg, self.bundle, self.service))
        return self._intent_stack

    def with_config(self, cfg: StreamRagConfig) -> "StreamingStack":
        embed = (lambda texts: self.service.embedder.embed(texts, "query")) if self.service.embedder else None
        return StreamingStack(cfg, self.bundle, self.service, make_policy(cfg, self.bundle, embed))

    @property
    def index_hash(self) -> str:
        return self.bundle.manifest.content_hash

    def with_policy(self, strategy: str, act_classifier: str = "rules") -> "StreamingStack":
        cfg = self.cfg.model_copy(update={"controller": self.cfg.controller.model_copy(
            update={"strategy": strategy, "act_classifier": act_classifier})})
        embed = (lambda texts: self.service.embedder.embed(texts, "query")) if self.service.embedder else None
        return StreamingStack(cfg, self.bundle, self.service, make_policy(cfg, self.bundle, embed))


def build_stack(cfg: StreamRagConfig, index_path: Path | None = None) -> StreamingStack:
    bundle_path = index_path or resolve_index(cfg)
    service = RetrievalService.from_config(cfg, index_path=bundle_path, load_rerank=cfg.streaming.rerank)
    bundle = service.bundle if service.bundle.path == Path(bundle_path) else load_index(bundle_path)
    embed = (lambda texts: service.embedder.embed(texts, "query")) if service.embedder else None
    return StreamingStack(cfg, bundle, service, make_policy(cfg, bundle, embed))
