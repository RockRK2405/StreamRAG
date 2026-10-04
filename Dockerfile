# StreamRAG - Samsung PRISM Theme 4 (Streaming Live RAG). One image: the frozen final pipeline + live demo UI.
#   docker compose up --build          -> http://localhost:8080  (verified extractive answers, no LLM, offline)
#   docker compose --profile llm up    -> + local Ollama sidecar (pull the model once, see README)
# Model weights are baked in at build time (spec REQ-DEP-003): copied from ./models when present, else downloaded
# from the pinned registry (configs/models.yaml, sha256-checked). No secrets are needed or accepted.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMRAG_HOST=0.0.0.0 STREAMRAG_PORT=8080 STREAMRAG_ROOT=/app
WORKDIR /app
RUN useradd --create-home --uid 10001 streamrag

COPY --chown=streamrag requirements.lock pyproject.toml README.md ./
COPY --chown=streamrag src ./src
RUN pip install -r requirements.lock && pip install --no-deps -e .

COPY --chown=streamrag configs ./configs
COPY --chown=streamrag demo ./demo
# ./models is git-ignored. "model[s]" copies a local models/ directory when it exists (fast, offline build; the
# always-present requirements.lock keeps the COPY valid when it does not) - a fresh clone downloads the two models the
# final pipeline uses instead. Then the demo index is built once at build time.
COPY --chown=streamrag requirements.lock model[s] ./models/
RUN rm -f models/requirements.lock \
    && for m in bge-small-en-v1.5 nli-deberta-v3-xsmall; do \
         [ -f "models/$m/MODEL_INFO.json" ] || streamrag fetch-models "$m"; done \
    && streamrag --set paths.corpus=demo/corpus build-index \
    && find models indexes -user root -exec chown streamrag {} +

USER streamrag
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=4).status == 200 else 1)"
CMD ["streamrag", "serve"]
