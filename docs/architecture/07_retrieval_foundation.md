# 07: Retrieval Foundation (as implemented in Phase 3)

This diagram matches `src/streamrag/corpus` and `src/streamrag/retrieval` exactly. Module names are shown in each node.

```mermaid
flowchart TD
  subgraph BUILD["Index build: streamrag build-index"]
    C[("Corpus directory<br/>paths.corpus")] --> SRC["CorpusSource<br/>scan, corpus_hash, fixture marker"]
    SRC --> LD["Loader<br/>txt / md / pdf"]
    LD --> NM["Normalizer<br/>unicode, headers/footers, dehyphenate, page offsets"]
    NM --> SP["SectionParser<br/>headings, section ids, hierarchy"]
    SP --> CH["Chunker<br/>paragraph 180/300, section-bounded"]
    CH --> MF[("Manifest + chunks.jsonl + documents.jsonl")]
    MF --> BI["BM25Index<br/>scipy CSC weights"]
    MF --> DI["DenseIndex<br/>bge-small ONNX, exact float32"]
  end

  subgraph QUERY["Query: RetrievalService.retrieve()"]
    Q["Query + RetrievalOptions"] --> BQ["BM25 search<br/>top lexical_k"]
    Q --> EQ["Query embedding<br/>ONNX, timeout"]
    EQ --> DQ["Dense search<br/>top dense_k"]
    BQ --> UN["Candidate union by chunk"]
    DQ --> UN
    UN --> RRF["RRF fusion, k_rrf"]
    RRF --> DD["Deduplication<br/>exact, overlap, guarded near-dup"]
    DD --> RK{"rerank enabled?"}
    RK -- yes --> CE["Cross-encoder<br/>top rerank_k, timeout"]
    RK -- no --> TK["Top-k"]
    CE --> TK
    TK --> ES[("EvidenceSet<br/>Evidence items + RetrievalTrace")]
  end

  BI -.-> BQ
  DI -.-> DQ
  DQ -. dense failure: lexical-only, status degraded .-> UN
```

## Notes

- **BM25-only or dense-only modes** skip the other branch. The fused list is then that single ranked list, and evidence is labeled `bm25` or `dense`.
- **Isolation:** the query path reads only the loaded index. There is no network import in `corpus/` or `retrieval/` (tested), and no document-injection API.
- **Integrity:** loading an index verifies the version, every artifact's sha256, chunk counts and embedding dimensions before any query runs.
