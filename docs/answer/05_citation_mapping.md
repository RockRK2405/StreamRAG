# 05: Citation Model, Lineage and Mapping

**Code:** `citations/models.py` (`Citation`, `CitationLocation`, `CitationMap`), `citations/mapper.py` (`CitationMapper`, `ChunkCatalog`), `answer_state/render.py`
**Schema:** `docs/schemas/GroundedCitation.schema.json`

## Citation (brief §12)

| Field | Content |
|---|---|
| `citation_id` | `GA<n>-CIT<k>` (answer-scoped) |
| `claim_id` | the verified claim it supports (`AC-…`, stable across versions) |
| `evidence_id`, `chunk_id` | the supporting chunk (one chunk = one evidence item) |
| `source_id` | the source document id |
| `location` | document, section, chunk, source path, **supporting span** in the normalised document text, the chunk's own span, pages (only for paged sources) |
| `display_metadata` | citation key ("Doc_07 §2.2"), document title, section title, label shown to the generator |
| `strength` | STRONG / MODERATE (only supporting evidence is ever cited) |

**Lineage** (brief §13) is fully inspectable:

```
Answer GA3 -> claim AC-5f97e1aeae -> citation GA3-CIT2 -> evidence Doc_07§2.2#1 -> chunk Doc_07§2.2#1 -> document Doc_07
              (verification: STRONG, sentence premise [0, 89))   (index: source path, char span, section "Ladder Storage")
```

## CitationMapper (brief §14)

- **Citations come from the verification**, never from the model's labels:
  - each verified claim cites its supporting evidence, STRONG first and cited-by-the-model first on ties, at most 2 per claim;
  - a wrong label is replaced by the evidence that actually supports the claim;
  - evidence the model cited that does not support the claim is not cited.
- **Smallest useful unit:** the supporting *sentence* (or window) span inside the chunk, converted to document offsets with the chunk's span from the index. A whole document is never cited.
- **Metadata from the index, never the model** (brief §15): location and display metadata are read from the index (`ChunkCatalog`: the `CorpusChunk` records). A chunk that is not in the index cannot be cited. Pages exist only for PDF sources; for the txt/md fixtures they are `None`, never guessed. URLs or versions are not in the fixture corpus metadata and are therefore absent.

## Placement (brief §30)

Each factual sentence is followed by its *own* citation keys:

```
Ladders must be returned to the tool shed. [Doc_07 §2.2]
The sources differ: “… 40 euros” [fixture_permit_notice_2023 §1] / “… 55 euros” [fixture_permit_notice_2025 §1].
Not established: The retrieved documents do not state the value asked for in “How long does processing … take”.
```

- Conflict sides carry one citation each.
- Uncertainty sentences are never cited and never phrased as facts.
