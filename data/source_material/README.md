# Source Material

These files were moved from `stale-corpora/` for future continued pretraining and RAG work. They are not SFT data. Do not copy them directly into SFT; chunk them, attach provenance, and create grounded examples only after manual review.

| Original path | Current path | Approx words | Reason | Risk | Allowed use | Template repetition | Current factual claims | Chunk with provenance |
|---|---|---:|---|---|---|---|---|---|
| `stale-corpora/catalog_expanded_corpus.txt` | `data/source_material/continued_pretrain_candidates/catalog_expanded_corpus.txt` | 244,113 | Catalog-style text with useful academic/course-planning language if filtered. | Medium | Continued pretraining candidate; RAG candidate after chunking. | Some | Yes | Yes |
| `stale-corpora/handbook_catalog_distinction_corpus.txt` | `data/source_material/continued_pretrain_candidates/handbook_catalog_distinction_corpus.txt` | 70,601 | Focused distinction material useful for simple grounded examples and small continued pretraining. | Low | Continued pretraining candidate; RAG candidate after chunking. | Low | Some | Yes |
| `stale-corpora/webb_school_context_large_lm_corpus.txt` | `data/source_material/rag_candidates/webb_school_context_large_lm_corpus.txt` | 36,072 | School-context material is better retrieved as context than learned as model weights. | Medium | RAG only after chunking and provenance labels. | Some | Yes | Yes |
| `stale-corpora/webb_domain_large_lm_corpus.txt` | `data/source_material/rejected_template_heavy/webb_domain_large_lm_corpus.txt` | 5,000,096 | Very large, repetitive domain text risks overpowering the small model. | High | Rejected for direct use. | High | Yes | No, except small manually reviewed excerpts. |
| `stale-corpora/webb_catalog_large_lm_corpus.txt` | `data/source_material/rejected_template_heavy/webb_catalog_large_lm_corpus.txt` | 2,998,938 | Repetitive catalog expansions risk hallucinated catalog/course phrasing. | High | Rejected for direct use. | High | Yes | No, except small manually reviewed excerpts. |
| `stale-corpora/webb_advising_large_lm_corpus.txt` | `data/source_material/rejected_template_heavy/webb_advising_large_lm_corpus.txt` | 2,125,038 | Repetitive advising text risks generic advice loops. | High | Rejected for direct use. | High | Yes | No, except small manually reviewed excerpts. |
| `stale-corpora/local_mvp_v2_advising_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_advising_continuations.txt` | 27,723 | Generated continuation set; not reliable source material. | High | Rejected. | High | Some | No |
| `stale-corpora/local_mvp_v2_catalog_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_catalog_continuations.txt` | 27,959 | Generated catalog continuations risk fake catalog style. | High | Rejected. | High | Some | No |
| `stale-corpora/local_mvp_v2_everyday_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_everyday_continuations.txt` | 9,501 | Generated continuations are not source-grounded. | Medium | Rejected. | High | Low | No |
| `stale-corpora/local_mvp_v2_general_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_general_continuations.txt` | 8,037 | Generated continuations are not source-grounded. | Medium | Rejected. | High | Low | No |
| `stale-corpora/local_mvp_v2_narrative_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_narrative_continuations.txt` | 8,760 | Generated continuations are not source-grounded. | Medium | Rejected. | High | Low | No |
| `stale-corpora/local_mvp_v2_school_academic_continuations.txt` | `data/source_material/rejected_template_heavy/local_mvp_v2_school_academic_continuations.txt` | 3,014 | Generated school continuations risk domain overconfidence. | High | Rejected. | High | Some | No |
| `stale-corpora/advising_expanded_corpus.txt` | `data/source_material/needs_manual_review/advising_expanded_corpus.txt` | 207,934 | Advising material may be useful but can contain policy/current-fact risk. | High | Manual review only. | Some | Yes | Yes, after review |
| `stale-corpora/webb_handbook_large_lm_corpus.txt` | `data/source_material/needs_manual_review/webb_handbook_large_lm_corpus.txt` | 125,177 | Handbook-like material can contain stale policies and current names/details. | High | Manual review only. | Some | Yes | Yes, after review |

## Curated RAG Safe Files

These short files were created as low-risk explanatory RAG sources after reviewing the stale source-material themes. They are not current policy sources and they are not SFT answer dumps.

| Original path | Current path | Approx words | Reason | Risk | Allowed use | Template repetition | Current factual claims | Chunk with provenance |
|---|---|---:|---|---|---|---|---|---|
| Curated from reviewed catalog prerequisite themes | `data/source_material/rag_candidates/catalog_prerequisites_safe.txt` | 147 | Gives direct stable language for prerequisites and course planning. | Low | RAG | No | No | Yes |
| Curated from reviewed catalog recommendation themes | `data/source_material/rag_candidates/catalog_recommendations_safe.txt` | 141 | Distinguishes recommendations from prerequisites without inventing course facts. | Low | RAG | No | No | Yes |
| Curated from reviewed catalog purpose themes | `data/source_material/rag_candidates/catalog_purpose_safe.txt` | 151 | Explains what a course catalog helps students understand. | Low | RAG | No | No | Yes |
| Curated from reviewed course-description themes | `data/source_material/rag_candidates/course_descriptions_safe.txt` | 158 | Explains what course descriptions can and cannot prove. | Low | RAG | No | No | Yes |
| Curated from reviewed handbook/catalog distinction themes | `data/source_material/rag_candidates/handbook_catalog_distinction_safe.txt` | 153 | Separates handbook policy language from catalog/course language. | Low | RAG | No | No | Yes |
| Curated from reviewed school-community themes | `data/source_material/rag_candidates/boarding_school_community_safe.txt` | 153 | Gives a general, non-current-factual passage for boarding school community questions. | Low | RAG | No | No | Yes |

## Use Rules

- Continued pretraining candidates need filtering for repetition, source balance, and stale factual claims before training.
- RAG candidates need chunk IDs, source labels, and dates before retrieval use.
- Manual-review files should not enter any model-facing pipeline until current-factual claims are checked.
- Rejected template-heavy files should stay out of SFT and continued pretraining.
- Curated RAG safe files may be rebuilt into `data/rag/webbgpt_chunks.jsonl`; their provenance headers are stored in `data/rag/webbgpt_sources_manifest.json`, while retrieved chunk text omits header boilerplate.
