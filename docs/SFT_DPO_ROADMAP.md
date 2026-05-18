# WebbGPT SFT, DPO, and RAG Roadmap

This track starts after base generation is readable enough to follow instructions. The 22M local-MVP model should only be used for SFT plumbing tests, not for quality work.

## When SFT Starts

Do not run serious SFT on the weak 22M local-MVP checkpoint. Use it only to verify data formatting, loss masking, checkpoint loading, and evaluation plumbing.

Begin meaningful SFT on a 3B checkpoint when raw generations are readable and topic-stable enough that instruction tuning has something to shape. A practical gate is: raw LM samples pass the family holdout sanity checks, no severe semantic repetition, and no consistent drift from everyday/narrative prompts into unrelated medical, history, or generic expository text.

## SFT Phases

### SFT-0: Plumbing Test

Goal: prove the posttraining pipeline works.

- Use the 22M checkpoint or a 3B smoke checkpoint.
- Use 50-200 tiny examples.
- Verify chat formatting, label masking, validation split, checkpoint save/load, sample generation, and eval reporting.
- Do not treat quality as meaningful.

### SFT-1: General Assistant Behavior

Goal: teach basic instruction following without turning the model into a Webb-specific answer generator.

- Use a readable 3B base checkpoint.
- Use curated instruction examples with normal assistant behavior: concise answers, step-by-step reasoning when useful, refusal/uncertainty handling, and user-intent following.
- Keep Webb content minimal here.
- Evaluate on general assistant, chat sanity, and raw LM regression prompts to catch overfitting.

### SFT-2: RAG/Grounded Webb Answers

Goal: teach the model to answer from supplied context and abstain when context is missing.

- Use context-answer pairs built from retrieved Webb snippets.
- Include citations or source references when the serving path can expose them.
- Include missing-context and stale-context examples.
- Teach boundaries: answer from context, say when the context does not support the answer, and avoid invented facts.

### SFT-3: Webb Advising, Catalog, and Handbook Behavior

Goal: teach role-specific Webb assistant behavior after grounding behavior works.

- Catalog/course comparisons.
- Handbook/policy answers.
- Admissions, student-life, athletics, college-guidance, museum-programs, and faculty answer styles.
- Planning/advising workflows that cite the relevant source or state uncertainty.
- Separation between official policy, retrieved information, and general advice.

## When DPO Starts

Begin DPO only after SFT produces plausible answers. DPO is not a repair tool for an incoherent base or a broken SFT run.

Minimum gate:

- SFT answers are usually on topic.
- Grounded answers use the provided context.
- The model can abstain when context is insufficient.
- Preference pairs exist where the chosen answer is clearly better than the rejected answer.

DPO should start with small, high-confidence batches and explicit validation. Keep LM health eval enabled so preference optimization does not damage legibility.

## Data Needed

SFT data:

- General instruction examples for SFT-1.
- Multi-turn examples only when the turn structure is intentional and clean.
- Grounded context-answer pairs for SFT-2.
- Webb-specific advising/catalog/handbook examples for SFT-3.
- Validation examples held out by source/page/topic, not just random rows.

DPO data:

- Prompt, chosen answer, rejected answer triples.
- Pairs where rejected answers include realistic failure modes: unsupported claims, missing citations, stale facts, wrong course/policy details, overconfident answers, or verbose but unhelpful answers.
- Preference validation pairs that are never trained on.

RAG/eval data:

- Retrieval fixtures with source URL, title, timestamp, section, and text span.
- Grounded QA evals with expected evidence.
- Missing-answer evals where abstention is correct.
- Hallucination/grounding evals for catalog, handbook, athletics, admissions, student life, college guidance, faculty, mission/values, and museum programs.

## Reuse and Rebuild Guidance

There is no `junk/` directory in the current checkout. The current tree does have legacy and rerun-filtered posttrain files under `data/posttrain/`, plus deleted legacy files in git status. Treat all of these as candidates for review, not as ready production data.

Can be reused for plumbing or reviewed reuse:

- `data/local/sft.jsonl`
- `data/local/sft_validation.jsonl`
- `data/local/sft_grounding_guardrails.jsonl`
- `data/local/preference.jsonl`
- `data/local/preference_validation.jsonl`
- `data/posttrain/sft_curated_batch_*_rerun_filtered.jsonl`, after provenance, dedupe, and quality review
- `data/posttrain/preference_public_seed.jsonl` and `data/posttrain/preference_conversation_seed.jsonl`, only if chosen/rejected quality is inspectable and real

Use as holdouts/evals, not training:

- `data/posttrain/sft_curated_batch_*_validation_holdout_rerun_filtered.jsonl`
- `data/posttrain/sft_validation.jsonl`
- `data/posttrain/sft_webb_validation.jsonl`
- `data/posttrain/preference_validation.jsonl`
- `data/posttrain/preference_webb_validation.jsonl`
- Existing `data/eval/webb_*.responses` and catalog response fixtures

Rebuild before serious use:

- `data/posttrain/sft_domain_synthetic.jsonl`
- `data/posttrain/preference_domain_synthetic.jsonl`
- `data/posttrain/sft_webb_seed.jsonl`
- `data/posttrain/preference_webb_seed.jsonl`
- Deleted expanded domain corpora such as `webb_domain_large_lm_corpus.txt`, advising/catalog expanded corpora, and handbook/catalog distinction corpora

Do not use those rebuilt domain or Webb files as major base-pretraining sources. They belong in SFT, RAG evals, or small DPO preference sets after provenance is clear.

## RAG Plan

RAG can be developed before final SFT because retrieval, chunking, freshness, and citation behavior are system capabilities. Use the existing Webb seed URL packs and source policies to build retrieval fixtures, then connect SFT-2 examples to the same context format the serving path will use.

Required RAG outputs:

- Source snapshots with URL, timestamp, title, section, and content hash.
- Chunked passages with stable IDs.
- Retrieval evals for present, missing, stale, and conflicting facts.
- Answer evals that check whether claims are supported by retrieved context.

The model should not memorize Webb facts during base pretraining. It should learn to read supplied context, cite or refer to it, and decline unsupported answers.
