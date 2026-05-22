# WebbGPT

WebbGPT is a Python CLI and local demo stack for training a small decoder-only model, preparing posttraining data, grounding responses against local Webb-related source material, serving a local assistant, and generating project documentation visuals.

The current local development path is `local-mvp`.

## Current Final Direction

The active local-MVP path is:

1. Curated base pretraining.
2. Use the final pretrained checkpoint as the website default.
3. Add local lexical RAG for source-supported demo behavior.
4. Keep continued pretraining and SFT-RAG as explicit optional experiments that must beat the pretrained baseline before promotion.
5. Generate documentation graphics from saved artifacts for the final presentation/poster.

DPO is disabled legacy work. It is not part of the active final-demo path and should not be launched for the current project.

## Current Local-MVP

`local-mvp` now means curated-real-data base pretraining:

- Pretraining uses curated real text for general legibility.
- Current pretraining sources are `data/raw/tokenizer_corpus_local_mvp.txt` and `data/raw/tokenizer_corpus.txt`.
- No generated continuation corpora are used for base pretraining.
- No Webb/domain-heavy generated corpora are used for base pretraining.
- Domain realization is disabled for this base-pretrain recipe.
- Webb-specific behavior is deferred to SFT, RAG, and domain grounding.
- Former stale corpora now live under `data/source_material/` and must be filtered or chunked with provenance before future use.

The final local-MVP experiment before 3B planning is the lower-LR curated recipe in `sample-configs/train-local-mvp.json`.

Current checked-in local-MVP artifact state:

- Best pretrained checkpoint: `artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain`
- Prepared pretraining data: 157,442 sequences and 63,590,914 tokens from `artifacts/runs/local-mvp/prepared/pretrain.json`
- Best final-selection validation loss: 5.18789005279541
- Best final-selection perplexity: 179.09028296542547
- Pretraining stage quality status: `weak_raw_lm`
- Demo default model mode: `pretrained`
- RAG index files: `data/rag/webbgpt_chunks.jsonl`, `data/rag/webbgpt_index.json`, and `data/rag/webbgpt_sources_manifest.json`

Active local-MVP configs:

- `sample-configs/model-local-mvp.json`
- `sample-configs/tokenizer-local-mvp.json`
- `sample-configs/tokenizer-corpus-local-mvp.json`
- `sample-configs/data-local-mvp.json`
- `sample-configs/data-local-mvp-prepared.json`
- `sample-configs/train-local-mvp.json`
- `sample-configs/train-local-mvp-sanity-probe.json`

## Setup

Examples assume Python `3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train,data,serve]'
webbgpt --help
```

## Local-MVP Commands

Build the tokenizer corpus and tokenizer if they are missing:

```bash
webbgpt build-tokenizer-corpus --config sample-configs/tokenizer-corpus-local-mvp.json
webbgpt tokenize \
  --config sample-configs/tokenizer-local-mvp.json \
  --input data/raw/tokenizer_corpus_local_mvp.txt
```

Prepare curated local-MVP pretraining data:

```bash
webbgpt prepare-data \
  --config sample-configs/data-local-mvp.json \
  --stage pretrain \
  --output artifacts/runs/local-mvp/prepared/pretrain.json \
  --force-rebuild

webbgpt prepare-data \
  --config sample-configs/data-local-mvp.json \
  --stage validation \
  --output artifacts/runs/local-mvp/prepared/validation.json \
  --force-rebuild
```

Inspect the prepared corpus:

```bash
python3.12 tools/pretrain_manifest_report.py artifacts/runs/local-mvp/prepared/pretrain.json
python3.12 tools/curated_pretrain_data_audit.py
```

Run the sanity probe:

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp-prepared.json \
  --train-config sample-configs/train-local-mvp-sanity-probe.json
```

Run the final lower-LR local-MVP pretraining experiment:

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp-prepared.json \
  --train-config sample-configs/train-local-mvp.json
```

`webbgpt main --profile local-mvp` is also pretrain-only now. It builds/reuses tokenizer assets, materializes the curated pretrain and validation manifests, then runs base pretraining. It does not silently run continued pretraining, SFT, or legacy DPO.

## Documentation Graphics

The project can regenerate presentation/poster graphics from local artifacts without starting training or changing model checkpoints:

```bash
python3.12 tools/build_documentation_graphs.py
```

The generated files live under `documentation/`. The current set contains 23 requested visuals plus a generated contact sheet:

- `documentation/00_graph_index.png`
- `documentation/01_pretrain_loss_curve.png` through `documentation/18_demo_readiness_scorecard.png`
- `documentation/19_training_loss_animation.gif` through `documentation/23_data_filtering_funnel_animation.gif`
- `documentation/visual_manifest.json`
- `documentation/README.md`

The generator uses saved pretraining history, prepared-data manifests, RAG chunks/indexes, saved RAG regression outputs, serving verification, static UI/server evidence, and deterministic heuristics where direct measurements are unavailable. It does not make network calls, start training, or modify checkpoints.

## Posttraining And Grounding

The current 22M local-MVP base is still a raw LM experiment. Continued pretraining and SFT are optional improvement paths and should be evaluated against the pretrained checkpoint before changing the website default.

DPO was tested as a small plumbing/stretch experiment. It moved tiny preference metrics but worsened real samples, so `src/cli.py train-dpo` is retained only as a disabled legacy command.

### Continued Pretraining

Small continued pretraining now reads reviewed source material from `data/source_material/continued_pretrain_candidates/`. The first local run is intentionally bounded and uses a small token budget because the clean source set is smaller than the original 500K-2M target.

Audit and prepare the continued-pretraining data:

```bash
python3.12 src/cli.py audit-data \
  --config sample-configs/data-local-mvp-continue-small.json \
  --stage continue

python3.12 src/cli.py prepare-data \
  --config sample-configs/data-local-mvp-continue-small.json \
  --stage continue \
  --output artifacts/runs/local-mvp-continue-small/prepared/pretrain.json \
  --force-rebuild

python3.12 src/cli.py prepare-data \
  --config sample-configs/data-local-mvp-continue-small.json \
  --stage validation \
  --output artifacts/runs/local-mvp-continue-small/prepared/validation.json \
  --force-rebuild
```

The training command is available, but should not be run until explicitly approved:

```bash
PYTHONPATH=src python3.12 src/cli.py train-continue \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp-continue-small-prepared.json \
  --train-config sample-configs/train-local-mvp-continue-small.json
```

### RAG Source And Index

RAG is the safer path for stale Webb-like source material because source text stays outside model weights and is returned with chunk metadata.

RAG is experimental. Retrieval can surface useful passages, but the 22M local-MVP model can still fail to synthesize a reliable answer. In the WebbGPT 0.2 demo, local-MVP generation remains visible even when the quality checker flags weak output. Retrieved sources support the generated answer; source cards are not proof that the generated answer is correct.

The current RAG source expansion adds low-risk curated explanatory files for prerequisites, recommendations, catalog purpose, course descriptions, handbook/catalog distinction, and boarding school community. These files live in `data/source_material/rag_candidates/`, are marked `Allowed use: RAG`, and avoid staff names, current-year deadlines, phone policies, dorm rules, admissions deadlines, dining rules, and other current factual claims.

Build and query the local lexical RAG store:

```bash
python3.12 tools/build_rag_corpus.py
python3.12 tools/build_rag_index.py
python3.12 tools/query_rag_index.py "school community" --top-k 3
python3.12 tools/query_rag_index.py "What does a course catalog help students understand?" --top-k 3
```

The generated files are:

- `data/rag/webbgpt_chunks.jsonl`
- `data/rag/webbgpt_index.json`
- `data/rag/webbgpt_sources_manifest.json`

After the source expansion, the local RAG corpus has 166 chunks. Direct retrieval checks now return safe curated sources for:

- `What does the catalog say about prerequisites?`
- `What is the difference between a prerequisite and a recommendation?`
- `What does a course catalog help students understand?`
- `What is a course description used for?`
- `What is the difference between handbook language and catalog language?`

The same checks return no-hit for `Who is the dean?`, `What is the phone policy in the dining hall?`, and `hi im dr dzula` unless a safe source is later added for those prompts.

Run the fixed reliability prompt set against a running RAG server:

```bash
python3.12 tools/run_rag_reliability_regression.py \
  --output artifacts/runs/local-mvp/rag_reliability_regression.json
```

The source-expansion regression artifact is:

`artifacts/runs/local-mvp/rag_reliability_regression_after_source_expansion.json`

Current result: prerequisite/catalog prompts retrieve relevant chunks, then WebbGPT 0.2 still asks the local-MVP checkpoint to generate. If the quality checker flags the output, the UI shows `Weak generation` while still displaying the generated text and keeping sources collapsed below the message.

### SFT With RAG Context

SFT-RAG rows teach short context-grounded answers. They do not copy stale corpora directly into assistant targets and they do not use fake citation labels.

Build the first SFT-RAG data set:

```bash
python3.12 tools/build_sft_rag_data.py
```

The training command is available, but should not be run until explicitly approved:

```bash
PYTHONPATH=src python3.12 src/cli.py train-sft \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp-sft-rag-v1.json \
  --train-config sample-configs/train-local-mvp-sft-rag-v1.json \
  --force-rebuild
```

Compare checkpoints before changing the demo default:

```bash
PYTHONPATH=src python3.12 tools/compare_checkpoints.py \
  --rag-index data/rag/webbgpt_index.json \
  --rag-chunks data/rag/webbgpt_chunks.jsonl \
  --output artifacts/runs/local-mvp-continue-small/comparison.json
```

Grounding work can still use the checked-in Webb fixtures:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls_demo.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url data/webb/mock/handbook.txt
```

## Local Demo Server

### Current Demo Default

The current demo default is the final 22M pretrained local-MVP checkpoint:

`artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain`

SFT-small, SFT-v2, and SFT-v3 were run as local posttraining/plumbing experiments, but they are not the website default. SFT-v3 completed and reduced validation loss, but the final demo prompt comparison stayed mixed/worse than pretrained: prompt retention and grounded-context behavior did not improve enough to justify switching. DPO is disabled legacy work and is not an active candidate. Keep `WEBBGPT_MODEL_MODE=pretrained` unless a later checkpoint clearly beats pretrained on behavior samples.

Start the local-MVP demo server with the pretrained 22M checkpoint:

```bash
PYTHONPATH=src \
WEBBGPT_CHECKPOINT=artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain \
WEBBGPT_MODEL_MODE=pretrained \
python3.12 src/cli.py serve \
  --serve-config sample-configs/serve-local-mvp.json \
  --force-untrusted
```

Open the browser demo at `http://127.0.0.1:8000/`. The root page is a ChatGPT-style WebbGPT 0.2 chat interface with a centered transcript, bottom composer, example prompt chips, compact model/RAG badges, a collapsible settings panel, collapsed source cards under RAG-supported responses, and a collapsible run-details panel for metadata.

The canonical generation API is `/v1/chat/completions`; `/generate` is a small demo compatibility alias for prompt-only calls. `/generate_stream` is the browser UI streaming route.

Check server status:

```bash
curl http://127.0.0.1:8000/status
```

Test canonical generation:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"At Webb, students often..."}],"tools":false,"citations":false,"max_new_tokens":32,"temperature":0.4,"top_k":40,"top_p":0.95}'
```

Test the prompt-only demo alias:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"At Webb, students often","tools":false,"citations":false,"max_new_tokens":40,"temperature":0.7,"top_k":40,"top_p":0.95}'
```

Test the streaming demo route:

```bash
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the difference between a prerequisite and a recommendation?","tools":true,"citations":true,"max_new_tokens":32,"temperature":0.3,"top_k":30,"top_p":0.92}'
```

`/generate_stream` returns Server-Sent Events with `start`, `delta`, `metadata`, `done`, and `error` events. This is UI-level progressive rendering: the native backend still produces the response normally, then the server reveals the final text in word-like chunks while preserving the same quality-gate, abstention, source, and provenance metadata.

Start the same pretrained demo with local RAG enabled:

```bash
PYTHONPATH=src \
WEBBGPT_CHECKPOINT=artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain \
WEBBGPT_MODEL_MODE=pretrained \
WEBBGPT_USE_RAG=1 \
WEBBGPT_RAG_INDEX=data/rag/webbgpt_index.json \
WEBBGPT_RAG_CHUNKS=data/rag/webbgpt_chunks.jsonl \
python3.12 src/cli.py serve \
  --serve-config sample-configs/serve-local-mvp.json \
  --force-untrusted
```

When RAG is enabled, the server retrieves local chunks, feeds relevant source context into the local-MVP prompt, and returns retrieved chunk metadata in response metadata. If RAG is off, the server does not retrieve sources and uses raw local-MVP generation only. If retrieval is weak or absent in the main chat demo, the model still generates unless a legacy grounded-only route explicitly requires deterministic abstention. If generation is punctuation-only, token-garbage, or otherwise low confidence, the response is labeled `Weak generation`; the generated text is still shown because WebbGPT 0.2 is a research demo of a small local model.

The UI shows these states per assistant message:

- `Generated`
- `Generated with sources`
- `Weak generation`
- `Abstained`
- `Generation failed`

RAG source cards show chunk ID, source file, retrieval score, risk level, allowed use, and a short preview. Source cards are collapsed by default under `Sources available (n)`. Raw metadata is hidden under `Run details`.

Current reliability settings in `sample-configs/serve-local-mvp.json`:

- Decode changed from `max_new_tokens=96`, `temperature=0.4`, `top_k=40`, `top_p=0.95`, `repetition_penalty=1.08` to `max_new_tokens=48`, `temperature=0.3`, `top_k=30`, `top_p=0.92`, `repetition_penalty=1.15`.
- RAG retrieval requires `rag_min_score=0.05`, `rag_min_lexical_overlap=0.45`, `rag_min_matched_terms=2`, and named query terms must appear in retrieved chunks.
- RAG source expansion improves prerequisite/catalog retrieval coverage, but generated answers still fail often enough that `Weak generation` should be expected on some grounded prompts.
- These settings make the demo safer; they do not make the 22M checkpoint a polished assistant.

## Remote Profiles

`remote-3b` and `remote-7b` are starter profiles for future scale planning. They are not the current source of truth for local-MVP quality work. Review their data mix and training schedule before treating either as production-ready.

## 3B/7B Scale Launch Prep

3B/7B pretraining configs are 8-GPU recommended, and the standard configs assume 8 distributed ranks. Before launching scale pretraining, use the dry-run helper to inspect visible CUDA GPUs and print the correct command:

```bash
python3.12 tools/prepare_scale_launch.py \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b-curated-smoke-100m.json \
  --train-config sample-configs/train-3b-smoke-100m.json \
  --run-dir artifacts/runs/scale-3b-smoke-100m \
  --recommended-gpus 8 \
  --desired-global-batch-size 64 \
  --micro-batch-size 1 \
  --allow-auto-config
```

If exactly 8 CUDA GPUs are visible, the helper uses the standard train config as-is. If non-8 GPU hardware is visible, it writes a run-specific config under `artifacts/runs/...` and adjusts `gradient_accumulation_steps` to preserve effective global batch size where possible. The helper prints the command; it does not start training.

## Tests

```bash
webbgpt test
webbgpt test --all
```

Focused pytest runs are also supported:

```bash
python3.12 -m pytest src/tests/test_cli_profiles.py
```

## Repo Layout

- `sample-configs/`: active starter configs
- `data/raw/`: active real-text tokenizer/pretraining corpora
- `data/source_material/`: reviewed buckets for future continued pretraining and RAG source material
- `data/eval/`: active eval fixtures and historical evals
- `data/webb/`: Webb grounding fixtures and source packs
- `src/`: CLI, model, training, eval, grounding, and serving code
- `tools/`: active reporting and audit tools
- `documentation/`: generated project graphics, README, and visual manifest
- `artifacts/`: generated tokenizers, manifests, checkpoints, reports, and exports
