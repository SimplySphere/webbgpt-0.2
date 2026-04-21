# WebbGPT

WebbGPT is an end-to-end repository for building a real decoder-only language model from scratch, aligning it into a helpful assistant, grounding it on a multi-domain Webb service stack, evaluating it honestly, exporting it for inference, and serving it through a local API and browser UI.

This repo is not only about getting a model to answer. It is also about keeping the important layers separate and inspectable:

- what the raw model can do
- what the retrieval and grounding pipeline can do
- what the benchmark suite is actually measuring
- whether an exported runtime still behaves like the internal runtime

The current local-MVP lane is intentionally opinionated about trustworthiness. It prefers explicit post-train validation, provenance-rich eval output, deterministic missing-data abstention, shared decode presets, and benchmark identity tracking over flattering but weak scores.

The serve UI follows the same idea: answer first, but keep grounding, routing, provenance, and reproduction details close at hand.

## What It Is

WebbGPT is a full stack for owning the model, the retrieval layer, the evaluation surface, and the user-facing product shell around them.

- The base-model path is a custom decoder-only transformer with grouped-query attention, RoPE, RMSNorm, SwiGLU, KV-cache generation, repetition controls, and export support.
- The data and training path covers tokenizer freezing, raw-corpus collection, prepared-data materialization, pretraining, continued pretraining, supervised fine-tuning, and DPO preference optimization.
- The evaluation path separates raw-model quality from grounded-pipeline quality, tracks provenance and release gates, and reports attribution lanes, retrieval audit, route audit, reliability warnings, and grounding quality.
- The grounding path uses a Webb knowledge stack with family-aware snapshots, partial refresh, carry-forward behavior, handbook parsing, hybrid lexical/fuzzy retrieval, source classification, citation support, and snapshot diffing.
- The serving path exposes the assistant through FastAPI plus an answer-first browser UI with grounded/cited status, malformed-output handling, snapshot-aware startup behavior, and shared decode assumptions.
- The repo is structured for both small local experiments and larger Linux multi-GPU runs, and it can be used with either live Webb sources or the structured offline fixtures in `data/webb/mock/`.

The current supported lanes are:

- `debug`: fastest smoke-test lane; proves plumbing, not quality
- `local-mvp`: best local experimental lane for a Mac-class machine
- `remote-3b`: intermediate serious lane
- `remote-7b`: intended full serious lane

`local-mvp` is the most practical lane for local iteration, but it should still be treated as an experimental recovery lane, not as a finished strong assistant.

You can use WebbGPT as:

- a from-scratch model-building repo
- a post-train and evaluation harness for trustworthiness work
- a snapshot-backed Webb grounding and retrieval system
- an offline fixture lab for ingest, routing, diffing, and regression testing
- a local API and browser product shell for manual demos and product iteration

## Intended workflow

1. Generate starter configs and checked-in seed data.
2. Build a tokenizer corpus and freeze the tokenizer for the chosen lane.
3. Prepare sharded manifests if you want explicit reusable data artifacts.
4. Run base pretraining.
5. Run continued pretraining on LM-safe domain text.
6. Run SFT with explicit held-out validation.
7. Run DPO with explicit held-out validation plus LM-health tracking.
8. Evaluate with benchmark provenance, attribution lanes, release gates, and retrieval audit.
9. Export the best checkpoint, ingest grounded sources or sync a Webb snapshot, and serve the model with the same decode, tokenizer, and grounding assumptions used by evaluation.

That is the canonical end-to-end flow, but the repo is intentionally modular. If you already have a model export, or if you are only iterating on grounding, benchmarks, or serving, you can enter at the later stages without running the full training stack.

## Quick Start

### One-Time Setup

Create an environment, install the project, confirm the CLI is available, and write the maintained starter config pack:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train,data,serve]'
webbgpt --help
webbgpt init-config --output-dir sample-configs
```

If this venv was created before the current dependency pins, refresh the core compatibility packages once:

```bash
pip install --upgrade --force-reinstall 'numpy<2' 'transformers>=4.43,<5'
```

The full extras install above is the simplest default. If you only need one slice of the repo, the extras are also modular:

- `.[dev,data,serve]` for grounding, benchmark, and serving work
- `.[dev,train,data]` for tokenizer, data prep, training, and evaluation work

### Choose A Lane

- `debug`: fastest smoke test; validates the pipeline, not model quality
- `local-mvp`: strongest repo-supported local lane for a Mac-class machine
- `remote-3b`: intermediate serious lane if you explicitly want a 3B-scale path
- `remote-7b`: intended full lane for a strong Linux multi-GPU machine

### Automated Debug Smoke Test

Use this when you want the quickest end-to-end plumbing check. It is meant to prove that tokenizer creation, data prep, training, export, grounding, and serving all connect correctly.

```bash
webbgpt main --profile debug
```

### Automated Local-MVP

Use this when you want the strongest realistic local lane instead of the toy smoke test. This profile uses a much smaller model, shorter context, explicit post-train validation, stricter evaluation, deterministic no-hit abstention in serve-time grounding, and more conservative decode defaults.

```bash
webbgpt main --profile local-mvp
```

If you want to stop after training, Webb-grounded evaluation, and export so you can serve later instead, run:

```bash
webbgpt main --profile local-mvp --no-serve

webbgpt serve --serve-config sample-configs/serve-local-mvp.json
```

### Manual Webb Sync

You usually do not need this when you run `webbgpt main --profile ...`, because the profile-driven path already uses Webb grounding, and `serve` can resync with `--sync-on-start`. Use these commands when you want to inspect or refresh the Webb knowledge store yourself.

Sync the live Webb sources:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

If you have private or Webb-authorized local documents that are not on the public site, you can layer them on top of the live sync with an offline overlay pack. The live pack is still fetched, and the local/private pack is ingested in the same snapshot as an additive overlay:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --seed-url-pack data/webb/seed_urls.json \
  --offline-seed-url-pack data/webb/seed_urls_private.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

Sync the offline demo source pack instead when you want to exercise ingest, routing, retrieval, snapshot diffing, or serve behavior without depending on live Webb availability:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-demo.db \
  --seed-url-pack data/webb/seed_urls_demo.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url data/webb/mock/handbook.txt
```

The demo pack uses the structured fixture set under `data/webb/mock/`, which is the fastest way to work on the Webb grounding surface offline.

Evaluate a checkpoint manually against the Webb benchmark suite:

```bash
webbgpt eval \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --eval-config sample-configs/eval-debug.json \
  --checkpoint artifacts/runs/debug/checkpoints/dpo/step-00000020
```

Serve the debug lane and force a fresh sync immediately before boot:

```bash
webbgpt serve --serve-config sample-configs/serve-debug.json --sync-on-start
```

The grounded Webb service surface now covers:

- course catalog and year diffs
- handbook and policy
- faculty
- admissions, mission/values, and college guidance
- student-life and culture/community
- museum and unique programs
- athletics
- planner/advice 

If you only want to refresh part of the Webb knowledge store, the sync path is now family-aware:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890' \
  --families athletics faculty
```

Unselected families are carried forward from the newest completed trusted snapshot instead of being discarded.

The same manual pattern works for `local-mvp`, `remote-3b`, and `remote-7b`; just swap the `*-debug.json` files for the matching lane configs.

### Reading Eval Output

When you run `webbgpt eval`, read the output in this order:

1. `validation`: language-model health on held-out validation text. If this is still weak, post-training and retrieval can only mask so much.
2. `assistant_summary`: general assistant quality on non-grounded prompts.
3. `assistant_benchmarks`, `catalog_benchmarks`, or `webb_benchmarks`: grounded quality by benchmark family.
4. `webb_domain_summary`: per-domain Webb quality for course, handbook, faculty, admissions/general info, student life, mission/values, college guidance, museum, athletics, and any mixed multi-domain slices included in the active suite.
5. `attribution_lanes`: separation between `model_only`, `pipeline_grounded`, and `retrieval_oracle` behavior so the pipeline does not get mistaken for raw model quality.
6. `retrieval_audit` and `route_audit`: whether a grounded miss came from missing data, retrieval failure, model behavior after no-hit, routing failure, or low-confidence route fan-out that still failed to find support.
7. `grounding_quality`: whether the active grounded snapshot is citable, especially for handbook extraction quality, and whether the snapshot was completed cleanly.
8. `provenance`: exact checkpoint, tokenizer, decode preset, seed bundle, benchmark suite hash, scorer hash, release-gate hash, the active `catalog_snapshot` or `grounding_snapshot`, and family metadata when the Webb stack is active.
9. `release_gates`: final go or no-go summary, only meaningful after reading the items above.

Compare two eval runs only when the benchmark suite, scorer version, release-gate config, and grounding snapshot all match.

### Automated Remote-3B Intermediate

Use this only when you explicitly want the intermediate serious 3B lane. It is no longer the recommended local default. The `--mvp` preset shortens the serious run, while `--full` keeps the longer 3B budget.

```bash
webbgpt main --profile remote-3b --no-serve --mvp
```

Swap `--mvp` for `--full` when you want the longer 3B run.

After it finishes, serve the exported 3B model with:

```bash
webbgpt serve --serve-config sample-configs/serve-3b.json
```

### Automated Remote-7B Full

Use this when you are on the intended powerful Linux multi-GPU machine and want the canonical full version instead of the intermediate 3B path.

```bash
webbgpt main --profile remote-7b --no-serve
```

After it finishes, serve the exported 7B model with:

```bash
webbgpt serve --serve-config sample-configs/serve-7b.json
```

### Manual Paths

Use the sections below when you want to drive the pipeline stage by stage yourself instead of relying on `webbgpt main`.

### Manual Webb Grounding Path

Use this path when you want to exercise the Webb grounded service stack directly. This is the same grounding interface the standard profile configs now use by default.

Sync the full Webb source pack defined by a seed URL list and handbook source:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

You can also drive the ingest legs separately or restrict site refreshes to specific families:

```bash
webbgpt ingest-webb-site \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --families student_life athletics faculty

webbgpt ingest-webb-handbook \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890' \
  --allow-ocr-fallback
```

Diff two saved Webb snapshots:

```bash
webbgpt diff-webb-snapshot \
  --dsn sqlite:///artifacts/grounding/webbgpt-webb.db \
  --from-snapshot snapshot-a \
  --to-snapshot snapshot-b
```

Serve using the Webb grounding config and force a sync immediately before boot if needed:

```bash
webbgpt serve \
  --serve-config sample-configs/serve-debug.json \
  --sync-on-start
```

When `sync_on_start` fails, serve now falls back to the newest completed trusted Webb snapshot if one exists. If no completed snapshot exists, it fails closed instead of booting against partial data.

### Shared Manual Tokenizer Setup

Build a tokenizer corpus and freeze the tokenizer manually before running individual training stages. Pick the tokenizer config pair that matches the lane you want.

```bash
webbgpt build-tokenizer-corpus \
  --config sample-configs/tokenizer-corpus.json

webbgpt tokenize \
  --config sample-configs/tokenizer.json \
  --input data/raw/tokenizer_corpus.txt
```

For the local-MVP or 7B lanes, use the matching tokenizer config instead:

```bash
webbgpt build-tokenizer-corpus \
  --config sample-configs/tokenizer-corpus-local-mvp.json

webbgpt tokenize \
  --config sample-configs/tokenizer-local-mvp.json \
  --input data/raw/tokenizer_corpus_local_mvp.txt
```

```bash
webbgpt build-tokenizer-corpus \
  --config sample-configs/tokenizer-corpus-7b.json

webbgpt tokenize \
  --config sample-configs/tokenizer-7b.json \
  --input data/raw/tokenizer_corpus_7b.txt
```

### Shared Manual Prepared Data

Materialize prepared artifacts for specific stages when you want to inspect them, rerun only part of the pipeline, or prepare data separately from training.

```bash
webbgpt prepare-data \
  --config sample-configs/data-debug.json \
  --stage pretrain \
  --output artifacts/manifests/pretrain.json

webbgpt prepare-data \
  --config sample-configs/data-3b.json \
  --stage validation \
  --output artifacts/manifests/validation.json
```

`webbgpt prepare-data` auto-reuses completed matching manifests and auto-resumes interrupted matching runs when resumable state exists. Use `--force-rebuild` to discard prior prepared outputs and rebuild safely from scratch.

### Manual Debug Path

Use this lane when you want to inspect or troubleshoot the small local debug run one stage at a time.

### Manual Debug Base Training

Run the debug base-model stages by hand.

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --train-config sample-configs/train-debug.json

webbgpt train-continue \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --train-config sample-configs/train-debug.json
```

### Manual Debug Alignment

Align the debug checkpoint into a more assistant-like model.

These manual debug stage commands all reuse `artifacts/checkpoints-debug`. If you instead ran `webbgpt main --profile debug`, skip this manual path and use `artifacts/runs/debug/checkpoints/dpo/best` for eval or export.

```bash
webbgpt train-sft \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --train-config sample-configs/train-debug.json

webbgpt train-dpo \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --train-config sample-configs/train-debug.json \
  --reference-checkpoint artifacts/checkpoints-debug/best
```

### Manual Debug Evaluation

Measure the debug checkpoint on validation loss, assistant behavior, and the grounded evaluation set before exporting or serving anything.

```bash
webbgpt eval \
  --model-config sample-configs/model-debug.json \
  --data-config sample-configs/data-debug.json \
  --eval-config sample-configs/eval-debug.json \
  --checkpoint artifacts/checkpoints-debug/best
```

### Manual Debug Export

Turn the internal debug checkpoint into a Hugging Face-style model directory.

```bash
webbgpt export-hf \
  --model-config sample-configs/model-debug.json \
  --checkpoint artifacts/checkpoints-debug/best \
  --output artifacts/runs/debug/export/final
```

### Manual Debug Webb Sync

Refresh the live Webb grounding snapshot before testing grounded answers in the debug lane.

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-debug.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

### Manual Debug Serving

Launch the local grounded assistant API and browser playground against the debug export plus the latest Webb grounding snapshot.

```bash
webbgpt serve --serve-config sample-configs/serve-debug.json
```

### Manual Local-MVP Path

Use this when you want to run the local conversational MVP lane stage by stage instead of using `webbgpt main --profile local-mvp`.

The manual local-MVP path now has stricter trustworthiness defaults:

- manual stage commands materialize and reuse profile-scoped prepared manifests under `artifacts/runs/local-mvp/prepared`
- manual stage commands write stage checkpoints under `artifacts/runs/local-mvp/checkpoints/<stage>`
- `train-continue` initializes from the latest completed pretrain checkpoint
- `train-sft` initializes from the latest completed continue checkpoint
- `train-dpo` starts from the SFT stage output and uses the supplied SFT checkpoint as the frozen reference
- SFT and DPO require explicit validation sources by default for local-MVP
- SFT and DPO save numbered checkpoints, a pinned `best` checkpoint, and top-K candidate checkpoint metadata
- `selection.json` explains why a new best replaced the prior one
- eval reports provenance, attribution lanes, benchmark reliability, and retrieval audit
- serve uses the same Webb grounding snapshot store as eval and writes manual transcripts when `transcript_path` is configured
- manual demos are only comparable when the recorded checkpoint, tokenizer/export artifact, backend, decode preset, seed bundle, and grounding snapshot match

For direct local reruns, prefer the explicit Python 3.12 entrypoint below:

- use `python3.12 -m src.cli ...` if you want the exact direct command path
- add `--force-rebuild` when stage inputs changed or a prepared-manifest mismatch appears
- do not run the same `local-mvp` stage concurrently against the same output directory
- `train-continue` should be allowed to skip honestly if the readiness gate fails
- `train-sft` uses the rerun-filtered SFT files already wired in `sample-configs/data-local-mvp.json`
- `train-dpo` should only be run when the reviewed pair set exceeds the configured local-mvp floor and the SFT artifact is promotable

```bash
python3.12 -m src.cli train-pretrain \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

python3.12 -m src.cli train-continue \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

python3.12 -m src.cli train-sft \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

python3.12 -m src.cli train-dpo \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --reference-checkpoint artifacts/runs/local-mvp/checkpoints/sft/best

python3.12 -m src.cli eval \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --eval-config sample-configs/eval-local-mvp.json \
  --checkpoint artifacts/runs/local-mvp/checkpoints/dpo/best

python3.12 -m src.cli export-hf \
  --model-config sample-configs/model-local-mvp.json \
  --checkpoint artifacts/runs/local-mvp/checkpoints/dpo/best \
  --output artifacts/runs/local-mvp/export/final

python3.12 -m src.cli webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'

python3.12 -m src.cli serve --serve-config sample-configs/serve-local-mvp.json
```

If you need to rebuild prepared manifests cleanly for a rerun, use the same commands with `--force-rebuild` on the train stage you are re-running:

```bash
python3.12 -m src.cli train-pretrain \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --force-rebuild

python3.12 -m src.cli train-continue \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --force-rebuild

python3.12 -m src.cli train-sft \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --force-rebuild

python3.12 -m src.cli train-dpo \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --reference-checkpoint artifacts/runs/local-mvp/checkpoints/sft/best \
  --force-rebuild
```

Useful local-mvp status checks:

```bash
python3.12 -m src.cli audit-data \
  --config sample-configs/data-local-mvp.json \
  --stage continue

ls artifacts/runs/local-mvp/checkpoints/pretrain
ls artifacts/runs/local-mvp/checkpoints/continue
ls artifacts/runs/local-mvp/checkpoints/sft
ls artifacts/runs/local-mvp/checkpoints/dpo

cat artifacts/runs/local-mvp/checkpoints/pretrain/stage_summary.json
cat artifacts/runs/local-mvp/checkpoints/continue/stage_summary.json
cat artifacts/runs/local-mvp/checkpoints/sft/stage_summary.json
cat artifacts/runs/local-mvp/checkpoints/dpo/stage_summary.json

cat artifacts/runs/local-mvp/checkpoints/pretrain/run_metadata.json
cat artifacts/runs/local-mvp/checkpoints/continue/run_metadata.json
cat artifacts/runs/local-mvp/checkpoints/sft/run_metadata.json
cat artifacts/runs/local-mvp/checkpoints/dpo/run_metadata.json
```

### Manual Remote-3B Path

Use this lane when you want to run the intermediate serious 3B path stage by stage instead of `webbgpt main`. This path is config-driven rather than preset-driven: `--mvp` and `--full` only exist on `webbgpt main`.

Manual remote-3B stage commands materialize and reuse profile-scoped prepared manifests under `artifacts/runs/remote-3b/prepared` and write stage checkpoints under `artifacts/runs/remote-3b/checkpoints/<stage>`.

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b.json \
  --train-config sample-configs/train-3b.json

webbgpt train-continue \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b.json \
  --train-config sample-configs/train-3b.json

webbgpt train-sft \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b.json \
  --train-config sample-configs/train-3b.json

webbgpt train-dpo \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b.json \
  --train-config sample-configs/train-3b.json \
  --reference-checkpoint artifacts/runs/remote-3b/checkpoints/sft/best

webbgpt eval \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b.json \
  --eval-config sample-configs/eval-3b.json \
  --checkpoint artifacts/runs/remote-3b/checkpoints/dpo/best

webbgpt export-hf \
  --model-config sample-configs/model-3b.json \
  --checkpoint artifacts/runs/remote-3b/checkpoints/dpo/best \
  --output artifacts/runs/remote-3b/export/final

webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-3b.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'

webbgpt serve --serve-config sample-configs/serve-3b.json
```

### Manual Remote-7B Path

Use this lane on the intended Linux multi-GPU system when you want to run the full 7B path stage by stage instead of relying on `webbgpt main --profile remote-7b`.

Manual remote-7B stage commands materialize and reuse profile-scoped prepared manifests under `artifacts/runs/remote-7b/prepared` and write stage checkpoints under `artifacts/runs/remote-7b/checkpoints/<stage>`.

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-7b.json \
  --data-config sample-configs/data-7b.json \
  --train-config sample-configs/train-7b.json

webbgpt train-continue \
  --model-config sample-configs/model-7b.json \
  --data-config sample-configs/data-7b.json \
  --train-config sample-configs/train-7b.json

webbgpt train-sft \
  --model-config sample-configs/model-7b.json \
  --data-config sample-configs/data-7b.json \
  --train-config sample-configs/train-7b.json

webbgpt train-dpo \
  --model-config sample-configs/model-7b.json \
  --data-config sample-configs/data-7b.json \
  --train-config sample-configs/train-7b.json \
  --reference-checkpoint artifacts/runs/remote-7b/checkpoints/sft/best

webbgpt eval \
  --model-config sample-configs/model-7b.json \
  --data-config sample-configs/data-7b.json \
  --eval-config sample-configs/eval-7b.json \
  --checkpoint artifacts/runs/remote-7b/checkpoints/dpo/best

webbgpt export-hf \
  --model-config sample-configs/model-7b.json \
  --checkpoint artifacts/runs/remote-7b/checkpoints/dpo/best \
  --output artifacts/runs/remote-7b/export/final

webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-7b.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'

webbgpt serve --serve-config sample-configs/serve-7b.json
```

### Tests

Run the default regression suite:

```bash
webbgpt test
```

Run the full suite, including torch-dependent tests:

```bash
webbgpt test --all
```

## Architecture

### Pipeline At A Glance

1. `src/cli.py` loads typed configs, dispatches subcommands, writes starter configs and seed assets, and coordinates the project from tokenizer creation through serving.
2. `src/tokenizer/spm.py` freezes a SentencePiece tokenizer, while `src/data/tokenizer_corpus.py` builds a real tokenizer corpus from FineWeb-Edu and emits progress updates during long corpus builds.
3. `src/data/` loads raw text, JSONL, Hugging Face datasets, or prepared manifests, then cleans, tokenizes, packs, and materializes stage-specific artifacts.
4. `src/model/` defines the custom decoder-only transformer, grouped-query attention, rotary embeddings, KV-cache generation, and anti-collapse generation controls.
5. `src/train/` runs base pretraining and continued pretraining, including optimizer setup, checkpointing, distributed/FSDP hooks, logging, evaluation, resume logic, and top-level run metadata.
6. `src/posttrain/` reuses the same model stack for SFT and DPO alignment, including explicit validation support, pinned qualitative regression prompts, best-checkpoint metadata, top-K candidate preservation, and early stopping.
7. `src/eval/` measures held-out perplexity, assistant behavior, grounded catalog or Webb quality, attribution lanes, route audit, release gates, benchmark reliability, retrieval audit, and grounding quality across the expanded Webb domain set.
8. `src/export/hf.py` converts internal checkpoints and tokenizer artifacts into a Hugging Face-style export and now writes export provenance.
9. `src/grounding/` now powers the default Webb grounding stack for both live and fixture-backed syncs, with family-aware snapshots, source documents, handbook sections, faculty records, admissions facts, publications, athletics tables, retrieval chunks, partial refresh, carry-forward behavior, and hybrid no-embed lookup.
10. `src/serve/` exposes the assistant as a FastAPI app, chooses a runtime backend, grounds catalog or Webb answers, records provenance/transcripts, uses deterministic no-hit abstention for grounded factual misses, supports low-confidence route fan-out, and serves the answer-first browser UI with malformed-output interception plus structured details.

### Repository Map

- `README.md`: Main operator guide, setup reference, architecture map, and lane documentation.
- `pyproject.toml`: Packaging, dependency, entrypoint, and tool configuration.
- `src/`: All authored runtime code.
- `src/tests/`: Regression coverage for config loading, packing, checkpoints, prepared-data resume behavior, evaluation trustworthiness, Webb grounding/snapshot sync, and CLI/profile guard rails.
- `sample-configs/`: Supported starter configs for `debug`, `local-mvp`, `remote-3b`, and `remote-7b`, plus shared tokenizer configs.
- `data/`: Checked-in example datasets for tokenizer building, local post-training, evaluation, live Webb source packs, the structured Webb fixture corpus, and a few retained legacy catalog fixtures used by older internal tests.
- `artifacts/`: Generated runtime outputs such as tokenizer files, manifests, checkpoints, exports, evaluation results, transcripts, and snapshot-backed Webb grounding stores.

### Source Tree

#### CLI And Project Orchestration

- `src/cli.py`: Single CLI entrypoint for config init, tests, profile-aware `main`, tokenizer corpus generation, tokenizer training, prepared-data materialization, training stages, evaluation, export, Webb snapshot sync/diff commands, and serving. It manages profile defaults, stage lineage, hardware/profile guard rails, and starter-data generation.

#### `src/config/`

- `src/config/__init__.py`: Convenience re-export surface for config dataclasses and config I/O helpers.
- `src/config/io.py`: Generic config loader/saver used by the CLI and other subsystems to read JSON or TOML files into typed dataclasses.
- `src/config/schemas.py`: Core typed config layer. Defines tokenizer, model, data, training, evaluation, serving, checkpoint, release-gate, and Webb grounding schema fields.

#### `src/tokenizer/`

- `src/tokenizer/__init__.py`: Re-exports tokenizer utilities.
- `src/tokenizer/chat_template.py`: Shared chat prompt serializer used by SFT, DPO, evaluation, and serving.
- `src/tokenizer/spm.py`: SentencePiece wrapper for tokenizer training, tokenizer metadata export, special-token handling, tokenizer loading, and training heartbeats.

#### `src/data/`

- `src/data/__init__.py`: Re-exports dataset builders and data schemas.
- `src/data/dataset.py`: Central dataset builder. Reads configured raw or prepared sources, applies preprocessing, tokenizes text, builds training datasets, materializes prepared stage artifacts, manages safe prepared-data reuse/resume decisions, supports grouped post-train splitting, and enforces fail-closed post-train validation rules when requested.
- `src/data/packing.py`: Packs token sequences into fixed-length causal-language-model windows with EOS boundaries and padding.
- `src/data/prepared.py`: Writes prepared `.npy` shard artifacts plus manifests, manages resumable prepared-data state files and buffer snapshots, loads shard-backed datasets, and implements assistant-only SFT label masking.
- `src/data/preprocess.py`: Light cleaning and filtering layer for whitespace normalization, basic PII scrubbing hooks, simple quality filtering, and deduplication entry points.
- `src/data/schemas.py`: Dataclasses for raw documents, chat/SFT records, and preference/DPO examples.
- `src/data/tokenizer_corpus.py`: Streams a large tokenizer corpus from Hugging Face datasets, currently defaulting to FineWeb-Edu, writes tokenizer-corpus text plus metadata, and emits periodic progress for long corpus builds.

#### `src/model/`

- `src/model/__init__.py`: Re-export surface for model classes and helpers.
- `src/model/attention.py`: Grouped-query self-attention implementation, including RoPE application and KV-cache-aware attention execution.
- `src/model/cache.py`: KV-cache dataclasses used during autoregressive generation.
- `src/model/export.py`: Compatibility re-export so model-facing code can reach the Hugging Face export path without importing `src/export/` directly.
- `src/model/modules.py`: Shared neural-network building blocks such as RMSNorm, rotary embedding utilities, causal/additive attention masks, KV-head repetition, and SwiGLU components.
- `src/model/transformer.py`: Main `CausalTransformer` implementation. Defines the embedding stack, decoder blocks, LM head, loss computation, and token-by-token generation logic, including repetition penalty, no-repeat ngram support, and explicit stop-token handling.

#### `src/train/`

- `src/train/__init__.py`: Re-exports training entrypoints.
- `src/train/checkpoint.py`: Checkpoint manager for saving/loading model, optimizer, scheduler, RNG state, and extra metadata. Handles overwrite-safe checkpoint directories and FSDP-aware state dict behavior.
- `src/train/distributed.py`: Distributed runtime helpers for initialization, rank/world-size inspection, FSDP wrapping, synchronization barriers, and cleanup.
- `src/train/entrypoints.py`: Stage-specific training entrypoints that connect configs, datasets, dataloaders, model construction, and the shared loop for pretraining and continued pretraining.
- `src/train/loop.py`: Shared training loop with forward/backward passes, grad clipping, evaluation cadence, checkpoint saving, run metadata, eval history logging, top-K post-train candidate tracking, best-checkpoint metadata, and final-checkpoint handling.
- `src/train/optim.py`: Optimizer and scheduler factory code, including AdamW parameter grouping and cosine decay with warmup.

#### `src/posttrain/`

- `src/posttrain/__init__.py`: Re-exports SFT and DPO entrypoints.
- `src/posttrain/eval.py`: Post-train regression helpers for prompt-overlap protection, qualitative samples, selection metadata, and top-K candidate bookkeeping.
- `src/posttrain/sft.py`: Supervised fine-tuning stage with explicit-validation policy support, held-out eval, qualitative regression sampling, and seeded run metadata.
- `src/posttrain/dpo.py`: DPO stage with held-out preference eval, LM-health checks, fixed gradient accumulation semantics, qualitative regression sampling, best-checkpoint selection metadata, and top-K candidate preservation.

#### `src/eval/`

- `src/eval/__init__.py`: Re-exports evaluation entrypoints and keeps imports light so config or scoring code does not eagerly pull in the full runtime stack.
- `src/eval/assistant.py`: Assistant benchmark runner with stricter scoring, anti-filler penalties, anti-degeneration heuristics, and shared stop-string generation behavior.
- `src/eval/catalog.py`: Legacy grounded catalog benchmark runner that can score canned responses or live model-backed answers. It reports `model_only`, `pipeline_grounded`, and `retrieval_oracle` attribution lanes plus retrieval audit signals.
- `src/eval/perplexity.py`: Held-out perplexity evaluation over validation batches.
- `src/eval/runner.py`: High-level evaluation coordinator that loads checkpoints, runs validation, dispatches grounded benchmark suites, computes release-gate summaries, and emits provenance plus benchmark/scorer/gate version information, route audit, and grounding quality.
- `src/eval/webb.py`: Webb benchmark runner for course, handbook, faculty, admissions/general info, student-life, mission/values, college guidance, museum, athletics, planner beta, and mixed multi-domain evaluation, including route audit, route fan-out handling, and grounded attribution lanes.

#### `src/grounding/`

- `src/grounding/__init__.py`: Lazy re-export surface for grounding helpers so lightweight imports do not eagerly require SQLAlchemy.
- `src/grounding/ingest.py`: Supports `ingest-webb-site`, `ingest-webb-handbook`, `webb-sync`, and `diff-webb-snapshot` for the Webb grounding stack, including live or fixture-backed sync, family-aware refresh, carry-forward behavior, athletics ingest, and text-first handbook extraction with optional OCR fallback.
- `src/grounding/provider.py`: Route-aware Webb grounding provider with `course_catalog`, `faculty`, `handbook_policy`, `student_life`, `admissions_general`, `museum_programs`, `athletics`, `planner_advice`, `chat`, and low-confidence top-2 route fan-out.
- `src/grounding/store.py`: SQLAlchemy-backed access layer for the Webb knowledge-store schema, including snapshots, source documents, retrieval chunks, handbook sections, faculty records, admissions facts, publication versions, athletics tables, snapshot diffing, carry-forward helpers, and snapshot quality checks.
- `src/grounding/types.py`: Dataclasses for grounding results, citations, and route decisions.
- `src/grounding/sql/__init__.py`: Re-exports SQLAlchemy metadata base definitions.
- `src/grounding/sql/models.py`: SQL schema for the Webb knowledge-store tables such as `knowledge_snapshots`, `source_documents`, `retrieval_chunks`, `course_versions`, `faculty_profiles`, `handbook_sections`, `admissions_facts`, and `publication_versions`.

#### `src/export/`

- `src/export/__init__.py`: Re-exports export helpers.
- `src/export/hf.py`: Hugging Face export path. Converts internal checkpoints to `config.json`, tokenizer metadata, generation config, model weights, and export provenance.

#### `src/serve/`

- `src/serve/__init__.py`: Lazy re-export surface for serving interfaces.
- `src/serve/app.py`: FastAPI application factory and server runner. Exposes `/`, `/status`, `/healthz`, `/docs`, and `/v1/chat/completions`, attaches provenance, can run Webb sync-on-start, falls back to the newest completed trusted snapshot when startup sync fails, and can append manual demo transcripts.
- `src/serve/orchestrator.py`: Assistant layer that decides when to ground against the Webb store, supports low-confidence route fan-out, uses deterministic no-hit fallbacks for grounded factual misses, asks for clarification on ambiguous mixed-timeframe grounded questions, and assembles the final response payload plus metadata, route traces, and response-quality signals.
- `src/serve/playground.py`: Inlined browser playground renderer for manual testing at the root route. It keeps the UI answer-first, surfaces compact status pills, exposes retry/repro/debug actions for malformed outputs, and renders a structured details panel instead of raw JSON by default.
- `src/serve/types.py`: Serving request and response types.
- `src/serve/backends/__init__.py`: Re-exports serving backend adapters.
- `src/serve/backends/transformers_backend.py`: Development fallback inference backend using `transformers` + `torch`, with explicit stop strings and anti-loop settings.
- `src/serve/backends/vllm_backend.py`: Preferred inference backend for vLLM deployments, also wired to explicit stop strings and decode settings.

#### Shared Runtime Utilities

- `src/generation.py`: Shared generation helpers for stop-string handling, stop-token resolution, prompt fingerprinting, repetition penalty, and no-repeat ngram masking.
- `src/provenance.py`: Shared provenance and versioning helpers for checkpoints, tokenizer artifacts, exports, catalog or grounding snapshots, benchmark suites, scorers, and release-gate configs.
- `src/repro.py`: Shared seeding helper for Python, NumPy, and torch.

### Config Packs

#### `sample-configs/` (active starter configs)

Unsuffixed `sample-configs/*.json` files are shared Webb base/template configs written by `webbgpt init-config` for manual use and the offline demo path. Profile automation uses the suffixed lane-specific files such as `*-debug.json`, `*-local-mvp.json`, `*-3b.json`, and `*-7b.json`.

- `sample-configs/tokenizer-corpus.json`: FineWeb-Edu tokenizer-corpus build settings.
- `sample-configs/tokenizer.json`: Real SentencePiece tokenizer settings for the frozen project tokenizer.
- `sample-configs/tokenizer-corpus-local-mvp.json`: Capped tokenizer-corpus build for the Mac-class local MVP lane.
- `sample-configs/tokenizer-local-mvp.json`: Smaller 32k-vocab tokenizer settings for `local-mvp`.
- `sample-configs/tokenizer-corpus-7b.json`: Tokenizer-corpus build settings for the 7B lane.
- `sample-configs/tokenizer-7b.json`: SentencePiece tokenizer settings for the 7B lane.
- `sample-configs/model-local-mvp.json`: Small conversational model spec tuned for local training.
- `sample-configs/train-local-mvp.json`: Local-MVP training defaults with explicit SFT/DPO validation requirements, fail-closed validation-size floors, lower post-train learning rates, top-K checkpoint preservation, and gradient accumulation for small-device runs.
- `sample-configs/data-local-mvp.json`: Local-MVP data registry with a larger pretraining target, a larger LM-safe continued-pretraining target, explicit held-out post-train validation files, and a cleaned LM-safe continue corpus.
- `sample-configs/eval-local-mvp.json`: Local-MVP evaluation config with live Webb grounding defaults, sync-on-start snapshot refresh, stricter release gates than before, explicit anti-loop decode settings, and enforced grounded missing-data abstention.
- `sample-configs/serve-local-mvp.json`: Local-MVP serving config targeting `artifacts/runs/local-mvp/export/final`, using the export directory as tokenizer source, live Webb grounding defaults, explicit anti-loop decode settings, and transcript capture.
- `sample-configs/model-3b.json`: Canonical 3B model spec for serious training runs.
- `sample-configs/train-3b.json`: Intermediate 3B training defaults with stage-specific step budgets and more frequent checkpoints.
- `sample-configs/data-3b.json`: Intermediate serious data registry for FineWeb-Edu pretraining, held-out validation, multi-source domain continued pretraining, and multi-source SFT/preference bundles.
- `sample-configs/eval-3b.json`: Serious-run evaluation config with chat sanity, Webb grounded benchmark coverage, sync-on-start snapshot refresh, and enforced release gates.
- `sample-configs/serve-3b.json`: Serious-run serving config targeting `artifacts/runs/remote-3b/export/final` with live Webb grounding defaults.
- `sample-configs/model-7b.json`: Balanced 7B model spec for the full powerful-machine lane.
- `sample-configs/train-7b.json`: Full 7B training defaults for the Linux multi-GPU path.
- `sample-configs/data-7b.json`: Full 7B data registry with large token budgets and the serious post-training bundle.
- `sample-configs/eval-7b.json`: Full 7B evaluation config with Webb grounded benchmark coverage, sync-on-start snapshot refresh, and enforced release gates.
- `sample-configs/serve-7b.json`: Full 7B serving config targeting `artifacts/runs/remote-7b/export/final` with live Webb grounding defaults.
- `sample-configs/model-debug.json`: Small debug model config meant only to run locally as a smoke test.
- `sample-configs/train-debug.json`: Small debug training config with short runs and local checkpoint output.
- `sample-configs/data-debug.json`: Local dataset wiring for debug pretraining, continued pretraining, SFT, preference training, and validation.
- `sample-configs/eval-debug.json`: Local debug evaluation settings using the Webb benchmark families without enforced release gates.
- `sample-configs/serve-debug.json`: Local debug serving settings pointing at `artifacts/runs/debug/export/final` with live Webb grounding defaults.

### Checked-In Example Data

#### `data/raw/`

- `data/raw/tokenizer_corpus.txt`: Local tokenizer corpus output written by `webbgpt build-tokenizer-corpus`.
- `data/raw/tokenizer_corpus.txt.meta.json`: Metadata summary describing the tokenizer-corpus build that produced the text file.
- `data/raw/tokenizer_corpus_local_mvp.txt`: Local-MVP tokenizer corpus output.
- `data/raw/tokenizer_corpus_local_mvp.txt.meta.json`: Metadata summary for the local-MVP tokenizer corpus.

#### `data/local/`

- `data/local/sft.jsonl`: Small local SFT seed set for assistant behavior.
- `data/local/sft_validation.jsonl`: Local held-out SFT validation set.
- `data/local/preference.jsonl`: Small local preference seed set.
- `data/local/preference_validation.jsonl`: Local held-out DPO validation set.

#### `data/domain/`

- `data/domain/education_corpus.txt`: Continued-pretraining seed text for general academic explanation and education-style writing.
- `data/domain/advising_corpus.txt`: Continued-pretraining seed text for advising behavior and decision support.
- `data/domain/philosophy_corpus.txt`: Continued-pretraining seed text for philosophy-style explanation and comparison.
- `data/domain/catalog_corpus.txt`: Continued-pretraining seed text for catalog-grounded language, citation habits, and abstention behavior.
- `data/domain/local_mvp_continue_corpus.txt`: LM-safe local-MVP continued-pretraining corpus assembled from domain guidance and catalog facts. It intentionally avoids rejected preference strings and meta-policy wrapper prose.
- `data/domain/webb_public_corpus.txt`: LM-safe public Webb prose for continued pretraining when you want Webb-specific language without teaching current facts directly through post-train supervision.

#### `data/posttrain/`

- `data/posttrain/sft_public_seed.jsonl`: Seed SFT examples for basic conversational behavior and anti-gibberish behavior.
- `data/posttrain/sft_domain_synthetic.jsonl`: Seed SFT examples for advising, explanation, grounded catalog behavior, and explicit abstention.
- `data/posttrain/sft_conversation_seed.jsonl`: Additional conversational SFT examples with less canned filler and more direct user-facing answers.
- `data/posttrain/sft_validation.jsonl`: Held-out SFT validation examples for local-MVP and serious profiles.
- `data/posttrain/sft_webb_seed.jsonl`: Webb-specific SFT seeds for handbook answers, course comparisons, course-diff answers, faculty lookup, admissions/general info, student-life, and museum-style grounded answers.
- `data/posttrain/sft_webb_validation.jsonl`: Held-out Webb SFT validation examples.
- `data/posttrain/preference_public_seed.jsonl`: Seed preference pairs for conversational quality and degeneration resistance.
- `data/posttrain/preference_domain_synthetic.jsonl`: Seed preference pairs for grounded answers, abstention, and advising quality.
- `data/posttrain/preference_conversation_seed.jsonl`: Additional preference pairs focused on anti-repetition, direct answers, and refusing to invent missing facts.
- `data/posttrain/preference_validation.jsonl`: Held-out DPO validation examples.
- `data/posttrain/preference_webb_seed.jsonl`: Webb-specific preference pairs for citation-first answers, abstention, anti-hallucination behavior, and clear fact-vs-advice boundaries.
- `data/posttrain/preference_webb_validation.jsonl`: Held-out Webb DPO validation examples.

#### `data/eval/`

- `data/eval/chat_sanity.jsonl`: Sanity benchmark covering greetings, short explanations, uncertainty handling, catch-up planning, and anti-collapse checks.
- `data/eval/assistant.jsonl`: Assistant benchmark with more domain-help and nontrivial prompts than the old minimal slice.
- `data/eval/posttrain_regression.jsonl`: Pinned qualitative regression prompt suite that must not overlap post-train train or validation prompts.
- `data/eval/catalog.responses`: Retained legacy catalog-present benchmark fixture used by older internal eval/tests.
- `data/eval/catalog_missing.responses`: Retained legacy catalog-missing benchmark fixture used by older internal eval/tests.
- `data/eval/webb_course_present.responses`: Webb course-present benchmark.
- `data/eval/webb_course_missing.responses`: Webb course-missing abstention benchmark.
- `data/eval/webb_course_diff.responses`: Webb year-over-year course-diff benchmark.
- `data/eval/webb_handbook_present.responses`: Webb handbook-present benchmark with citation expectations.
- `data/eval/webb_handbook_missing.responses`: Webb handbook abstention benchmark.
- `data/eval/webb_faculty.responses`: Webb faculty lookup benchmark.
- `data/eval/webb_admissions.responses`: Webb admissions benchmark.
- `data/eval/webb_student_life.responses`: Webb student-life benchmark.
- `data/eval/webb_mission_values.responses`: Webb mission/values benchmark.
- `data/eval/webb_college_guidance.responses`: Webb college-guidance benchmark.
- `data/eval/webb_museum_programs.responses`: Webb museum/unique-program benchmark.
- `data/eval/webb_athletics_present.responses`: Webb athletics present-data benchmark.
- `data/eval/webb_athletics_missing.responses`: Webb athletics abstention benchmark.
- `data/eval/webb_planner.responses`: Webb planner/advice benchmark slice used for beta evaluation, not hard release support.
- `data/eval/webb_mixed_multi_domain.responses`: Mixed multi-domain route-fan-out benchmark.

#### `data/webb/`

- `data/webb/seed_urls.json`: Real Webb source pack starter list for live crawling or sync.
- `data/webb/seed_urls_demo.json`: Offline Webb fixture source pack used by the Webb test suite and optional local fixture-based sync runs. The demo pack now carries fixture metadata such as `source_kind`, `fixture_format`, and explicit department/year labels where relevant.
- `data/webb/seed_urls_private.json`: Optional local/private overlay pack for Webb-authorized documents that are not on the public site. Use it together with `--offline-seed-url-pack` when you want a live sync plus private local documents in the same grounding snapshot.
- `data/webb/source_policies.json`: Page-type routing rules used for Webb source classification, including department pages, student-life pages, athletics pages, and publication/catalog URLs.
- `data/webb/mock/`: Offline Webb fixtures for course catalog, department pages, faculty, admissions, student-life, mission/values, college guidance, museum/programs, athletics, publications, and handbook content used by fixture-based syncs and the Webb test suite. HTML fixtures now use cohesive semantic sections plus JSON payloads instead of raw transcript dumps, and the course-description pages follow a more consistent high-structure pattern across departments.
- `data/webb/mock/README.md`: Fixture conventions for keeping Webb mock sources consistent, structured, and ingestion-friendly.

The checked-in eval suite is intentionally broader than the earlier tiny slices, but it is still versioned content. Do not compare scores across runs unless the benchmark-suite, scorer, and release-gate hashes match.

#### `data/catalog/`

- `data/catalog/catalog.json`: Retained legacy catalog fixture used by older internal tests and benchmark coverage.

### Tests

- `webbgpt test`: Thin CLI wrapper around `pytest`. When torch cannot be imported cleanly and you do not pass `--all`, the CLI falls back to the safe non-torch subset instead of failing immediately.
- `src/tests/test_config.py`: Verifies config defaults, nested config round-tripping, and extended schema fields.
- `src/tests/test_packing.py`: Verifies token packing, EOS handling, and long-sequence splitting.
- `src/tests/test_model.py`: Verifies transformer forward shapes, generation-cache growth, and attention-mask behavior.
- `src/tests/test_checkpoint.py`: Verifies checkpoint save/load round-trips and overwrite-safe same-step checkpoint replacement.
- `src/tests/test_prepared.py`: Verifies assistant-only SFT masking behavior.
- `src/tests/test_prepare_resume.py`: Verifies prepared-data reuse, safe resume for interrupted runs, legacy-partial failure behavior, and `--force-rebuild`.
- `src/tests/test_posttrain_eval.py`: Verifies grouped validation splitting, prepared-vs-raw post-train validation behavior, overlap guards, and qualitative cleaning helpers.
- `src/tests/test_catalog_eval.py`: Verifies grounded catalog scoring, including missing-data abstention.
- `src/tests/test_cli_profiles.py`: Verifies profile guard rails and stage-lineage config construction.
- `src/tests/test_train_loop.py`: Verifies training falls back cleanly when `torch.compile` is unavailable in the current runtime combination.
- `src/tests/test_trustworthiness.py`: Verifies deterministic no-hit fallback, benchmark provenance/reliability helpers, fail-closed explicit validation behavior, anti-filler assistant scoring, top-K candidate pruning, and trusted-snapshot serve fallback behavior.
- `src/tests/test_webb_grounding.py`: Verifies Webb snapshot sync, handbook ingest, routing, OCR fallback behavior, family-aware carry-forward sync behavior, athletics/student-life retrieval, and snapshot diffing.

### Generated Runtime Outputs

- `artifacts/tokenizer/`: Frozen tokenizer artifacts written by tokenizer training.
- `artifacts/manifests/`: Prepared stage manifests written by `webbgpt prepare-data`.
- `artifacts/runs/<profile>/`: Profile-scoped prepared data, stage checkpoints, evaluation outputs, exports, and manual demo transcripts.
- `artifacts/catalog/`: Local SQLite grounding databases.
- `artifacts/grounding/`: Snapshot-backed Webb grounding databases and other non-legacy grounding artifacts. These store `knowledge_snapshots`, `source_documents`, `retrieval_chunks`, family metadata, and the structured Webb tables including athletics.
- `artifacts/runs/<profile>/prepared/*.json` plus sibling shard directories: Prepared-manifest entrypoints and the packed `.npy` shard payloads that large runs consume, including optional `sft_validation.json` and `preference_validation.json` when explicit post-train validation sources are configured.
- `artifacts/runs/<profile>/prepared/*.resume.json` plus sibling `.resume/` directories: In-progress prepared-data resume state and buffer snapshots used for safe auto-resume of interrupted preparation jobs.
- `artifacts/runs/<profile>/checkpoints/<stage>/run_metadata.json`: Stage-level run metadata including seeds, config snapshots, and post-train validation policy details.
- `artifacts/runs/<profile>/checkpoints/<stage>/eval_history.jsonl`: Chronological held-out evaluation history for post-train stages.
- `artifacts/runs/<profile>/checkpoints/<stage>/best/selection.json`: Best-checkpoint selection metadata for a post-train stage.
- `artifacts/runs/<profile>/checkpoints/<stage>/candidate-step-*/`: Preserved candidate checkpoint directories used for top-K selection and later inspection.
- `artifacts/runs/<profile>/checkpoints/<stage>/topk.json`: Preserved top-K post-train candidate summary.
- `artifacts/runs/<profile>/export/final/provenance.json`: Export provenance tying the export back to its checkpoint and tokenizer artifacts.
- `artifacts/runs/<profile>/manual_demos/chat_transcript.jsonl`: Optional manual serve transcripts when the serve config enables transcript capture.

### How The Pieces Fit Together

#### 1. Tokenizer And Corpus

- `webbgpt build-tokenizer-corpus` uses `src/data/tokenizer_corpus.py` plus the selected tokenizer-corpus config to build the raw text file that a profile’s tokenizer will train on.
- `webbgpt tokenize` uses `src/tokenizer/spm.py` plus the selected tokenizer config to freeze the SentencePiece tokenizer used everywhere else in the project.

#### 2. Data And Manifest Preparation

- `webbgpt prepare-data` uses `src/data/dataset.py`, `src/data/preprocess.py`, `src/data/packing.py`, and `src/data/prepared.py` to turn raw text or JSONL sources into stage-specific manifests and training examples.
- On serious runs, `webbgpt prepare-data` materializes shard-backed `.npy` artifacts plus manifests so training can read prepared datasets instead of rebuilding raw corpora in memory.

#### 3. Base Training

- `webbgpt train-pretrain` and `webbgpt train-continue` combine `src/model/`, `src/train/`, and the active config pack to run base LM training and continued pretraining.
- `src/train/checkpoint.py` owns resumability, while `src/train/loop.py` owns logging, eval cadence, candidate preservation, and checkpoint metadata.

#### 4. Alignment

- `webbgpt train-sft` uses `src/posttrain/sft.py` and explicit held-out validation sources by default in the supported local-MVP lane.
- `webbgpt train-dpo` uses `src/posttrain/dpo.py`, a frozen reference model, held-out preference validation, LM-health monitoring, and top-K candidate preservation.

#### 5. Evaluation, Export, Grounding, And Serving

- `webbgpt eval` runs `src/eval/runner.py` over validation, assistant, and grounded benchmark suites, then emits provenance, release-gate results, benchmark reliability, attribution lanes, retrieval audit, route audit, and grounding-quality signals.
- `webbgpt export-hf` uses `src/export/hf.py` to produce a serving-ready model directory plus export provenance.
- `webbgpt ingest-webb-site`, `webbgpt ingest-webb-handbook`, `webbgpt webb-sync`, and `webbgpt diff-webb-snapshot` drive the Webb grounding stack, which keeps current facts in retrieval rather than in the model weights and can refresh only selected source families when needed.
- `webbgpt serve` uses `src/serve/app.py`, `src/serve/orchestrator.py`, `src/serve/playground.py`, and the selected backend adapter to expose both an API and a manual browser UI with shared decode assumptions, snapshot-aware grounding, malformed-output interception, structured details, deterministic no-hit behavior, and trusted-snapshot startup fallback.

## Notes

- You do not need to use every layer of the repo. Training from scratch is only one path; grounding, evaluation, export, and serving can all be iterated on independently once you have a compatible model artifact.
- Treat `pipeline_grounded` scores and `model_only` scores as different measurements. A strong grounded answer does not necessarily mean the raw model is strong.
- Compare evaluation runs only when the benchmark-suite version, scorer version, release-gate config version, and grounding snapshot match.
- Keep eval and serve on the same grounding snapshot if you want apples-to-apples grounded comparisons.
- Treat manual demo transcripts the same way: only compare them when checkpoint, tokenizer/export artifact, backend, decode preset, seed bundle, and grounding snapshot all match.
- `local-mvp` is still an experimental local lane. If your hardware cannot sustain a useful token budget, shrink the model or the training objective instead of leaning harder on tiny post-training datasets.
- Webb service support is now broader than the original academics-first baseline: courses, handbook, faculty, admissions/general info, student-life, mission/values, college guidance, museum/programs, and athletics are grounded lanes, while planner/advice remains an opt-in beta lane.
- Webb retrieval is still intentionally hybrid lexical/fuzzy rather than embedding-based. The goal remains a transparent, inspectable grounding stack before adding a vector layer.
- Handbook ingest is text-first and only falls back to OCR when you explicitly allow it or when the source requires it for citable extraction.
- Family-aware freshness is policy-driven rather than scheduler-driven inside the repo today. Athletics, staff/admissions pages, and slower-moving families can refresh at different cadences, but you still trigger actual refreshes through `webb-sync` or `sync_on_start`.
- The Webb demo source pack is offline fixture data. It proves the Webb service architecture, route fan-out behavior, family-aware snapshots, and trusted-snapshot serving behavior, not the freshness of the live Webb site.
- Current questions default to the latest snapshot. Explicit year or season questions pin to stored historical versions when they exist. If a grounded question mixes current and historical timeframes in a way that changes the answer, serve asks for clarification instead of guessing.
- Source precedence is intentional: handbook beats generated advice on policy questions, current course/faculty/athletics surfaces beat stale historical summaries for current facts, and unresolved conflicts should be surfaced rather than blended away.
- The browser UI is intentionally conservative about presentation: it keeps the answer front and center, hides most trace data behind details, and surfaces malformed generations with retry plus repro actions instead of styling them as normal confident answers.
- `webbgpt test` assumes the dev extras are installed. If `pytest`, `torch`, `sentencepiece`, `datasets`, or SQLAlchemy are missing, the CLI or imports will fail for the parts of the stack that depend on them.
