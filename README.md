# WebbGPT

WebbGPT is a Python CLI for training a small decoder-only model, preparing posttraining data, grounding against Webb data, and serving a local assistant.

The current local development path is `local-mvp`.

## Current Local-MVP

`local-mvp` now means curated-real-data base pretraining:

- Pretraining uses curated real text for general legibility.
- Current pretraining sources are `data/raw/tokenizer_corpus_local_mvp.txt` and `data/raw/tokenizer_corpus.txt`.
- No generated continuation corpora are used for base pretraining.
- No Webb/domain-heavy generated corpora are used for base pretraining.
- Domain realization is disabled for this base-pretrain recipe.
- Webb-specific behavior is deferred to SFT, RAG, and domain grounding.

The final local-MVP experiment before 3B planning is the lower-LR curated recipe in `sample-configs/train-local-mvp.json`.

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

`webbgpt main --profile local-mvp` is also pretrain-only now. It builds/reuses tokenizer assets, materializes the curated pretrain and validation manifests, then runs base pretraining. It does not silently run continued pretraining, SFT, or DPO.

## Posttraining And Grounding

The current 22M local-MVP base is still a raw LM experiment. SFT, DPO, and Webb-specific behavior should be added only after selecting a usable base checkpoint.

Grounding/RAG work remains separate and can use the checked-in Webb fixtures:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls_demo.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url data/webb/mock/handbook.txt
```

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
- `data/eval/`: active eval fixtures and historical evals
- `data/webb/`: Webb grounding fixtures and source packs
- `src/`: CLI, model, training, eval, grounding, and serving code
- `tools/`: active reporting and audit tools
- `artifacts/`: generated tokenizers, manifests, checkpoints, reports, and exports
- `junk/`: preserved legacy files moved out of the active path

See `docs/CURRENT_LOCAL_MVP.md` for the current local-MVP baseline and rationale.
