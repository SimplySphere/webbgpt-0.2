# WebbGPT

WebbGPT is a Python CLI for four related jobs:

- train a decoder-only model from scratch
- post-train it with SFT and DPO
- ground it against Webb data
- serve it through a local FastAPI app and browser playground

If you are not sure where to start, use `local-mvp`. It is the main local-development lane. Use `debug` only when you want to prove the pipeline works end to end as quickly as possible.

## Start Here

| If you want to... | Use this |
| --- | --- |
| do the fastest end-to-end smoke test | `webbgpt main --profile debug` |
| do the best supported local run | `webbgpt main --profile local-mvp` |
| run the full pipeline but stop before serving | `webbgpt main --profile local-mvp --no-serve` |
| serve an already-exported model | `webbgpt serve --serve-config sample-configs/serve-local-mvp.json` |
| run stages manually | `build-tokenizer-corpus`, `tokenize`, `train-*`, `eval`, `export-hf`, `serve` |
| work on Webb grounding only | `webbgpt webb-sync`, `webbgpt ingest-webb-site`, `webbgpt ingest-webb-handbook`, `webbgpt diff-webb-snapshot` |
| run tests | `webbgpt test` |

## Setup

Examples below assume you are in the repo root and using Python `3.11` or `3.12`. The examples use `python3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,train,data,serve]'
webbgpt --help
```

If you only need part of the stack, the extras are split like this:

- `.[dev,data,serve]`: grounding, eval, and serving work
- `.[dev,train,data]`: tokenizer, data prep, training, and eval work

`sample-configs/` is already checked in. You only need to regenerate it if you want a fresh config pack:

```bash
webbgpt init-config --output-dir sample-configs
```

That command overwrites files in the target directory.

## Profiles

| Profile | Use it when... | Notes |
| --- | --- | --- |
| `debug` | you want the quickest plumbing check | smallest lane; good for smoke tests, not quality |
| `local-mvp` | you want the best repo-supported local path | recommended default on a Mac-class machine |
| `remote-3b` | you want a more serious 3B training run | supports `--mvp` and `--full`; not the default local choice |
| `remote-7b` | you are on a powerful Linux multi-GPU machine | intended for Linux; the CLI rejects clearly underpowered hosts |

If you are undecided, pick `local-mvp`.

## What `main` Does

`webbgpt main --profile ...` runs the full staged workflow for a profile:

1. build or reuse the tokenizer corpus
2. train or reuse the tokenizer
3. materialize prepared data manifests
4. run pretraining
5. run continued pretraining
6. run SFT
7. run DPO
8. run evaluation
9. export Hugging Face-style artifacts
10. optionally sync Webb grounding data and start the server

Outputs for profile-based runs land under `artifacts/runs/<profile>/`:

- `prepared/`: prepared manifests
- `checkpoints/pretrain|continue|sft|dpo/`: stage checkpoints
- `eval/result.json`: evaluation output
- `export/final/`: exported model

Profile eval and serve configs use Webb grounding. In practice, that means a profile run may sync live Webb sources unless you change the grounding config or use the offline demo workflow shown below.

## Common Workflows

### 1. Fastest Smoke Test

Use this when you only want to prove the repo works from end to end.

```bash
webbgpt main --profile debug
```

If you want the same run without starting the server:

```bash
webbgpt main --profile debug --no-serve
```

### 2. Best Local Run

Use this when you want the main local-development path.

```bash
webbgpt main --profile local-mvp
```

If you want training, evaluation, and export but want to serve later:

```bash
webbgpt main --profile local-mvp --no-serve
webbgpt serve --serve-config sample-configs/serve-local-mvp.json
```

Notes:

- `local-mvp` is the normal choice for local iteration.
- The repo already includes `sample-configs/`, local seed data, and checked-in tokenizer corpora.
- `main` reuses existing tokenizer corpus and tokenizer artifacts when the config still matches them.
- The default profile workflow still uses Webb grounding during evaluation and serving.
- Serving uses `http://127.0.0.1:8000/` by default, with API docs at `http://127.0.0.1:8000/docs`.

### 3. Serve an Existing Export

Use this when you already have an exported model directory and do not want to rerun training.

If `artifacts/runs/local-mvp/export/final/` already exists:

```bash
webbgpt serve --serve-config sample-configs/serve-local-mvp.json
```

If you only have a checkpoint and need to export first:

```bash
webbgpt export-hf \
  --model-config sample-configs/model-local-mvp.json \
  --checkpoint artifacts/runs/local-mvp/checkpoints/dpo/best \
  --output artifacts/runs/local-mvp/export/final

webbgpt serve --serve-config sample-configs/serve-local-mvp.json
```

By default, `serve` refuses non-promotable artifacts. Use `--force-untrusted` only for local debugging.

### 4. Run Stages Manually

Use this when you want stage-by-stage control instead of `webbgpt main`.

Build the tokenizer corpus and tokenizer:

```bash
webbgpt build-tokenizer-corpus --config sample-configs/tokenizer-corpus-local-mvp.json
webbgpt tokenize \
  --config sample-configs/tokenizer-local-mvp.json \
  --input data/raw/tokenizer_corpus_local_mvp.txt
```

Run the training stages:

```bash
webbgpt train-pretrain \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

webbgpt train-continue \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

webbgpt train-sft \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json

webbgpt train-dpo \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --train-config sample-configs/train-local-mvp.json \
  --reference-checkpoint artifacts/runs/local-mvp/checkpoints/sft/best
```

Evaluate, export, and serve:

```bash
webbgpt eval \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp.json \
  --eval-config sample-configs/eval-local-mvp.json \
  --checkpoint artifacts/runs/local-mvp/checkpoints/dpo/best

webbgpt export-hf \
  --model-config sample-configs/model-local-mvp.json \
  --checkpoint artifacts/runs/local-mvp/checkpoints/dpo/best \
  --output artifacts/runs/local-mvp/export/final

webbgpt serve --serve-config sample-configs/serve-local-mvp.json
```

Important behavior when you use the matching profile config files:

- `train-pretrain`, `train-continue`, `train-sft`, and `train-dpo` automatically materialize prepared manifests under `artifacts/runs/local-mvp/prepared/`.
- `train-continue` automatically initializes from the latest staged pretrain checkpoint unless the train config says otherwise.
- `train-sft` automatically initializes from the latest staged continue checkpoint, or from pretrain if the continue stage was skipped.
- `train-dpo` still requires an explicit `--reference-checkpoint`.

You only need `prepare-data` and `audit-data` when you want to inspect or control those lower-level steps yourself:

```bash
webbgpt audit-data --config sample-configs/data-local-mvp.json --stage continue

webbgpt prepare-data \
  --config sample-configs/data-local-mvp.json \
  --stage sft \
  --output /tmp/webbgpt-sft-manifest.json
```

### 5. Grounding Only

Use this when you want to work on Webb ingest, retrieval, or snapshot diffs without retraining a model.

Offline demo sync from the checked-in fixture pack:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-demo.db \
  --seed-url-pack data/webb/seed_urls_demo.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url data/webb/mock/handbook.txt
```

Live Webb sync:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

Add a private or local overlay pack on top of the live sync:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls.json \
  --offline-seed-url-pack data/webb/seed_urls_private.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890'
```

Sync only selected families:

```bash
webbgpt webb-sync \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --seed-url-pack data/webb/seed_urls.json \
  --source-policy-path data/webb/source_policies.json \
  --handbook-url 'https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890' \
  --families athletics faculty
```

Diff two snapshots:

```bash
webbgpt diff-webb-snapshot \
  --dsn sqlite:///artifacts/grounding/webbgpt-local-mvp.db \
  --from-snapshot <older-snapshot-id> \
  --to-snapshot <newer-snapshot-id>
```

### 6. Remote Profiles

Use these only when you intentionally want the larger training lanes.

Remote 3B:

```bash
webbgpt main --profile remote-3b --no-serve --mvp
```

Use `--full` instead of `--mvp` for the longer 3B run.

Remote 7B:

```bash
webbgpt main --profile remote-7b --no-serve
```

The CLI is deliberately opinionated here:

- `remote-3b` is not the recommended local default.
- `remote-7b` is intended for a powerful Linux host and will be rejected on obviously unsuitable machines.

### 7. Tests

There is a dedicated test command even though it is not shown in `webbgpt --help`:

```bash
webbgpt test
webbgpt test --all
```

`webbgpt test` runs a safe subset automatically if `torch` is not importable. Use `--all` to force the full suite.

You can still run `pytest` directly if you want a specific file:

```bash
python3.12 -m pytest src/tests/test_cli_profiles.py
```

## Repo Layout

- `sample-configs/`: checked-in starter configs for all supported profiles
- `data/local/`: local SFT and preference seed data used by `local-mvp`
- `data/webb/mock/`: offline Webb fixture set for grounding and demo syncs
- `data/eval/`: evaluation prompts and benchmark fixtures
- `src/`: CLI, model, training, eval, grounding, and serving code
- `artifacts/`: generated tokenizers, grounding DBs, checkpoints, eval results, and exports

## Troubleshooting

- If a profile run says config files are missing, regenerate them with `webbgpt init-config --output-dir sample-configs`.
- If `serve`, `eval`, or `export-hf` refuses an artifact, the checkpoint/export is not promotable. Use `--force-untrusted` only for local debugging.
- If you only want training and export, add `--no-serve` to `webbgpt main --profile ...`.
- If you want a fully offline grounding demo, use `data/webb/seed_urls_demo.json` and `data/webb/mock/handbook.txt`.
- If you are on a Mac and a remote profile is rejected, use `local-mvp` unless you explicitly need the larger lane.
