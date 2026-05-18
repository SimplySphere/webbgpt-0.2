# 3B Smoke and SFT-0 Checklist

Do not start training until the checklist is explicitly approved. This file separates local config/data checks from GPU-cluster training.

## 3B Config Audit

Current 3B model:

- Config: `sample-configs/model-3b.json`
- Parameter estimate: `3,375,565,824` parameters, about `3.376B`
- Model-only bf16 size: about `6.29 GiB`
- Full training checkpoint estimate with optimizer state: about `38-70 GiB` per checkpoint, depending on optimizer/checkpoint payload details
- Context: `8192`
- Hidden size: `3072`
- Layers: `32`
- Attention heads: `24`
- Key/value heads: `8`
- Vocab size: `50176`

Current curated 3B data config:

- Config: `sample-configs/data-3b-curated.json`
- Tokenizer path: `artifacts/tokenizer/webbgpt.model`
- Sequence length: `8192`
- Default token budget: `5,000,000,000`, which is a pilot budget, not the first smoke-prep target
- Source target mix: 80 percent FineWeb-Edu, 10 percent local-MVP curated real prose, 10 percent local FineWeb extension prose
- Webb/domain pretraining share: none
- Synthetic pretraining corpora: none

Current 3B smoke train config:

- Config: `sample-configs/train-3b-smoke.json`
- Token budget: `500,000,000`
- Global batch size: `64`
- Microbatch: `1`
- Gradient accumulation: `8`
- World size assumption: `8`, because `1 * 8 * 8 = 64`
- Tokens per optimizer step at sequence length 8192: about `524,288`
- Approximate optimizer steps for 500M tokens: about `954`

The 3B smoke train config is not local-safe. It is a real distributed smoke config and will fail batch-size validation on a single process unless `global_batch_size` is changed. It also instantiates the full 3B model, so it is not appropriate for local CPU/MPS training.

Config-only helper:

- `sample-configs/train-3b-config-check.json` is for dataclass/config loading and tiny path-plumbing checks only.
- It is not a real 3B training recipe.
- Even this config should not be used to instantiate the 3B model on a small local machine unless memory has been verified.

100M cluster smoke helper:

- `sample-configs/train-3b-smoke-100m.json` matches the 100M prepared-data target.
- It keeps the 8-rank batch assumption from `train-3b-smoke.json`.
- It is still a real 3B distributed training config, not a local test config.

Generated profile status:

- `src/cli.py` still maps the `remote-3b` generated profile to `model-3b.json`, `data-3b.json`, and `train-3b.json`.
- Those generated 3B profile files are legacy broad placeholders.
- Use `data-3b-curated-smoke-100m.json` plus the tiered train configs for this roadmap.

Missing or external paths:

- `artifacts/tokenizer/webbgpt.model` is required for 3B data prep and is currently not present in this local checkout.
- `data/raw/tokenizer_corpus.txt` and `data/raw/tokenizer_corpus_local_mvp.txt` exist locally.
- `HuggingFaceFW/fineweb-edu` is external and requires network/cache access.
- `data/eval/pretrain_general_regression.jsonl` exists.
- `data/eval/pretrain_family_holdouts_general.json` exists.

## 100M 3B Smoke Data Prep

Use the 100M config first:

- Config: `sample-configs/data-3b-curated-smoke-100m.json`
- Token budget: `100,000,000`
- Expected packed shard floor: about `0.4 GB`
- Practical disk expectation with metadata/cache/staging: reserve at least `1-2 GB` for prepared artifacts, plus dataset cache space for FineWeb-Edu streaming/cache

Build the 3B tokenizer if `artifacts/tokenizer/webbgpt.model` is missing:

```bash
webbgpt build-tokenizer-corpus --config sample-configs/tokenizer-corpus.json
webbgpt tokenize \
  --config sample-configs/tokenizer.json \
  --input data/raw/tokenizer_corpus.txt
```

Prepare the 100M pretrain and validation manifests:

```bash
webbgpt prepare-data \
  --config sample-configs/data-3b-curated-smoke-100m.json \
  --stage pretrain \
  --output artifacts/runs/scale-3b-smoke-100m/prepared/pretrain.json \
  --force-rebuild \
  --preprocessing-num-workers 8 \
  --tokenizer-num-workers 8

webbgpt prepare-data \
  --config sample-configs/data-3b-curated-smoke-100m.json \
  --stage validation \
  --output artifacts/runs/scale-3b-smoke-100m/prepared/validation.json \
  --force-rebuild \
  --preprocessing-num-workers 8 \
  --tokenizer-num-workers 8
```

Inspect the manifests:

```bash
python3.12 tools/pretrain_manifest_report.py \
  artifacts/runs/scale-3b-smoke-100m/prepared/pretrain.json

python3.12 tools/pretrain_manifest_report.py \
  artifacts/runs/scale-3b-smoke-100m/prepared/validation.json

python3.12 tools/curated_pretrain_data_audit.py \
  --data-config sample-configs/data-3b-curated-smoke-100m.json \
  --manifest artifacts/runs/scale-3b-smoke-100m/prepared/pretrain.json \
  --output-dir artifacts/runs/scale-3b-smoke-100m/reports/data_audit
```

Required gates before any 3B smoke training:

- Pretrain manifest reaches about 100M packed tokens.
- Validation manifest is present and non-empty.
- Source shares are close to the 80/10/10 target, or any capacity-limited deviation is explained in the manifest.
- No Webb/domain-heavy or generated pretraining source appears.
- Exact dedupe is enabled on all sources.
- Near-duplicate ratios and largest clusters are reviewed.
- Top rejection reasons are dominated by expected quality filters, not a broken parser.
- Artifact densities stay below configured thresholds.
- Repeated n-gram report does not show dominant boilerplate/page chrome.
- Document shape stats remain plausible for prose.

## 3B Smoke Training Gate

Only after the 100M data prep gates pass, run the dry-run launch helper on the GPU machine. It prints the command and does not start training:

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

On an 8-GPU machine, the printed cluster command shape should be:

```bash
torchrun --nproc_per_node=8 src/cli.py train-pretrain \
  --model-config sample-configs/model-3b.json \
  --data-config sample-configs/data-3b-curated-smoke-100m.json \
  --train-config sample-configs/train-3b-smoke-100m.json
```

## SFT-0 Plumbing Plan

SFT-0 uses the weak 22M local-MVP checkpoint only to test the SFT path. It must not be used to claim quality.

Current local SFT fixtures:

- `data/local/sft.jsonl`: 7 examples
- `data/local/sft_validation.jsonl`: 6 validation examples
- `data/local/sft_grounding_guardrails.jsonl`: 134 examples
- Total SFT-0 train examples in `data-local-mvp-sft0.json`: 141

SFT-0 configs:

- `sample-configs/data-local-mvp-sft0.json`
- `sample-configs/train-local-mvp-sft0-plumbing.json`
- Parent checkpoint: `artifacts/runs/local-mvp/checkpoints/pretrain/best-pretrain`
- Tokenizer: `artifacts/tokenizer/webbgpt-local-mvp.model`

Prepare SFT-0 train artifacts for formatting and label-mask inspection:

```bash
webbgpt prepare-data \
  --config sample-configs/data-local-mvp-sft0.json \
  --stage sft \
  --output artifacts/runs/sft0-plumbing/prepared/sft.json \
  --force-rebuild
```

The explicit validation file is consumed by `train-sft` through `sft_validation_sources`; the standalone `prepare-data` CLI does not expose a separate `sft_validation` stage.

Checks to perform before running SFT-0:

- JSONL parses cleanly and every row has `messages`.
- Every training row has at least one assistant message.
- Label masking supervises assistant spans only.
- Validation split is explicit and non-empty.
- Parent checkpoint loads from the 22M local-MVP checkpoint.
- Checkpoint save/load works after a tiny run.
- Sample generation and eval reporting emit records.

When approved, the SFT-0 plumbing command is:

```bash
webbgpt train-sft \
  --model-config sample-configs/model-local-mvp.json \
  --data-config sample-configs/data-local-mvp-sft0.json \
  --train-config sample-configs/train-local-mvp-sft0-plumbing.json
```

Do not start DPO. Do not start serious SFT. Do not promote the SFT-0 output as a quality model.

## Local vs Cluster

Can be done locally:

- Config parsing.
- Path existence checks.
- SFT JSONL format inspection.
- Tiny SFT-0 prepare-data artifacts with the local-MVP tokenizer.
- Possibly SFT-0 plumbing on the 22M model, if explicitly approved.

Requires network or dataset cache:

- Building the 3B tokenizer corpus from FineWeb-Edu.
- Preparing the 100M 3B curated smoke corpus with FineWeb-Edu.

Requires GPU cluster:

- Any real 3B training.
- The current `train-3b-smoke.json` distributed smoke run.
- 500M-token 3B smoke and larger pilot/serious runs.
