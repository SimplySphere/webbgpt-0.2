# WebbGPT 3B/7B Scaling Roadmap

This roadmap treats the 22M local-MVP run as complete. The final local-MVP recipe is useful as a data-quality and training-plumbing baseline, but its checkpoint is not a base for serious SFT: it improved perplexity and cleaned up contamination, while still failing raw LM quality through topic drift and semantic repetition.

Core principle: base pretraining should build general language ability from curated real data. Webb-specific behavior should mostly come later through SFT, RAG/grounding, and possibly DPO. Generated custom corpora should not become a main pretraining source, and Webb/domain-heavy text should not be a major base-pretraining share.

## Recommendation

Run 3B before 7B. The 3B track is the right place to validate the scaled data pipeline, exact and near-duplicate reporting, step/token accounting, checkpoint size, raw-LM eval gates, and SFT readiness criteria. Move to 7B only after a 3B pilot or meaningful run produces readable, topic-stable generations and the manifest reports look clean.

For product progress, continued pretraining from an existing capable base is the practical path. For owning the full stack and testing WebbGPT architecture/tokenizer choices, train from scratch, but expect the first useful from-scratch checkpoint to require tens of billions of tokens, not hundreds of millions.

## Existing Config Assessment

- `sample-configs/model-3b.json` is usable as a 3B-class starter. The current parameter estimate is about 3.38B with tied embeddings and grouped-query attention.
- `sample-configs/model-7b.json` was a placeholder in practice: with 8 key/value heads it estimated at about 5.88B parameters. It has been updated to 32 key/value heads, which brings the estimate to about 6.68B parameters while keeping the 4096 context length.
- `sample-configs/train-3b.json` and `sample-configs/train-7b.json` are broad placeholders. Use the tiered configs added here instead.
- `sample-configs/data-3b.json` and `sample-configs/data-7b.json` are legacy mixed-purpose configs with posttrain and synthetic references. Use `data-3b-curated.json` and `data-7b-curated.json` for base-pretraining scale work.

New starter configs:

- `sample-configs/data-3b-curated.json`
- `sample-configs/train-3b-smoke.json`
- `sample-configs/train-3b-pilot.json`
- `sample-configs/train-3b-serious.json`
- `sample-configs/data-7b-curated.json`
- `sample-configs/train-7b-smoke.json`
- `sample-configs/train-7b-pilot.json`

The data configs default to the high end of pilot budgets so accidental preparation is bounded: 5B tokens for 3B and 10B tokens for 7B. Before preparing serious data, explicitly change `pretraining_token_budget` to the matching tier budget.

## Track A: Base Pretraining Scale Plan

### From Scratch vs Continued Pretraining

From scratch means initializing WebbGPT weights randomly and learning general language modeling from curated real data. It is useful when the goal is to validate WebbGPT's own architecture, tokenizer, data filters, and training system. It is expensive, and small token runs mainly validate plumbing and loss curves.

Continued pretraining means starting from an existing capable base model and adapting it on a smaller curated corpus. It is the recommended path if the near-term goal is a usable Webb assistant, because general language ability already exists and Webb behavior can then be added through SFT and RAG.

Do not use the 22M local-MVP checkpoint as a meaningful SFT base. It can still be used for plumbing tests.

### 3B From-Scratch Budgets

| Tier | Tokens | Purpose |
| --- | ---: | --- |
| Smoke | 100M-500M | Prove training launches, loss decreases, manifests are sane, eval hooks work. Do not expect usable generations. |
| Pilot | 1B-5B | Validate throughput, checkpoint size, raw generation trend, data-source balance, and continuation of loss curves. |
| Meaningful | 10B-30B | First range where a 3B from-scratch model may become readable enough to judge topic retention. |
| Serious | 30B-60B | Minimum serious from-scratch 3B attempt for raw LM legibility and SFT-readiness evaluation. |

Starter configs use 500M for smoke, 5B for pilot, and 60B for serious.

### 7B From-Scratch Budgets

| Tier | Tokens | Purpose |
| --- | ---: | --- |
| Smoke | 500M-1B | Prove distributed memory, optimizer state, checkpointing, eval, and data loading. |
| Pilot | 3B-10B | Validate throughput and quality trend before committing to a large run. |
| Meaningful | 25B-70B | First range where a 7B from-scratch model may show broadly readable raw LM behavior. |
| Serious | 70B-140B | Minimum serious from-scratch 7B attempt. Do not start here before 3B has de-risked the stack. |

Starter configs use 1B for smoke and 10B for pilot. Add a 7B serious config only after the 3B serious path and 7B pilot have clean reports.

### Continued-Pretraining Budgets

Use much smaller adaptation budgets when starting from an existing base:

| Goal | Tokens |
| --- | ---: |
| Plumbing and compatibility | 100M-300M |
| Light style/domain adaptation | 300M-1B |
| Stronger curriculum adaptation | 1B-3B |
| Heavy adaptation with regression risk | 3B-5B |

Keep continued pretraining broad and real-data-first. Do not turn it into generated Webb pretraining. If Webb-specific knowledge is needed, prefer RAG and SFT grounded examples.

## Data Scaling Requirements

Scale the local-MVP curated-real-data recipe. The required controls are:

- Exact dedupe by document identity/hash.
- Near-dedupe reporting, with rejection added before serious runs if ratios remain high.
- Boilerplate and page chrome removal.
- Navigation/menu/list-heavy rejection.
- Medical, product, and commercial density controls.
- Repeated n-gram controls.
- URL and metadata-heavy filtering.
- Document shape filters for length, sentence structure, and fragment quality.
- No giant synthetic pretraining corpora.
- No major Webb/domain share in base pretraining.

Every prepared run should keep manifest reporting for:

- Total tokens and packed sequence counts.
- Source and family shares.
- Rejection reasons.
- Near-duplicate ratios and cluster sizes.
- Repeated n-grams.
- Artifact densities.
- Document length and shape stats.

Use:

```bash
python3.12 tools/pretrain_manifest_report.py artifacts/runs/<run>/prepared/pretrain.json
python3.12 tools/curated_pretrain_data_audit.py --data-config sample-configs/data-3b-curated.json
```

## Disk, Runtime, and Memory Implications

Prepared LM shards are saved as int32 arrays, so the packed-token floor is about 4 bytes per token before metadata. Plan for 10-30 percent overhead for manifests, metadata, caches, and staging, and keep extra space for checkpoints.

Approximate prepared-data disk floor:

| Token budget | Packed shard floor |
| ---: | ---: |
| 100M | 0.4 GB |
| 500M | 2 GB |
| 1B | 4 GB |
| 5B | 20 GB |
| 10B | 40 GB |
| 30B | 120 GB |
| 60B | 240 GB |
| 70B | 280 GB |
| 140B | 560 GB |

Checkpoint estimates from current parameter counts:

- 3B config: about 3.38B parameters. Model-only bf16 export is roughly 7 GB. Full training checkpoints with optimizer state are likely 40-70 GB each.
- 7B config: about 6.68B parameters. Model-only bf16 export is roughly 14 GB. Full training checkpoints with optimizer state are likely 80-140 GB each.

The 3B/7B pretraining starter configs are 8-GPU recommended. The standard configs assume 8 distributed ranks, and this repo validates:

```text
global_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
```

Before launching scale pretraining, run `tools/prepare_scale_launch.py` on the GPU machine. It inspects visible CUDA GPUs, prints the `torchrun` command, and does not start training. If exactly 8 CUDA GPUs are visible, it uses the standard train config as-is. If non-8 GPU hardware is visible, pass `--allow-auto-config`; the helper writes a run-specific train config under `artifacts/runs/...` and adjusts `gradient_accumulation_steps` to preserve effective global batch size where possible.

Example 3B 100M smoke dry run:

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

If the GPU count changes manually, update `global_batch_size` or `gradient_accumulation_steps` before launching.

Approximate optimizer-step counts from the starter configs:

| Config | Sequence length | Global batch | Tokens/step | Budget | Approx steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train-3b-smoke.json` | 8192 | 64 | 0.52M | 500M | 954 |
| `train-3b-pilot.json` | 8192 | 128 | 1.05M | 5B | 4,768 |
| `train-3b-serious.json` | 8192 | 256 | 2.10M | 60B | 28,610 |
| `train-7b-smoke.json` | 4096 | 128 | 0.52M | 1B | 1,908 |
| `train-7b-pilot.json` | 4096 | 256 | 1.05M | 10B | 9,537 |

Wall-clock runtime is cluster-specific. Use:

```text
runtime_seconds = token_budget / sustained_tokens_per_second
```

Examples: 10B tokens takes about 55.6 hours at 50k tokens/sec and about 27.8 hours at 100k tokens/sec. 60B tokens takes about 13.9 days at 50k tokens/sec and about 6.9 days at 100k tokens/sec.

Memory assumptions:

- 3B smoke/pilot: target 8x80GB for the current 8192 context. 8x40GB may require smaller sequence length, lower microbatch, or more aggressive sharding checks.
- 3B serious: 8x80GB minimum practical target; 16x80GB gives more room for throughput and failures.
- 7B smoke/pilot: target 8x80GB with 4096 context. Do not start a 7B serious run without first proving memory and checkpoint throughput.
- Local MPS/CPU should only be used for config validation and tiny plumbing tests, not scale training.

## Operating Plan

1. Run no more than 3B smoke first. Validate manifest quality, loss decrease, final eval, checkpoint size, and raw samples.
2. If smoke is clean, run 3B pilot at 1B-5B tokens. Use the high end only if throughput and disk are comfortable.
3. Run 3B meaningful at 10B-30B if pilot samples improve and data reports remain clean.
4. Run 3B serious at 30B-60B only after a meaningful run shows readable raw generation.
5. Start 7B smoke only after 3B pilot or meaningful proves the data/training system.
6. Start 7B pilot only after 7B smoke validates distributed memory and checkpoints.

## What Reuses Local-MVP Work

Reuse:

- The curated real-data pretraining principle.
- The lower-LR, one-prepared-pass discipline.
- Full validation and raw LM family holdouts.
- Provenance logging and low-loss contamination checks.
- Manifest reporting with source shares, rejection reasons, duplicate stats, artifact densities, and document shape stats.
- Local real corpora as small seed/validation sources, not as the whole scaled corpus.

Do not reuse:

- The 22M checkpoint as a serious SFT base.
- Generated domain corpora for base pretraining.
- Webb/domain-heavy corpora as a major base-pretraining source.
- Local-MVP repeat allowances as a default for scale. The new curated configs set `lm_max_source_repeat_rate` to `0.0`.
