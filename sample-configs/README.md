# Sample Configs

This directory keeps active configs flat for quick demo use. Legacy DPO and failed
older SFT experiment configs were moved to `junk/` so the active path is easier to
scan.

## Local-MVP

Core 22M local-MVP pretraining and evaluation configs:

- `model-local-mvp.json` - 22M local-MVP model shape.
- `data-local-mvp.json` - local-MVP data sources for the curated pretrain profile.
- `data-local-mvp-prepared.json` - prepared-data variant for local-MVP runs.
- `train-local-mvp.json` - main local-MVP pretrain run.
- `train-local-mvp-sanity-probe.json` - short local-MVP sanity/probe training run.
- `eval-local-mvp.json` - local-MVP evaluation config.

## Continued Pretraining

Small bounded continued pretraining from reviewed source material. These configs start
from the pretrained local-MVP checkpoint and write to
`artifacts/runs/local-mvp/checkpoints/continue-small/`.

- `data-local-mvp-continue-small.json` - source-material data config using `data/source_material/continued_pretrain_candidates/`.
- `data-local-mvp-continue-small-prepared.json` - prepared-manifest variant for the continued run.
- `train-local-mvp-continue-small.json` - bounded continued-pretraining train config.

Do not use `data/source_material/rejected_template_heavy/` or
`data/source_material/needs_manual_review/` directly for continued pretraining.

## SFT And Grounded Posttraining

Local-MVP posttraining and plumbing configs. These are deadline-safe local
posttraining experiments, not evidence that the 22M model is a polished assistant.

- `data-local-mvp-sft0.json` - tiny SFT-0 plumbing data config.
- `train-local-mvp-sft0-plumbing.json` - tiny SFT-0 plumbing training run.
- `data-local-mvp-sft-v3.json` - safer SFT-v3 data config kept for analysis and possible future grounded SFT work.
- `train-local-mvp-sft-v3.json` - bounded SFT-v3 train config kept as the latest SFT experiment.
- `data-local-mvp-sft-rag-v1.json` - first context-grounded SFT data config built from RAG chunks.
- `train-local-mvp-sft-rag-v1.json` - bounded SFT-RAG train config. It should be run only after RAG data is rebuilt and audited.

Older SFT-small and SFT-v2 configs are archived under `junk/legacy-sft-experiments/`.

## DPO Legacy

DPO is no longer active for the final path. The local-MVP DPO-small configs and
preference data were moved to `junk/dpo-legacy/` after qualitative samples worsened.
`src/cli.py train-dpo` is retained only as a disabled legacy command.

## 3B

3B model, data, training, and evaluation configs. These are not expected to train on
the local Mac before the Thursday demo.

- `model-3b.json` - 3B model shape.
- `data-3b.json` - broad 3B data config.
- `data-3b-curated.json` - curated 3B data config.
- `data-3b-curated-smoke-100m.json` - curated 100M-token smoke data config.
- `train-3b.json` - main 3B training config.
- `train-3b-pilot.json` - pilot-scale 3B training config.
- `train-3b-smoke.json` - short 3B smoke run.
- `train-3b-smoke-100m.json` - 100M-token 3B smoke run.
- `train-3b-config-check.json` - lightweight 3B config validation run.
- `train-3b-serious.json` - larger serious 3B run config.
- `eval-3b.json` - 3B evaluation config.

## 7B

7B model, data, training, and evaluation configs. Treat these as remote/GPU-oriented
configs, not local Mac deadline candidates.

- `model-7b.json` - 7B model shape.
- `data-7b.json` - broad 7B data config.
- `data-7b-curated.json` - curated 7B data config.
- `train-7b.json` - main 7B training config.
- `train-7b-pilot.json` - pilot-scale 7B training config.
- `train-7b-smoke.json` - short 7B smoke run.
- `eval-7b.json` - 7B evaluation config.

## Tokenizer

Tokenizer and tokenizer-corpus configs:

- `tokenizer.json` - default tokenizer config.
- `tokenizer-local-mvp.json` - local-MVP tokenizer config.
- `tokenizer-7b.json` - 7B tokenizer config.
- `tokenizer-corpus.json` - default tokenizer corpus config.
- `tokenizer-corpus-local-mvp.json` - local-MVP tokenizer corpus config.
- `tokenizer-corpus-7b.json` - 7B tokenizer corpus config.

## Serving

Serving configs for local and remote/demo modes:

- `serve-local-mvp.json` - local-MVP website/API demo config. It supports optional RAG through `WEBBGPT_USE_RAG`, `WEBBGPT_RAG_INDEX`, and `WEBBGPT_RAG_CHUNKS`; it also carries the safer demo decode defaults and stricter RAG thresholds.
- `serve-3b.json` - 3B serving config.
- `serve-7b.json` - 7B serving config.
- `serve-debug.json` - debug serving config.

## Debug

Small debug configs used for tests and fast local checks:

- `model-debug.json` - tiny debug model shape.
- `data-debug.json` - tiny debug data config.
- `train-debug.json` - tiny debug training config.
- `eval-debug.json` - debug evaluation config.
