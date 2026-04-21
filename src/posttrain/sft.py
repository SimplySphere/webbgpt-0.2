from __future__ import annotations

import math
import sys
from pathlib import Path

from config import DataConfig, ModelConfig, TrainConfig
from data.dataset import DatasetBuilder
from model.transformer import CausalTransformer
from posttrain.eval import (
    assess_sample_behavior,
    ensure_no_regression_prompt_overlap,
    generate_qualitative_samples,
)
from repro import seed_everything
from train.checkpoint import CheckpointManager, resolve_parent_lineage
from train.console import print_lm_train_event, print_sft_eval_event, simplify_samples
from train.distributed import cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.entrypoints import build_stage_data_fingerprint, snapshot_configs
from train.loop import EvalControl, build_dataloader, evaluate_language_model, run_training, save_run_metadata, save_stage_summary
from train.optim import build_optimizer, build_scheduler


def _apply_sft_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.sft_learning_rate is not None:
        stage_config.learning_rate = stage_config.sft_learning_rate
    if stage_config.sft_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.sft_min_learning_rate
    if stage_config.sft_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.sft_warmup_steps
    if stage_config.sft_max_steps is not None:
        stage_config.max_steps = stage_config.sft_max_steps
    return stage_config


def _evaluate_sft_validation(model, dataloader, _max_batches: int | None) -> dict[str, float]:
    metrics = evaluate_language_model(model, dataloader, None)
    metrics["batches_evaluated"] = len(dataloader)
    metrics["examples_evaluated"] = len(dataloader.dataset)
    return metrics


def _compute_sft_schedule(
    *,
    train_loader_steps: int,
    stage_config: TrainConfig,
) -> tuple[int, int, int, int]:
    steps_per_epoch = max(
        1,
        math.ceil(train_loader_steps / max(stage_config.gradient_accumulation_steps, 1)),
    )
    effective_max_steps = stage_config.max_steps
    if stage_config.sft_max_epochs is not None and stage_config.sft_max_epochs > 0:
        effective_max_steps = min(effective_max_steps, steps_per_epoch * stage_config.sft_max_epochs)
    eval_interval = max(
        max(stage_config.sft_min_eval_interval_steps, 1),
        math.ceil(steps_per_epoch / max(stage_config.sft_evals_per_epoch, 1)),
    )
    early_eval_step = min(10, eval_interval)
    return steps_per_epoch, effective_max_steps, eval_interval, early_eval_step


def _should_include_sft_samples(
    *,
    step: int,
    final_eval: bool,
    best_eval_step: int,
    sample_every_steps: int,
) -> bool:
    if step == 0 or final_eval or step == best_eval_step:
        return True
    return sample_every_steps > 0 and step > 0 and step % sample_every_steps == 0


def _is_severe_sft_qualitative_failure(sample_behavior: dict[str, int | bool | list[str]]) -> bool:
    return bool(
        int(sample_behavior.get("blank_count", 0)) > 0
        or int(sample_behavior.get("generic_refusal_count", 0)) > 0
        or int(sample_behavior.get("repetitive_count", 0)) >= 2
        or int(sample_behavior.get("wrong_source_attribution_count", 0)) > 0
        or int(sample_behavior.get("unsupported_source_tag_count", 0)) >= 2
    )


def _sft_grounded_stop_warmup_step(*, steps_per_epoch: int) -> int:
    return max(1, math.ceil(max(steps_per_epoch, 1) * 0.5))


def _should_count_sft_collapse_gate_hit(
    *,
    step: int,
    steps_per_epoch: int,
    sample_behavior: dict[str, int | bool | list[str]],
) -> bool:
    if step <= 0 or not bool(sample_behavior.get("collapse_detected")):
        return False
    if _is_severe_sft_qualitative_failure(sample_behavior):
        return True
    if int(sample_behavior.get("grounded_abstention_fail_count", 0)) > 0:
        return step >= _sft_grounded_stop_warmup_step(steps_per_epoch=steps_per_epoch)
    return True


def run_sft_job(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    stage_config = _apply_sft_overrides(train_config)
    builder = DatasetBuilder(data_config)
    trust_blockers: list[str] = []
    collapse_gate_hits = 0
    train_dataset, validation_dataset = builder.build_sft_split(
        seed=stage_config.seed,
        validation_fraction=stage_config.sft_validation_fraction,
        validation_min_examples=stage_config.sft_validation_min_examples,
        allow_weak_validation=stage_config.allow_weak_posttrain_validation,
        require_explicit_validation=stage_config.require_explicit_sft_validation,
    )
    train_examples = getattr(train_dataset, "examples", None)
    validation_examples = getattr(validation_dataset, "examples", None)
    if train_examples is not None and validation_examples is not None:
        ensure_no_regression_prompt_overlap(
            stage_name="sft",
            train_examples=train_examples,
            validation_examples=validation_examples,
        )
    elif is_main_process():
        trust_blockers.extend(["behavior_eval_untrusted", "overlap_guard_skipped"])
        print(
            "WebbGPT: skipping SFT regression-prompt overlap guard because prepared datasets do not expose raw prompt metadata in v1.",
            file=sys.stderr,
            flush=True,
        )
    if stage_config.checkpoint.initialize_from and stage_config.checkpoint.resume_from:
        raise ValueError("Set either checkpoint.initialize_from or checkpoint.resume_from, not both.")
    init_distributed()
    try:
        seed_bundle = seed_everything(stage_config.seed)
        train_loader = build_dataloader(train_dataset, batch_size=stage_config.micro_batch_size, shuffle=True)
        val_loader = None
        if validation_dataset is not None and len(validation_dataset) > 0:
            val_loader = build_dataloader(
                validation_dataset, batch_size=stage_config.micro_batch_size, shuffle=False
            )
        configured_max_steps = stage_config.max_steps
        steps_per_epoch, effective_max_steps, eval_interval, early_eval_step = _compute_sft_schedule(
            train_loader_steps=len(train_loader),
            stage_config=stage_config,
        )
        stage_config.max_steps = effective_max_steps
        if is_main_process():
            if val_loader is not None:
                print(
                    "WebbGPT: sft schedule "
                    f"(steps_per_epoch={steps_per_epoch}, "
                    f"configured_max_steps={configured_max_steps}, "
                    f"effective_max_steps={effective_max_steps}, "
                    f"eval_interval={eval_interval}, "
                    f"early_eval_step={early_eval_step}, "
                    f"sample_every_steps={stage_config.sft_sample_every_steps}).",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    "WebbGPT: sft schedule "
                    f"(steps_per_epoch={steps_per_epoch}, "
                    f"configured_max_steps={configured_max_steps}, "
                    f"effective_max_steps={effective_max_steps}, "
                    "validation=disabled).",
                    file=sys.stderr,
                    flush=True,
                )
        model = CausalTransformer(model_config)
        model = maybe_wrap_fsdp(model, stage_config)
        checkpoint_manager = CheckpointManager(
            output_dir=stage_config.checkpoint.output_dir,
            keep_last_n=stage_config.checkpoint.keep_last_n,
        )
        if stage_config.checkpoint.initialize_from and not stage_config.checkpoint.resume_from:
            checkpoint_manager.load(stage_config.checkpoint.initialize_from, model, strict=True)
        optimizer = build_optimizer(model, stage_config)
        scheduler = build_scheduler(optimizer, stage_config)
        if is_main_process():
            snapshot_configs(model_config, data_config, stage_config)
            save_run_metadata(
                stage_config,
                stage_config.checkpoint.output_dir,
                extra={
                    "seed_bundle": seed_bundle,
                    "sft_schedule": {
                        "configured_max_steps": configured_max_steps,
                        "effective_max_steps": effective_max_steps,
                        "steps_per_epoch": steps_per_epoch,
                        "max_epochs": stage_config.sft_max_epochs,
                        "eval_interval_steps": eval_interval,
                        "early_eval_step": early_eval_step,
                        "min_eval_interval_steps": stage_config.sft_min_eval_interval_steps,
                        "sample_every_steps": stage_config.sft_sample_every_steps,
                    },
                    "validation_policy": {
                        "require_explicit_validation": stage_config.require_explicit_sft_validation,
                        "validation_min_examples": stage_config.sft_validation_min_examples,
                        "allow_weak_posttrain_validation": stage_config.allow_weak_posttrain_validation,
                    },
                },
            )
        eval_history_path = Path(stage_config.checkpoint.output_dir) / "eval_history.jsonl"
        parent_lineage = resolve_parent_lineage(
            stage_config.checkpoint.resume_from or stage_config.checkpoint.initialize_from,
            nominal_parent_stage="continue",
        )
        checkpoint_metadata = {
            "stage": "sft",
            **parent_lineage,
            "input_data_fingerprint": build_stage_data_fingerprint(data_config, "sft"),
            "artifact_status": "dev_only" if trust_blockers else "promotable",
            "promotion_blockers": trust_blockers,
            "promotion_eligible": not trust_blockers,
        }

        def _eval_payload_callback(model, step: int, final_eval: bool, state, _metrics):
            nonlocal collapse_gate_hits
            if not _should_include_sft_samples(
                step=step,
                final_eval=final_eval,
                best_eval_step=state.best_eval_step,
                sample_every_steps=stage_config.sft_sample_every_steps,
            ):
                return {"samples": []}
            samples = generate_qualitative_samples(
                model,
                data_config.tokenizer_path,
                regression_path="data/eval/posttrain_regression.jsonl",
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
            )
            sample_behavior = assess_sample_behavior(samples)
            if _should_count_sft_collapse_gate_hit(
                step=step,
                steps_per_epoch=steps_per_epoch,
                sample_behavior=sample_behavior,
            ):
                collapse_gate_hits += 1
                for blocker in sample_behavior["promotion_blockers"]:
                    if blocker not in trust_blockers:
                        trust_blockers.append(blocker)
                if "sft_behavior_collapse" not in trust_blockers:
                    trust_blockers.append("sft_behavior_collapse")
            elif step > 0 and sample_behavior["collapse_detected"]:
                collapse_gate_hits = 0
                for blocker in sample_behavior["promotion_blockers"]:
                    if blocker not in trust_blockers:
                        trust_blockers.append(blocker)
                if "sft_behavior_collapse" not in trust_blockers:
                    trust_blockers.append("sft_behavior_collapse")
            else:
                collapse_gate_hits = 0
            return {
                "samples": simplify_samples(samples),
                "sample_behavior": sample_behavior,
                "should_stop_training": collapse_gate_hits >= 2,
            }

        state = run_training(
            model=model,
            train_loader=train_loader,
            train_config=stage_config,
            checkpoint_manager=checkpoint_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loader=val_loader,
            resume_from=stage_config.checkpoint.resume_from,
            best_checkpoint_name="best" if val_loader is not None else None,
            eval_payload_callback=_eval_payload_callback if val_loader is not None else None,
            eval_fn=_evaluate_sft_validation if val_loader is not None else None,
            eval_control=(
                EvalControl(
                    stage_name="sft",
                    evaluate_at_start=True,
                    early_eval_step=early_eval_step,
                    eval_interval_steps=eval_interval,
                    validation_max_batches=None,
                    best_min_delta=stage_config.sft_best_min_delta,
                    early_stopping_patience_evals=stage_config.sft_early_stopping_patience_evals,
                    overfit_train_loss_threshold=0.05,
                    overfit_worsening_patience=2,
                    train_dataset_size=len(train_dataset),
                    validation_dataset_size=len(validation_dataset),
                    steps_per_epoch=steps_per_epoch,
                    eval_history_path=str(eval_history_path),
                )
                if val_loader is not None
                else None
            ),
            save_final_checkpoint=True,
            train_event_printer=print_lm_train_event,
            eval_event_printer=print_sft_eval_event if val_loader is not None else None,
            checkpoint_metadata=checkpoint_metadata,
        )
        if is_main_process():
            blockers = list(trust_blockers)
            if state.nonfinite_loss_steps > 0:
                blockers.append("nonfinite_loss_seen")
            artifact_status = "dev_only" if blockers else "promotable"
            save_stage_summary(
                stage_config.checkpoint.output_dir,
                {
                    "stage": "sft",
                    **parent_lineage,
                    "input_data_fingerprint": build_stage_data_fingerprint(data_config, "sft"),
                    "artifact_status": artifact_status,
                    "promotion_blockers": blockers,
                    "promotion_eligible": artifact_status == "promotable",
                    "tokens_seen": state.tokens_seen,
                    "examples_seen": state.examples_seen,
                    "best_eval_loss": None if state.best_eval_loss == float("inf") else state.best_eval_loss,
                    "best_eval_step": None if state.best_eval_step < 0 else state.best_eval_step,
                    "nonfinite_loss_steps": state.nonfinite_loss_steps,
                    "validation_enabled": val_loader is not None,
                },
            )
    finally:
        cleanup_distributed()
