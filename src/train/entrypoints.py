from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from config import DataConfig, ModelConfig, TrainConfig, save_config
from data.dataset import DatasetBuilder
from data.prepared import PreparedPackedDataset
from model.transformer import CausalTransformer
from posttrain.eval import (
    PRETRAIN_QUALITATIVE_RUBRIC,
    assess_raw_lm_sample_behavior,
    evaluate_pretrain_family_holdouts,
    generate_qualitative_samples,
    generate_raw_lm_qualitative_samples,
    raw_lm_quality_status,
)
from repro import seed_everything
from train.checkpoint import CheckpointManager
from train.console import print_lm_eval_event, print_lm_train_event, simplify_samples
from train.distributed import cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.loop import EvalControl, TrainingRunControl, build_dataloader, run_training, save_run_metadata, save_stage_summary
from train.optim import build_optimizer, build_scheduler


def snapshot_configs(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    config_dir = Path(train_config.checkpoint.output_dir) / "configs"
    save_config(model_config, config_dir / "model.json")
    save_config(data_config, config_dir / "data.json")
    save_config(train_config, config_dir / "train.json")


PRETRAIN_PROBE_PATH = "data/eval/pretrain_regression.jsonl"
CONTINUE_PROBE_PATH = "data/eval/continue_regression.jsonl"


def _effective_batch_size(train_config: TrainConfig, world_size: int) -> int:
    return int(train_config.micro_batch_size * train_config.gradient_accumulation_steps * world_size)


def _one_pass_optimizer_steps(
    *,
    dataset_size: int,
    effective_batch_size: int,
    flush_final_partial: bool,
) -> int:
    if effective_batch_size <= 0:
        return 0
    if flush_final_partial:
        return (dataset_size + effective_batch_size - 1) // effective_batch_size
    return dataset_size // effective_batch_size


def _final_partial_microbatches(
    *,
    dataset_size: int,
    effective_batch_size: int,
    micro_batch_size: int,
) -> int:
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        return 0
    remainder_examples = dataset_size % effective_batch_size
    if remainder_examples == 0:
        return 0
    return (remainder_examples + micro_batch_size - 1) // micro_batch_size


def _pretrain_run_control(
    dataset,
    train_config: TrainConfig,
    *,
    stage_name: str,
    world_size: int,
) -> TrainingRunControl:
    if stage_name != "pretrain":
        return TrainingRunControl(run_mode="max_steps_limited", progress_mode="steps")
    prepared_token_target = None
    if isinstance(dataset, PreparedPackedDataset):
        prepared_token_target = int(dataset.manifest.get("num_tokens", 0) or 0)
    prepared_sequence_target = len(dataset) if hasattr(dataset, "__len__") else None
    stop_mode = str(train_config.pretrain_stop_mode)
    progress_mode = str(train_config.pretrain_progress_mode)
    if stop_mode == "one_prepared_pass" and isinstance(dataset, PreparedPackedDataset):
        effective_batch_size = _effective_batch_size(train_config, world_size)
        effective_optimizer_steps = _one_pass_optimizer_steps(
            dataset_size=len(dataset),
            effective_batch_size=effective_batch_size,
            flush_final_partial=train_config.pretrain_flush_final_partial_accumulation,
        )
        return TrainingRunControl(
            run_mode="one_prepared_pass",
            progress_mode=progress_mode,
            prepared_token_target=prepared_token_target,
            prepared_sequence_target=prepared_sequence_target,
            stop_after_one_pass=True,
            flush_final_partial_accumulation=train_config.pretrain_flush_final_partial_accumulation,
            scheduler_max_steps=effective_optimizer_steps,
            effective_optimizer_steps=effective_optimizer_steps,
        )
    if stop_mode == "token_budget_repeat_allowed":
        return TrainingRunControl(
            run_mode="token_budget_repeat_allowed",
            progress_mode="token_budget",
            prepared_token_target=prepared_token_target,
            prepared_sequence_target=prepared_sequence_target,
            scheduler_max_steps=train_config.max_steps,
            effective_optimizer_steps=train_config.max_steps,
        )
    return TrainingRunControl(
        run_mode="max_steps_limited",
        progress_mode="steps",
        prepared_token_target=prepared_token_target,
        prepared_sequence_target=prepared_sequence_target,
        scheduler_max_steps=train_config.max_steps,
        effective_optimizer_steps=train_config.max_steps,
    )


def _lm_eval_payload_callback(
    tokenizer_path: str,
    *,
    regression_path: str,
    stage_name: str,
    sequence_length: int,
    best_family_eval: dict[str, Any] | None = None,
    best_raw_lm_quality: dict[str, Any] | None = None,
    train_config: TrainConfig | None = None,
):
    def _callback(model, step: int, _final_eval: bool, state, _metrics):
        if stage_name == "pretrain":
            effective_train_config = train_config or TrainConfig()
            stable_samples = generate_raw_lm_qualitative_samples(
                model,
                tokenizer_path,
                regression_path=regression_path,
                limit=None,
                max_new_tokens=effective_train_config.raw_lm_short_probe_max_new_tokens,
                temperature=effective_train_config.raw_lm_stable_temperature,
                top_p=effective_train_config.raw_lm_stable_top_p,
            )
            stress_samples = generate_raw_lm_qualitative_samples(
                model,
                tokenizer_path,
                regression_path=regression_path,
                limit=None,
                max_new_tokens=effective_train_config.raw_lm_long_probe_max_new_tokens,
                temperature=effective_train_config.raw_lm_stress_temperature,
                top_p=effective_train_config.raw_lm_stress_top_p,
            )
            short_stable_quality = assess_raw_lm_sample_behavior(stable_samples)
            long_stress_quality = assess_raw_lm_sample_behavior(stress_samples)
            quality_status = raw_lm_quality_status(short_stable_quality, long_stress_quality)
            payload: dict[str, Any] = {
                "samples": simplify_samples(stress_samples, limit=None),
                "short_stable_samples": simplify_samples(stable_samples, limit=None),
                "long_stress_samples": simplify_samples(stress_samples, limit=None),
                "sample_mode": "raw_lm",
                "sample_decode": {
                    "stable_profile": {
                        "temperature": effective_train_config.raw_lm_stable_temperature,
                        "top_p": effective_train_config.raw_lm_stable_top_p,
                        "max_new_tokens": effective_train_config.raw_lm_short_probe_max_new_tokens,
                    },
                    "stress_profile": {
                        "temperature": effective_train_config.raw_lm_stress_temperature,
                        "top_p": effective_train_config.raw_lm_stress_top_p,
                        "max_new_tokens": effective_train_config.raw_lm_long_probe_max_new_tokens,
                    },
                },
                "short_stable_quality": short_stable_quality,
                "long_stress_quality": long_stress_quality,
                "raw_lm_quality_gate_passed": bool(
                    short_stable_quality.get("raw_lm_quality_gate_passed")
                    and long_stress_quality.get("raw_lm_quality_gate_passed")
                ),
                "raw_lm_quality_gate_reasons": sorted(
                    set(short_stable_quality.get("raw_lm_quality_gate_reasons", []))
                    | set(long_stress_quality.get("raw_lm_quality_gate_reasons", []))
                ),
                "model_quality_status": quality_status,
                "qualitative_rubric": PRETRAIN_QUALITATIVE_RUBRIC,
            }
            family_eval = evaluate_pretrain_family_holdouts(
                model,
                tokenizer_path,
                sequence_length=sequence_length,
            )
            payload["family_eval"] = family_eval.get("families", {})
            payload["best_family"] = family_eval.get("best_family")
            payload["worst_family"] = family_eval.get("worst_family")
            if best_family_eval is not None and state.best_eval_step == step:
                best_family_eval.clear()
                best_family_eval.update(family_eval)
            if best_raw_lm_quality is not None and state.best_eval_step == step:
                best_raw_lm_quality.clear()
                best_raw_lm_quality.update(
                    {
                        "short_stable_quality": short_stable_quality,
                        "long_stress_quality": long_stress_quality,
                        "raw_lm_quality_gate_passed": payload["raw_lm_quality_gate_passed"],
                        "raw_lm_quality_gate_reasons": payload["raw_lm_quality_gate_reasons"],
                        "model_quality_status": quality_status,
                    }
                )
        else:
            samples = generate_qualitative_samples(
                model,
                tokenizer_path,
                regression_path=regression_path,
                limit=3,
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
            )
            payload = {
                "samples": simplify_samples(samples),
                "sample_mode": "chat",
                "sample_decode": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 128,
                },
            }
        return payload

    return _callback


def build_stage_data_fingerprint(data_config: DataConfig, stage_name: str) -> str:
    source_map = {
        "pretrain": data_config.pretrain_sources,
        "continue": data_config.continued_pretrain_sources,
        "sft": data_config.sft_sources,
        "preference": data_config.preference_sources,
        "validation": data_config.validation_sources,
    }
    payload = {
        "stage": stage_name,
        "tokenizer_path": data_config.tokenizer_path,
        "sequence_length": data_config.sequence_length,
        "prepared_shard_size": data_config.prepared_shard_size,
        "sources": [source.to_dict() for source in source_map.get(stage_name, [])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_checkpoint_metadata(
    *,
    stage_name: str,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> dict[str, object]:
    if stage_name == "pretrain":
        return {
            "stage": stage_name,
            "parent_stage": None,
            "parent_checkpoint_path": train_config.checkpoint.resume_from or train_config.checkpoint.initialize_from,
            "input_data_fingerprint": build_stage_data_fingerprint(data_config, stage_name),
            "run_health_status": "valid",
            "artifact_status": "archiveable",
            "model_quality_status": "weak_raw_lm",
            "promotion_eligible_for_sft": False,
            "promotion_blockers": [],
            "sft_promotion_blockers": ["raw_lm_quality_gate_not_passed"],
            "promotion_eligible": False,
        }
    return {
        "stage": stage_name,
        "parent_stage": {"pretrain": None, "continue": "pretrain"}.get(stage_name),
        "parent_checkpoint_path": train_config.checkpoint.resume_from or train_config.checkpoint.initialize_from,
        "input_data_fingerprint": build_stage_data_fingerprint(data_config, stage_name),
        "artifact_status": "promotable",
        "promotion_blockers": [],
        "promotion_eligible": True,
    }


def _run_stage(
    dataset,
    validation_dataset,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    *,
    stage_name: str,
    regression_path: str,
) -> dict[str, object]:
    if train_config.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be at least 1")
    if train_config.checkpoint.initialize_from and train_config.checkpoint.resume_from:
        raise ValueError("Set either checkpoint.initialize_from or checkpoint.resume_from, not both.")
    _rank, world_size, _local_rank = init_distributed()
    try:
        best_family_eval: dict[str, Any] = {}
        best_raw_lm_quality: dict[str, Any] = {}
        seed_bundle = seed_everything(train_config.seed)
        model = CausalTransformer(model_config)
        model = maybe_wrap_fsdp(model, train_config)
        checkpoint_manager = CheckpointManager(
            output_dir=train_config.checkpoint.output_dir,
            keep_last_n=train_config.checkpoint.keep_last_n,
        )
        if train_config.checkpoint.initialize_from and not train_config.checkpoint.resume_from:
            checkpoint_manager.load(train_config.checkpoint.initialize_from, model, strict=True)
        optimizer = build_optimizer(model, train_config)
        run_control = _pretrain_run_control(
            dataset,
            train_config,
            stage_name=stage_name,
            world_size=world_size,
        )
        scheduler = build_scheduler(
            optimizer,
            train_config,
            max_steps_override=run_control.scheduler_max_steps,
        )
        if is_main_process():
            snapshot_configs(model_config, data_config=data_config, train_config=train_config)
            save_run_metadata(
                train_config,
                train_config.checkpoint.output_dir,
                extra={
                    "seed_bundle": seed_bundle,
                    "run_mode": run_control.run_mode,
                    "progress_mode": run_control.progress_mode,
                    "scheduler_max_steps": run_control.scheduler_max_steps,
                    "effective_optimizer_steps": run_control.effective_optimizer_steps,
                    "prepared_token_target": run_control.prepared_token_target,
                    "prepared_sequence_target": run_control.prepared_sequence_target,
                },
            )
            if stage_name == "pretrain" and run_control.run_mode == "one_prepared_pass":
                final_partial = _final_partial_microbatches(
                    dataset_size=len(dataset),
                    effective_batch_size=_effective_batch_size(train_config, world_size),
                    micro_batch_size=train_config.micro_batch_size,
                )
                print(
                    "WebbGPT: pretrain run_mode=one_prepared_pass "
                    f"(prepared_sequences={len(dataset):,}, "
                    f"prepared_tokens={run_control.prepared_token_target:,}, "
                    f"scheduler_max_steps={run_control.scheduler_max_steps:,}, "
                    f"final_partial_accumulation_flushed={train_config.pretrain_flush_final_partial_accumulation}, "
                    f"final_partial_microbatches={final_partial}).",
                    file=sys.stderr,
                    flush=True,
                )
        train_loader = build_dataloader(
            dataset,
            batch_size=train_config.micro_batch_size,
            shuffle=True,
            drop_last=False if run_control.stop_after_one_pass else None,
        )
        val_loader = None
        validation_dataset_size = None
        if validation_dataset is not None and len(validation_dataset) > 0:
            validation_dataset_size = len(validation_dataset)
            val_loader = build_dataloader(validation_dataset, batch_size=train_config.micro_batch_size, shuffle=False)
        eval_control = None
        if val_loader is not None:
            eval_control = EvalControl(
                stage_name=stage_name,
                eval_interval_steps=train_config.eval_every_steps,
                validation_max_batches=train_config.num_eval_batches,
                final_validation_max_batches=(
                    None
                    if train_config.final_eval_full_validation
                    else (
                        train_config.final_num_eval_batches
                        if train_config.final_num_eval_batches is not None
                        else train_config.num_eval_batches
                    )
                ),
                final_eval_full_validation=train_config.final_eval_full_validation,
                train_dataset_size=len(dataset),
                validation_dataset_size=validation_dataset_size,
                eval_history_path=str(Path(train_config.checkpoint.output_dir) / "eval_history.jsonl"),
            )
        state = run_training(
            model=model,
            train_loader=train_loader,
            train_config=train_config,
            checkpoint_manager=checkpoint_manager,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loader=val_loader,
            resume_from=train_config.checkpoint.resume_from,
            best_checkpoint_name="best-pretrain" if stage_name == "pretrain" else None,
            eval_control=eval_control,
            eval_payload_callback=(
                _lm_eval_payload_callback(
                    data_config.tokenizer_path,
                    regression_path=regression_path,
                    stage_name=stage_name,
                    sequence_length=data_config.sequence_length,
                    best_family_eval=best_family_eval if stage_name == "pretrain" else None,
                    best_raw_lm_quality=best_raw_lm_quality if stage_name == "pretrain" else None,
                    train_config=train_config,
                )
                if val_loader is not None
                else None
            ),
            train_event_printer=print_lm_train_event,
            eval_event_printer=print_lm_eval_event if val_loader is not None else None,
            checkpoint_metadata=_stage_checkpoint_metadata(
                stage_name=stage_name,
                data_config=data_config,
                train_config=train_config,
            ),
            run_control=run_control,
        )
        if is_main_process():
            blockers: list[str] = []
            if state.nonfinite_loss_steps > 0:
                blockers.append("nonfinite_loss_seen")
            if stage_name == "pretrain":
                run_health_status = "unstable" if blockers else "valid"
                artifact_status = "incomplete" if blockers else "archiveable"
                model_quality_status = str(best_raw_lm_quality.get("model_quality_status") or "weak_raw_lm")
                quality_gate_passed = bool(best_raw_lm_quality.get("raw_lm_quality_gate_passed", False))
                promotion_eligible_for_sft = quality_gate_passed and model_quality_status == "usable_raw_lm" and not blockers
                sft_promotion_blockers = [] if promotion_eligible_for_sft else ["raw_lm_quality_gate_not_passed"]
            else:
                run_health_status = "unstable" if blockers else "valid"
                artifact_status = "dev_only" if blockers else "promotable"
                model_quality_status = "not_applicable"
                promotion_eligible_for_sft = artifact_status == "promotable"
                sft_promotion_blockers = []
            summary = {
                "stage": stage_name,
                "parent_stage": {"pretrain": None, "continue": "pretrain"}.get(stage_name),
                "parent_checkpoint_path": train_config.checkpoint.resume_from or train_config.checkpoint.initialize_from,
                "input_data_fingerprint": build_stage_data_fingerprint(data_config, stage_name),
                "run_health_status": run_health_status,
                "artifact_status": artifact_status,
                "model_quality_status": model_quality_status,
                "promotion_eligible_for_sft": promotion_eligible_for_sft,
                "sft_promotion_blockers": sft_promotion_blockers,
                "promotion_blockers": blockers,
                "promotion_eligible": artifact_status == "promotable",
                "run_mode": state.run_mode,
                "progress_mode": state.progress_mode,
                "scheduler_max_steps": state.scheduler_max_steps,
                "effective_optimizer_steps": state.effective_optimizer_steps,
                "prepared_token_target": state.prepared_token_target,
                "prepared_sequence_target": state.prepared_sequence_target,
                "prepared_token_progress_percent": state.prepared_token_progress_percent,
                "prepared_sequence_progress_percent": state.prepared_sequence_progress_percent,
                "final_partial_accumulation_flushed": state.final_partial_accumulation_flushed,
                "final_partial_microbatches": state.final_partial_microbatches,
                "dataloader_passes_completed": state.dataloader_passes_completed,
                "tokens_seen": state.tokens_seen,
                "examples_seen": state.examples_seen,
                "best_eval_loss": None if state.best_eval_loss == float("inf") else state.best_eval_loss,
                "best_eval_step": None if state.best_eval_step < 0 else state.best_eval_step,
                "nonfinite_loss_steps": state.nonfinite_loss_steps,
                "nonfinite_event_samples": list(state.nonfinite_event_samples),
                "validation_enabled": val_loader is not None,
                "validation_dataset_size": validation_dataset_size,
                "interim_num_eval_batches": train_config.num_eval_batches if val_loader is not None else None,
                "final_eval_full_validation": train_config.final_eval_full_validation if val_loader is not None else None,
                "final_num_eval_batches": (
                    None
                    if train_config.final_eval_full_validation
                    else (
                        train_config.final_num_eval_batches
                        if train_config.final_num_eval_batches is not None
                        else train_config.num_eval_batches
                    )
                )
                if val_loader is not None
                else None,
                "probe_path": regression_path,
            }
            if stage_name == "pretrain":
                summary["family_eval"] = best_family_eval.get("families", {})
                summary["best_family"] = best_family_eval.get("best_family")
                summary["worst_family"] = best_family_eval.get("worst_family")
                summary["raw_lm_quality_gate_passed"] = best_raw_lm_quality.get(
                    "raw_lm_quality_gate_passed",
                    False,
                )
                summary["raw_lm_quality_gate_reasons"] = best_raw_lm_quality.get(
                    "raw_lm_quality_gate_reasons",
                    ["raw_lm_quality_not_evaluated"],
                )
                summary["short_stable_quality"] = best_raw_lm_quality.get("short_stable_quality", {})
                summary["long_stress_quality"] = best_raw_lm_quality.get("long_stress_quality", {})
                summary["best_checkpoint_path"] = (
                    str(Path(train_config.checkpoint.output_dir) / "best-pretrain")
                    if state.best_eval_step >= 0
                    else None
                )
            save_stage_summary(train_config.checkpoint.output_dir, summary)
            return summary
    finally:
        cleanup_distributed()
    return {
        "stage": stage_name,
        "artifact_status": "unknown",
        "promotion_blockers": [],
        "promotion_eligible": False,
    }


def run_pretraining(
    model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig
) -> dict[str, object]:
    builder = DatasetBuilder(data_config)
    dataset = builder.build_pretrain()
    validation_dataset = builder.build_validation() if data_config.validation_sources else None
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.token_budget is None:
        stage_config.token_budget = data_config.pretraining_token_budget
    return _run_stage(
        dataset,
        validation_dataset,
        model_config,
        data_config,
        stage_config,
        stage_name="pretrain",
        regression_path=PRETRAIN_PROBE_PATH,
    )


def run_continued_pretraining(
    model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig
) -> dict[str, object]:
    builder = DatasetBuilder(data_config)
    readiness = builder.assess_continue_readiness()
    if not readiness["passed"]:
        summary = {
            "stage": "continue",
            "skipped": True,
            "skip_reason": "continue_readiness_failed",
            "continue_readiness": readiness,
            "parent_stage": "pretrain",
            "parent_checkpoint_path": train_config.checkpoint.resume_from or train_config.checkpoint.initialize_from,
            "input_data_fingerprint": build_stage_data_fingerprint(data_config, "continue"),
            "artifact_status": "dev_only",
            "promotion_blockers": ["continue_readiness_failed"],
            "promotion_eligible": False,
            "validation_enabled": bool(data_config.validation_sources),
            "probe_path": CONTINUE_PROBE_PATH,
        }
        if is_main_process():
            print(
                "WebbGPT: skipping continued pretraining because the continue corpus failed readiness checks: "
                + ", ".join(readiness["failures"]),
                file=sys.stderr,
                flush=True,
            )
            save_stage_summary(train_config.checkpoint.output_dir, summary)
        return summary
    dataset = builder.build_continued_pretrain()
    validation_dataset = builder.build_validation() if data_config.validation_sources else None
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.continued_learning_rate is not None:
        stage_config.learning_rate = stage_config.continued_learning_rate
    if stage_config.continued_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.continued_min_learning_rate
    if stage_config.continued_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.continued_warmup_steps
    if stage_config.continued_max_steps is not None:
        stage_config.max_steps = stage_config.continued_max_steps
    if stage_config.token_budget is None:
        stage_config.token_budget = data_config.continued_pretraining_token_budget
    return _run_stage(
        dataset,
        validation_dataset,
        model_config,
        data_config,
        stage_config,
        stage_name="continue",
        regression_path=CONTINUE_PROBE_PATH,
    )
