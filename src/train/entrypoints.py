from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from config import DataConfig, ModelConfig, TrainConfig, save_config
from data.dataset import DatasetBuilder
from model.transformer import CausalTransformer
from posttrain.eval import (
    evaluate_pretrain_family_holdouts,
    generate_qualitative_samples,
    generate_raw_lm_qualitative_samples,
)
from repro import seed_everything
from train.checkpoint import CheckpointManager
from train.console import print_lm_eval_event, print_lm_train_event, simplify_samples
from train.distributed import cleanup_distributed, init_distributed, is_main_process, maybe_wrap_fsdp
from train.loop import EvalControl, build_dataloader, run_training, save_run_metadata, save_stage_summary
from train.optim import build_optimizer, build_scheduler


def snapshot_configs(model_config: ModelConfig, data_config: DataConfig, train_config: TrainConfig) -> None:
    config_dir = Path(train_config.checkpoint.output_dir) / "configs"
    save_config(model_config, config_dir / "model.json")
    save_config(data_config, config_dir / "data.json")
    save_config(train_config, config_dir / "train.json")


PRETRAIN_PROBE_PATH = "data/eval/pretrain_regression.jsonl"
CONTINUE_PROBE_PATH = "data/eval/continue_regression.jsonl"


def _lm_eval_payload_callback(
    tokenizer_path: str,
    *,
    regression_path: str,
    stage_name: str,
    sequence_length: int,
    best_family_eval: dict[str, Any] | None = None,
):
    def _callback(model, step: int, _final_eval: bool, state, _metrics):
        if stage_name == "pretrain":
            sample_temperature = 0.7
            sample_top_p = 0.95
            samples = generate_raw_lm_qualitative_samples(
                model,
                tokenizer_path,
                regression_path=regression_path,
                limit=3,
                max_new_tokens=128,
                temperature=sample_temperature,
                top_p=sample_top_p,
            )
            payload: dict[str, Any] = {
                "samples": simplify_samples(samples),
                "sample_mode": "raw_lm",
                "sample_decode": {
                    "temperature": sample_temperature,
                    "top_p": sample_top_p,
                    "max_new_tokens": 128,
                },
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
    init_distributed()
    try:
        best_family_eval: dict[str, Any] = {}
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
        scheduler = build_scheduler(optimizer, train_config)
        if is_main_process():
            snapshot_configs(model_config, data_config=data_config, train_config=train_config)
            save_run_metadata(
                train_config,
                train_config.checkpoint.output_dir,
                extra={"seed_bundle": seed_bundle},
            )
        train_loader = build_dataloader(dataset, batch_size=train_config.micro_batch_size, shuffle=True)
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
        )
        if is_main_process():
            blockers: list[str] = []
            if state.nonfinite_loss_steps > 0:
                blockers.append("nonfinite_loss_seen")
            artifact_status = "dev_only" if blockers else "promotable"
            summary = {
                "stage": stage_name,
                "parent_stage": {"pretrain": None, "continue": "pretrain"}.get(stage_name),
                "parent_checkpoint_path": train_config.checkpoint.resume_from or train_config.checkpoint.initialize_from,
                "input_data_fingerprint": build_stage_data_fingerprint(data_config, stage_name),
                "artifact_status": artifact_status,
                "promotion_blockers": blockers,
                "promotion_eligible": artifact_status == "promotable",
                "tokens_seen": state.tokens_seen,
                "examples_seen": state.examples_seen,
                "best_eval_loss": None if state.best_eval_loss == float("inf") else state.best_eval_loss,
                "best_eval_step": None if state.best_eval_step < 0 else state.best_eval_step,
                "nonfinite_loss_steps": state.nonfinite_loss_steps,
                "nonfinite_event_samples": list(state.nonfinite_event_samples),
                "validation_enabled": val_loader is not None,
                "validation_dataset_size": validation_dataset_size,
                "probe_path": regression_path,
            }
            if stage_name == "pretrain":
                summary["family_eval"] = best_family_eval.get("families", {})
                summary["best_family"] = best_family_eval.get("best_family")
                summary["worst_family"] = best_family_eval.get("worst_family")
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
