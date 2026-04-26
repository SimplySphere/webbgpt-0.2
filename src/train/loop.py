from __future__ import annotations

import contextlib
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import TrainConfig
from data.prepared import derive_artifact_status
from posttrain.eval import update_topk_candidates
from progress import build_progress_snapshot
from train.checkpoint import CheckpointManager
from train.console import dump_rounded_json
from train.distributed import barrier, is_main_process

MAX_NONFINITE_EVENT_SAMPLES = 8


def _require_torch():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    return torch, dist, DataLoader


@dataclass(slots=True)
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    examples_seen: int = 0
    best_eval_loss: float = math.inf
    best_eval_step: int = -1
    nonfinite_loss_steps: int = 0
    nonfinite_event_samples: list[dict[str, Any]] = field(default_factory=list)
    run_mode: str = "max_steps_limited"
    progress_mode: str = "steps"
    scheduler_max_steps: int | None = None
    effective_optimizer_steps: int | None = None
    prepared_token_target: int | None = None
    prepared_sequence_target: int | None = None
    prepared_token_progress_percent: float | None = None
    prepared_sequence_progress_percent: float | None = None
    final_partial_accumulation_flushed: bool = False
    final_partial_microbatches: int = 0
    dataloader_passes_completed: int = 0


@dataclass(slots=True)
class EvalControl:
    stage_name: str
    eval_metric_key: str = "loss"
    evaluate_at_start: bool = False
    early_eval_step: int | None = None
    eval_interval_steps: int | None = None
    validation_max_batches: int | None = None
    best_min_delta: float = 0.0
    early_stopping_patience_evals: int | None = None
    overfit_train_loss_threshold: float | None = None
    overfit_worsening_patience: int | None = None
    train_dataset_size: int | None = None
    validation_dataset_size: int | None = None
    final_validation_max_batches: int | None = None
    final_eval_full_validation: bool = False
    steps_per_epoch: int | None = None
    eval_history_path: str | None = None


@dataclass(slots=True)
class TrainingRunControl:
    run_mode: str = "max_steps_limited"
    progress_mode: str = "steps"
    prepared_token_target: int | None = None
    prepared_sequence_target: int | None = None
    stop_after_one_pass: bool = False
    flush_final_partial_accumulation: bool = False
    scheduler_max_steps: int | None = None
    effective_optimizer_steps: int | None = None


def _to_device(batch: dict[str, Any], device):
    torch, _, _ = _require_torch()
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _infer_batch_size(batch: dict[str, Any]) -> int:
    torch, _, _ = _require_torch()
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return int(value.shape[0])
    return 1


def _world_size() -> int:
    _, dist, _ = _require_torch()
    return int(dist.get_world_size()) if dist is not None and dist.is_initialized() else 1


def compute_effective_batch_size(train_config: TrainConfig, world_size: int | None = None) -> int:
    if world_size is None:
        world_size = _world_size()
    return int(train_config.micro_batch_size * train_config.gradient_accumulation_steps * world_size)


def validate_effective_batch_size(train_config: TrainConfig, world_size: int | None = None) -> dict[str, int]:
    if world_size is None:
        world_size = _world_size()
    effective_batch_size = compute_effective_batch_size(train_config, world_size)
    configured_global_batch_size = int(train_config.global_batch_size)
    if configured_global_batch_size != effective_batch_size:
        raise ValueError(
            "Configured global_batch_size does not match the runtime effective batch size: "
            f"global_batch_size={configured_global_batch_size}, "
            f"micro_batch_size={train_config.micro_batch_size}, "
            f"gradient_accumulation_steps={train_config.gradient_accumulation_steps}, "
            f"world_size={world_size}, effective_batch_size={effective_batch_size}. "
            "Update the config instead of relying on an implicit batch-size mismatch."
        )
    return {
        "micro_batch_size": int(train_config.micro_batch_size),
        "gradient_accumulation_steps": int(train_config.gradient_accumulation_steps),
        "world_size": int(world_size),
        "effective_batch_size": int(effective_batch_size),
        "configured_global_batch_size": configured_global_batch_size,
    }


def _model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    torch, _, _ = _require_torch()
    return {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _summarize_nonfinite_batch(batch: dict[str, Any], *, step: int, loss_value: float) -> dict[str, Any]:
    attention_mask = batch.get("attention_mask")
    tokens_in_batch = None
    if attention_mask is not None and hasattr(attention_mask, "sum"):
        try:
            tokens_in_batch = int(attention_mask.sum().item())
        except Exception:
            tokens_in_batch = None
    provenance_entries: list[dict[str, Any]] = []
    raw_provenance = batch.get("provenance_json")
    if isinstance(raw_provenance, str):
        raw_values = [raw_provenance]
    elif isinstance(raw_provenance, list):
        raw_values = [value for value in raw_provenance if isinstance(value, str)]
    else:
        raw_values = []
    for raw_value in raw_values[:3]:
        try:
            parsed = json.loads(raw_value)
        except Exception:
            continue
        if isinstance(parsed, dict):
            provenance_entries.append(parsed)
    return {
        "step": step,
        "loss": loss_value,
        "tokens_in_batch": tokens_in_batch,
        "examples_in_batch": _infer_batch_size(batch),
        "provenance": provenance_entries,
    }


def build_dataloader(dataset, batch_size: int, shuffle: bool = True, drop_last: bool | None = None):
    _, dist, DataLoader = _require_torch()
    sampler = None
    actual_drop_last = shuffle if drop_last is None else bool(drop_last)
    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=actual_drop_last)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=True,
        drop_last=actual_drop_last,
    )


def evaluate_language_model(model, dataloader, max_batches: int | None) -> dict[str, float]:
    torch, dist, _ = _require_torch()
    device = next(model.parameters()).device
    model.eval()
    losses = []
    batches_evaluated = 0
    examples_evaluated = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = _to_device(batch, device)
            outputs = model(**_model_inputs(batch))
            losses.append(outputs.loss.detach())
            batches_evaluated += 1
            examples_evaluated += _infer_batch_size(batch)
    if not losses:
        return {
            "loss": math.nan,
            "perplexity": math.nan,
            "batches_evaluated": 0,
            "examples_evaluated": 0,
        }
    loss = torch.stack(losses).mean()
    if dist is not None and dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        counts = torch.tensor([batches_evaluated, examples_evaluated], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        batches_evaluated = int(counts[0].item())
        examples_evaluated = int(counts[1].item())
    scalar_loss = float(loss.item())
    model.train()
    return {
        "loss": scalar_loss,
        "perplexity": math.exp(min(scalar_loss, 20.0)),
        "batches_evaluated": batches_evaluated,
        "examples_evaluated": examples_evaluated,
    }


def maybe_compile_model(model, enabled: bool):
    torch, _, _ = _require_torch()
    if enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except RuntimeError as exc:
            if "Dynamo is not supported on Python 3.12+" in str(exc):
                print(
                    "WebbGPT: skipping torch.compile because TorchDynamo is not supported on Python 3.12+ in this environment.",
                    file=sys.stderr,
                    flush=True,
                )
                return model
            raise
    return model


def save_run_metadata(train_config: TrainConfig, output_dir: str, extra: dict[str, Any] | None = None) -> None:
    if not is_main_process():
        return
    path = Path(output_dir) / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = train_config.to_dict()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def save_stage_summary(output_dir: str, payload: dict[str, Any]) -> None:
    if not is_main_process():
        return
    path = Path(output_dir) / "stage_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def run_training(
    model,
    train_loader,
    train_config: TrainConfig,
    checkpoint_manager: CheckpointManager,
    optimizer,
    scheduler,
    val_loader=None,
    resume_from: str | None = None,
    best_checkpoint_name: str | None = None,
    eval_payload_callback: Callable[[Any, int, bool, TrainState, dict[str, Any]], dict[str, Any] | None] | None = None,
    eval_fn: Callable[[Any, Any, int | None], dict[str, Any]] | None = None,
    eval_control: EvalControl | None = None,
    save_final_checkpoint: bool = False,
    train_event_printer: Callable[[dict[str, Any]], None] | None = None,
    eval_event_printer: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    run_control: TrainingRunControl | None = None,
) -> TrainState:
    torch, _, _ = _require_torch()
    batch_config = validate_effective_batch_size(train_config)
    if is_main_process():
        print(
            dump_rounded_json({"batch_config": batch_config}),
            flush=True,
        )
    stage_start_time = time.perf_counter()
    state = TrainState()
    if run_control is None:
        run_control = TrainingRunControl()
    state.run_mode = run_control.run_mode
    state.progress_mode = run_control.progress_mode
    state.scheduler_max_steps = run_control.scheduler_max_steps
    state.effective_optimizer_steps = run_control.effective_optimizer_steps
    state.prepared_token_target = run_control.prepared_token_target
    state.prepared_sequence_target = run_control.prepared_sequence_target
    last_saved_step = -1
    last_eval_step = -1
    last_eval_value = math.nan
    no_improvement_evals = 0
    worsening_evals = 0
    should_stop_training = False
    last_train_loss = math.nan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if train_config.use_bf16 and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    optimizer.zero_grad(set_to_none=True)
    if resume_from:
        loaded = checkpoint_manager.load(resume_from, model, optimizer=optimizer, scheduler=scheduler)
        state.step = loaded.step
        if "train_state" in loaded.payload.get("extra_state", {}):
            persisted = loaded.payload["extra_state"]["train_state"]
            state.tokens_seen = persisted.get("tokens_seen", state.tokens_seen)
            state.examples_seen = persisted.get("examples_seen", state.examples_seen)
            state.best_eval_loss = persisted.get("best_eval_loss", state.best_eval_loss)
            state.best_eval_step = persisted.get("best_eval_step", state.best_eval_step)
            state.nonfinite_loss_steps = persisted.get("nonfinite_loss_steps", state.nonfinite_loss_steps)
            state.nonfinite_event_samples = list(
                persisted.get("nonfinite_event_samples", state.nonfinite_event_samples)
            )
    model = maybe_compile_model(model, train_config.compile_model)

    micro_step = 0
    token_budget_reached = False

    def _current_checkpoint_metadata() -> dict[str, Any] | None:
        if checkpoint_metadata is None:
            return None
        payload = dict(checkpoint_metadata)
        blockers = list(payload.get("promotion_blockers", []))
        if state.nonfinite_loss_steps > 0 and "nonfinite_loss_seen" not in blockers:
            blockers.append("nonfinite_loss_seen")
        payload["promotion_blockers"] = blockers
        base_artifact_status = str(payload.get("artifact_status", "promotable"))
        if base_artifact_status in {"archiveable", "incomplete", "corrupted"}:
            payload["artifact_status"] = "incomplete" if blockers else base_artifact_status
            if blockers:
                payload["run_health_status"] = "unstable"
        else:
            payload["artifact_status"] = derive_artifact_status(
                blockers,
                base_status=base_artifact_status,
            )
        payload["promotion_eligible"] = payload["artifact_status"] == "promotable"
        payload["promotion_eligible_for_sft"] = bool(
            payload.get("promotion_eligible_for_sft", payload["promotion_eligible"])
        ) and not blockers
        payload["nonfinite_loss_steps"] = state.nonfinite_loss_steps
        payload["nonfinite_event_samples"] = list(state.nonfinite_event_samples)
        return payload

    def _checkpoint_extra_state() -> dict[str, Any]:
        payload = {
            "train_state": {
                "tokens_seen": state.tokens_seen,
                "examples_seen": state.examples_seen,
                "best_eval_loss": state.best_eval_loss,
                "best_eval_step": state.best_eval_step,
                "nonfinite_loss_steps": state.nonfinite_loss_steps,
                "nonfinite_event_samples": list(state.nonfinite_event_samples),
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
            }
        }
        current_metadata = _current_checkpoint_metadata()
        if current_metadata is not None:
            payload["checkpoint_metadata"] = current_metadata
        return payload

    def _append_eval_history(payload: dict[str, Any]) -> None:
        if eval_control is None or eval_control.eval_history_path is None or not is_main_process():
            return
        history_path = Path(eval_control.eval_history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_best_checkpoint(step: int) -> Path | None:
        if best_checkpoint_name is None:
            return None
        barrier()
        target = None
        if is_main_process():
            target = checkpoint_manager.save_named(
                best_checkpoint_name,
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state=_checkpoint_extra_state(),
            )
        barrier()
        return target

    def _approx_epoch(step: int, *, initial_eval: bool, final_eval: bool) -> float | None:
        if eval_control is None or eval_control.steps_per_epoch is None or eval_control.steps_per_epoch <= 0:
            return None
        if initial_eval:
            return 0.0
        if final_eval:
            return state.step / float(eval_control.steps_per_epoch)
        return (step + 1) / float(eval_control.steps_per_epoch)

    def _write_selection_metadata(best_path: Path, payload: dict[str, Any]) -> None:
        selection_path = best_path / "selection.json"
        selection_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def _percent(completed: int | float | None, total: int | float | None) -> float | None:
        if completed is None or total is None:
            return None
        total_value = float(total)
        if total_value <= 0 or not math.isfinite(total_value):
            return None
        completed_value = float(completed)
        if not math.isfinite(completed_value):
            return None
        return round(max(0.0, min(completed_value / total_value, 1.0)) * 100.0, 2)

    def _refresh_prepared_progress() -> None:
        state.prepared_token_progress_percent = _percent(
            state.tokens_seen,
            run_control.prepared_token_target,
        )
        state.prepared_sequence_progress_percent = _percent(
            state.examples_seen,
            run_control.prepared_sequence_target,
        )

    def _progress_counts(completed_steps: int | None = None) -> list[tuple[int | float | None, int | float | None]]:
        if run_control.progress_mode == "prepared_tokens":
            return [(state.tokens_seen, run_control.prepared_token_target)]
        if run_control.progress_mode == "token_budget":
            return [(state.tokens_seen, train_config.token_budget)]
        step_total = run_control.scheduler_max_steps or train_config.max_steps
        return [(state.step if completed_steps is None else completed_steps, step_total)]

    def _progress_metadata() -> dict[str, Any]:
        _refresh_prepared_progress()
        return {
            "run_mode": run_control.run_mode,
            "progress_mode": run_control.progress_mode,
            "prepared_token_target": run_control.prepared_token_target,
            "prepared_sequence_target": run_control.prepared_sequence_target,
            "prepared_token_progress_percent": state.prepared_token_progress_percent,
            "prepared_sequence_progress_percent": state.prepared_sequence_progress_percent,
            "scheduler_max_steps": run_control.scheduler_max_steps,
            "effective_optimizer_steps": run_control.effective_optimizer_steps,
        }

    def _stage_progress(completed_steps: int | None = None):
        return build_progress_snapshot(
            time.perf_counter() - stage_start_time,
            *_progress_counts(completed_steps),
        )

    def _run_eval(step: int, *, final_eval: bool, initial_eval: bool = False) -> None:
        nonlocal last_eval_step, last_eval_value, no_improvement_evals, worsening_evals, should_stop_training
        evaluator = eval_fn or evaluate_language_model
        max_eval_batches = train_config.num_eval_batches
        if eval_control is not None:
            max_eval_batches = (
                eval_control.final_validation_max_batches
                if final_eval
                else eval_control.validation_max_batches
            )
        metrics = evaluator(model, val_loader, max_eval_batches)
        payload: dict[str, Any] = {"step": step, "eval": metrics}
        if initial_eval:
            payload["initial_eval"] = True
        if final_eval:
            payload["final_eval"] = True
        if eval_control is not None:
            approx_epoch = _approx_epoch(step, initial_eval=initial_eval, final_eval=final_eval)
            if approx_epoch is not None:
                payload["approx_epoch"] = approx_epoch
            payload["train_dataset_size"] = eval_control.train_dataset_size
            payload["validation_dataset_size"] = eval_control.validation_dataset_size
            payload["train_examples_seen"] = state.examples_seen
            payload["validation_batches_evaluated"] = metrics.get("batches_evaluated")
            payload["validation_examples_evaluated"] = metrics.get("examples_evaluated")
            eval_batches_used = metrics.get("batches_evaluated")
            eval_sequences_estimated = metrics.get("examples_evaluated")
            validation_sequences_total = eval_control.validation_dataset_size
            coverage_percent = _percent(eval_sequences_estimated, validation_sequences_total)
            eval_mode = "interim_subset"
            if final_eval:
                eval_mode = "full_validation" if max_eval_batches is None else "final_subset"
            payload.update(
                {
                    "eval_batches_used": eval_batches_used,
                    "eval_sequences_estimated": eval_sequences_estimated,
                    "validation_sequences_total": validation_sequences_total,
                    "eval_coverage_percent": coverage_percent,
                    "eval_mode": eval_mode,
                    "final_eval_full_validation": bool(eval_control.final_eval_full_validation)
                    if final_eval
                    else False,
                    "final_num_eval_batches": eval_control.final_validation_max_batches
                    if final_eval
                    else None,
                }
            )
        progress = _stage_progress(completed_steps=max(step, 0))
        payload["progress_percent"] = (
            None if progress.fraction_complete is None else round(progress.fraction_complete * 100.0, 2)
        )
        payload["stage_elapsed_sec"] = round(progress.elapsed_seconds, 2)
        payload["stage_eta_sec"] = None if progress.remaining_seconds is None else round(progress.remaining_seconds, 2)
        payload["progress_summary"] = progress.summary
        payload.update(_progress_metadata())
        improved = (
            not math.isnan(float(metrics.get((eval_control.eval_metric_key if eval_control else "loss"), math.nan)))
            and float(metrics.get((eval_control.eval_metric_key if eval_control else "loss"), math.nan))
            < state.best_eval_loss
            - (eval_control.best_min_delta if eval_control is not None else 0.0)
        )
        metric_key = eval_control.eval_metric_key if eval_control is not None else "loss"
        selection_value = float(metrics.get(metric_key, math.nan))
        previous_best_value = None if math.isinf(state.best_eval_loss) else state.best_eval_loss
        previous_best_step = state.best_eval_step if state.best_eval_step >= 0 else None
        if improved:
            state.best_eval_loss = selection_value
            state.best_eval_step = step
            no_improvement_evals = 0
        elif not math.isnan(selection_value):
            no_improvement_evals += 1
        if (
            not math.isnan(last_eval_value)
            and not math.isnan(selection_value)
            and selection_value > last_eval_value
        ):
            worsening_evals += 1
        elif not math.isnan(selection_value):
            worsening_evals = 0
        last_eval_value = selection_value
        payload["best_step_so_far"] = state.best_eval_step
        if is_main_process():
            if eval_payload_callback is not None:
                extra_payload = eval_payload_callback(model, step, final_eval, state, metrics)
                if extra_payload:
                    payload.update(extra_payload)
                    if bool(extra_payload.get("should_stop_training")):
                        should_stop_training = True
                        if is_main_process() and eval_control is not None:
                            print(
                                f"WebbGPT: stopping {eval_control.stage_name} early because a qualitative gate requested termination.",
                                file=sys.stderr,
                                flush=True,
                            )
            if eval_event_printer is not None:
                eval_event_printer(payload)
            else:
                print(dump_rounded_json(payload))
        _append_eval_history(payload)
        best_path = None
        if improved:
            best_path = _save_best_checkpoint(step)
            if best_path is not None and eval_control is not None and is_main_process():
                selection_payload = {
                    "stage": eval_control.stage_name,
                    "step": step,
                    "approx_epoch": _approx_epoch(step, initial_eval=initial_eval, final_eval=final_eval),
                    "train_dataset_size": eval_control.train_dataset_size,
                    "validation_dataset_size": eval_control.validation_dataset_size,
                    "train_examples_seen": state.examples_seen,
                    "validation_batches_evaluated": metrics.get("batches_evaluated"),
                    "validation_examples_evaluated": metrics.get("examples_evaluated"),
                    "eval_batches_used": payload.get("eval_batches_used"),
                    "eval_sequences_estimated": payload.get("eval_sequences_estimated"),
                    "validation_sequences_total": payload.get("validation_sequences_total"),
                    "eval_coverage_percent": payload.get("eval_coverage_percent"),
                    "eval_mode": payload.get("eval_mode"),
                    "final_eval_full_validation": payload.get("final_eval_full_validation"),
                    "final_num_eval_batches": payload.get("final_num_eval_batches"),
                    "run_mode": run_control.run_mode,
                    "progress_mode": run_control.progress_mode,
                    "prepared_token_target": run_control.prepared_token_target,
                    "prepared_sequence_target": run_control.prepared_sequence_target,
                    "prepared_token_progress_percent": state.prepared_token_progress_percent,
                    "prepared_sequence_progress_percent": state.prepared_sequence_progress_percent,
                    "scheduler_max_steps": run_control.scheduler_max_steps,
                    "metrics": metrics,
                    "selection_metric": metric_key,
                    "selection_value": selection_value,
                    "previous_best_value": previous_best_value,
                    "previous_best_step": previous_best_step,
                    "replacement_reason": (
                        "new best validation metric"
                        if previous_best_value is not None
                        else "first best validation checkpoint"
                    ),
                    "improvement_delta": (
                        None
                        if previous_best_value is None or math.isnan(selection_value)
                        else previous_best_value - selection_value
                    ),
                    "best_step_so_far": state.best_eval_step,
                }
                if payload.get("samples") is not None:
                    selection_payload["samples"] = payload.get("samples")
                elif payload.get("qualitative_samples") is not None:
                    selection_payload["qualitative_samples"] = payload.get("qualitative_samples")
                for key in (
                    "short_stable_samples",
                    "long_stress_samples",
                    "short_stable_quality",
                    "long_stress_quality",
                    "raw_lm_quality_gate_passed",
                    "raw_lm_quality_gate_reasons",
                    "model_quality_status",
                ):
                    if key in payload:
                        selection_payload[key] = payload[key]
                _write_selection_metadata(best_path, selection_payload)
                candidate_name = f"candidate-step-{step:08d}"
                candidate_path = checkpoint_manager.save_named(
                    candidate_name,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_state=_checkpoint_extra_state(),
                )
                _write_selection_metadata(candidate_path, selection_payload)
                update_topk_candidates(
                    checkpoint_manager.output_dir,
                    candidate_path=candidate_path,
                    candidate_payload=selection_payload,
                    metric_key=metric_key,
                    limit=train_config.posttrain_top_k_checkpoints,
                    lower_is_better=True,
                )
        if (
            eval_control is not None
            and eval_control.early_stopping_patience_evals is not None
            and no_improvement_evals >= eval_control.early_stopping_patience_evals
        ):
            should_stop_training = True
            if is_main_process():
                print(
                    f"WebbGPT: stopping {eval_control.stage_name} early after {no_improvement_evals} validation evals without a new best checkpoint.",
                    file=sys.stderr,
                    flush=True,
                )
        if (
            eval_control is not None
            and eval_control.overfit_train_loss_threshold is not None
            and eval_control.overfit_worsening_patience is not None
            and not math.isnan(last_train_loss)
            and last_train_loss < eval_control.overfit_train_loss_threshold
            and worsening_evals >= eval_control.overfit_worsening_patience
        ):
            should_stop_training = True
            if is_main_process():
                print(
                    f"WebbGPT: stopping {eval_control.stage_name} early because training loss fell below "
                    f"{eval_control.overfit_train_loss_threshold} while validation degraded for {worsening_evals} evals.",
                    file=sys.stderr,
                    flush=True,
                )
        last_eval_step = step

    def _should_run_scheduled_eval(step: int) -> bool:
        if val_loader is None:
            return False
        if eval_control is None:
            return (
                train_config.eval_every_steps > 0
                and step > 0
                and step % train_config.eval_every_steps == 0
            )
        if step <= 0:
            return False
        early_eval_due = eval_control.early_eval_step is not None and step == eval_control.early_eval_step
        interval_due = (
            eval_control.eval_interval_steps is not None
            and eval_control.eval_interval_steps > 0
            and step % eval_control.eval_interval_steps == 0
        )
        return early_eval_due or interval_due

    def _step_limit_reached() -> bool:
        if run_control.stop_after_one_pass:
            return False
        return state.step >= train_config.max_steps

    def _token_budget_reached() -> bool:
        return (
            not run_control.stop_after_one_pass
            and train_config.token_budget is not None
            and state.tokens_seen >= train_config.token_budget
        )

    def _scale_partial_gradients(partial_microbatches: int) -> None:
        if partial_microbatches <= 0 or partial_microbatches >= train_config.gradient_accumulation_steps:
            return
        scale = train_config.gradient_accumulation_steps / float(partial_microbatches)
        with torch.no_grad():
            for parameter in model.parameters():
                gradient = getattr(parameter, "grad", None)
                if gradient is not None:
                    gradient.mul_(scale)

    def _complete_optimizer_step(
        *,
        train_loss_value: float,
        start_time: float,
        final_partial_microbatches: int = 0,
    ) -> None:
        nonlocal last_saved_step, last_train_loss, token_budget_reached
        if final_partial_microbatches:
            _scale_partial_gradients(final_partial_microbatches)
            state.final_partial_accumulation_flushed = True
            state.final_partial_microbatches = final_partial_microbatches
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        last_train_loss = train_loss_value

        if state.step % train_config.log_every_steps == 0 and is_main_process():
            elapsed = time.perf_counter() - start_time
            progress = _stage_progress(completed_steps=state.step + 1)
            payload = {
                "step": state.step,
                "loss": last_train_loss,
                "lr": float(scheduler.get_last_lr()[0]),
                "tokens_seen": state.tokens_seen,
                "examples_seen": state.examples_seen,
                "step_time_sec": round(elapsed, 2),
                "progress_percent": (
                    None
                    if progress.fraction_complete is None
                    else round(progress.fraction_complete * 100.0, 2)
                ),
                "stage_elapsed_sec": round(progress.elapsed_seconds, 2),
                "stage_eta_sec": (
                    None if progress.remaining_seconds is None else round(progress.remaining_seconds, 2)
                ),
                "progress_summary": progress.summary,
            }
            if final_partial_microbatches:
                payload["final_partial_accumulation_flushed"] = True
                payload["final_partial_microbatches"] = final_partial_microbatches
            payload.update(_progress_metadata())
            if train_event_printer is not None:
                train_event_printer(payload)
            else:
                print(dump_rounded_json(payload))

        if _should_run_scheduled_eval(state.step):
            _run_eval(state.step, final_eval=False)

        if (
            train_config.checkpoint.save_every_steps > 0
            and state.step > 0
            and state.step % train_config.checkpoint.save_every_steps == 0
        ):
            barrier()
            if is_main_process():
                checkpoint_manager.save(
                    step=state.step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_state=_checkpoint_extra_state(),
                )
                last_saved_step = state.step
            barrier()

        state.step += 1
        if _token_budget_reached():
            token_budget_reached = True

    if val_loader is not None and eval_control is not None and eval_control.evaluate_at_start:
        _run_eval(0, final_eval=False, initial_eval=True)

    partial_loss_value = math.nan
    partial_start_time = stage_start_time
    while not _step_limit_reached() and not token_budget_reached and not should_stop_training:
        consumed_any_batch = False
        for batch in train_loader:
            if _step_limit_reached() or token_budget_reached or should_stop_training:
                break
            consumed_any_batch = True
            start_time = time.perf_counter()
            batch = _to_device(batch, device)
            with scaler_context:
                outputs = model(**_model_inputs(batch))
                raw_loss = outputs.loss
            raw_loss_value = float(raw_loss.item())
            if not math.isfinite(raw_loss_value):
                state.nonfinite_loss_steps += 1
                if len(state.nonfinite_event_samples) < MAX_NONFINITE_EVENT_SAMPLES:
                    state.nonfinite_event_samples.append(
                        _summarize_nonfinite_batch(batch, step=state.step, loss_value=raw_loss_value)
                    )
                optimizer.zero_grad(set_to_none=True)
                if is_main_process():
                    print(
                        f"WebbGPT: skipping non-finite training loss at step {state.step} "
                        f"(count={state.nonfinite_loss_steps}).",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            loss = raw_loss / train_config.gradient_accumulation_steps
            loss.backward()
            tokens_this_batch = int(batch["attention_mask"].sum().item())
            state.tokens_seen += tokens_this_batch
            state.examples_seen += _infer_batch_size(batch)
            micro_step += 1
            partial_loss_value = raw_loss_value
            partial_start_time = start_time

            if micro_step % train_config.gradient_accumulation_steps == 0:
                _complete_optimizer_step(
                    train_loss_value=float(loss.item() * train_config.gradient_accumulation_steps),
                    start_time=start_time,
                )
                if token_budget_reached:
                    break
        state.dataloader_passes_completed += 1 if consumed_any_batch else 0
        if run_control.stop_after_one_pass or not consumed_any_batch:
            break

    remaining_microbatches = micro_step % train_config.gradient_accumulation_steps
    if (
        run_control.stop_after_one_pass
        and run_control.flush_final_partial_accumulation
        and remaining_microbatches
        and not should_stop_training
    ):
        _complete_optimizer_step(
            train_loss_value=partial_loss_value,
            start_time=partial_start_time,
            final_partial_microbatches=remaining_microbatches,
        )

    if eval_control is not None:
        should_run_final_eval = (
            val_loader is not None
            and state.step > 0
            and last_eval_step != state.step
        )
    else:
        should_run_final_eval = (
            val_loader is not None
            and train_config.eval_every_steps > 0
            and state.step > 0
            and last_eval_step != state.step - 1
        )
    if should_run_final_eval:
        _run_eval(state.step, final_eval=True)
    if save_final_checkpoint and state.step > 0 and state.step != last_saved_step:
        barrier()
        if is_main_process():
            checkpoint_manager.save(
                step=state.step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state=_checkpoint_extra_state(),
            )
        barrier()
    _refresh_prepared_progress()
    return state
