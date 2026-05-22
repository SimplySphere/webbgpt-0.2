from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
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
from torch_runtime import autocast_if_available, get_torch_device

MAX_NONFINITE_EVENT_SAMPLES = 8
LOW_LOSS_TIER_ORDER = ("severe", "suspicious", "broad")


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
    low_loss_event_count: int = 0
    low_loss_event_steps: set[int] = field(default_factory=set)
    low_loss_events_by_tier: dict[str, int] = field(default_factory=dict)
    low_loss_events_by_source: dict[str, int] = field(default_factory=dict)
    low_loss_events_by_source_by_tier: dict[str, dict[str, int]] = field(default_factory=dict)
    low_loss_events_by_contributor: dict[str, int] = field(default_factory=dict)
    min_low_loss_event: dict[str, Any] | None = None
    min_low_loss_event_by_tier: dict[str, dict[str, Any]] = field(default_factory=dict)


def low_loss_summary_from_state(state: TrainState, *, top_k: int = 10) -> dict[str, Any]:
    raw_source_counts = getattr(state, "low_loss_events_by_source", {}) or {}
    source_counts = dict(sorted(dict(raw_source_counts).items()))
    raw_tier_counts = getattr(state, "low_loss_events_by_tier", {}) or {}
    tier_counts = {
        tier: int(raw_tier_counts.get(tier, 0))
        for tier in LOW_LOSS_TIER_ORDER
        if int(raw_tier_counts.get(tier, 0)) > 0
    }
    raw_source_counts_by_tier = getattr(state, "low_loss_events_by_source_by_tier", {}) or {}
    source_counts_by_tier = {
        source: {
            tier: int(tier_counts.get(tier, 0))
            for tier in LOW_LOSS_TIER_ORDER
            if int(tier_counts.get(tier, 0)) > 0
        }
        for source, tier_counts in sorted(dict(raw_source_counts_by_tier).items())
        if isinstance(tier_counts, dict)
    }
    raw_contributor_counts = getattr(state, "low_loss_events_by_contributor", {}) or {}
    low_loss_event_steps = getattr(state, "low_loss_event_steps", set()) or set()
    top_sources = [
        {"source": source, "count": count}
        for source, count in sorted(
            source_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    ]
    top_contributors: list[dict[str, Any]] = []
    for raw_key, count in sorted(
        dict(raw_contributor_counts).items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_k]:
        try:
            contributor = json.loads(raw_key)
        except json.JSONDecodeError:
            contributor = {"key": raw_key}
        if isinstance(contributor, dict):
            top_contributors.append({**contributor, "count": int(count)})
    return {
        "low_loss_event_count": int(getattr(state, "low_loss_event_count", 0)),
        "unique_low_loss_steps": len(low_loss_event_steps),
        "low_loss_events_by_tier": tier_counts,
        "low_loss_events_by_source": source_counts,
        "low_loss_events_by_source_by_tier": source_counts_by_tier,
        "min_low_loss_event": getattr(state, "min_low_loss_event", None),
        "min_low_loss_event_by_tier": getattr(state, "min_low_loss_event_by_tier", {}),
        "top_low_loss_sources": top_sources,
        "top_low_loss_contributors": top_contributors,
    }


def _low_loss_thresholds(train_config: TrainConfig) -> dict[str, float]:
    thresholds = {
        "severe": train_config.severe_low_loss_threshold,
        "suspicious": train_config.suspicious_low_loss_threshold,
        "broad": (
            train_config.broad_low_loss_threshold
            if train_config.broad_low_loss_threshold is not None
            else train_config.low_loss_probe_threshold
        ),
    }
    return {
        tier: float(value)
        for tier, value in thresholds.items()
        if value is not None
    }


def _low_loss_tier_for_value(raw_loss_value: float, thresholds: dict[str, float]) -> tuple[str, float] | None:
    for tier in LOW_LOSS_TIER_ORDER:
        threshold = thresholds.get(tier)
        if threshold is not None and raw_loss_value < threshold:
            return tier, threshold
    return None


def _low_loss_event_name(tier: str) -> str:
    return f"{tier}_low_loss_batch_provenance"


def _contributor_key(contributor: dict[str, Any]) -> str:
    payload = {
        "source": str(contributor.get("source") or "unknown"),
        "family": str(contributor.get("family") or ""),
        "document_id": str(contributor.get("document_id") or ""),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _should_emit_low_loss_event(tier: str, tier_count: int) -> bool:
    if tier == "severe":
        return True
    if tier == "suspicious":
        return tier_count <= 10 or tier_count % 100 == 0
    return False


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


def _token_count_from_attention_mask(batch: dict[str, Any]) -> int | None:
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None and hasattr(attention_mask, "sum"):
        try:
            return int(attention_mask.sum().item())
        except Exception:
            return None
    return None


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


def _provenance_json_values(batch: dict[str, Any]) -> list[str]:
    raw_provenance = batch.get("provenance_json")
    if isinstance(raw_provenance, str):
        return [raw_provenance]
    if isinstance(raw_provenance, list):
        return [value for value in raw_provenance if isinstance(value, str)]
    if isinstance(raw_provenance, tuple):
        return [value for value in raw_provenance if isinstance(value, str)]
    return []


def _parse_batch_provenance_entries(batch: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    provenance_entries: list[dict[str, Any]] = []
    raw_values = _provenance_json_values(batch)
    selected_values = raw_values if limit is None else raw_values[:limit]
    for raw_value in selected_values:
        try:
            parsed = json.loads(raw_value)
        except Exception:
            continue
        if isinstance(parsed, dict):
            provenance_entries.append(parsed)
    return provenance_entries


def _batch_token_id_preview(batch: dict[str, Any], *, max_tokens: int = 32) -> list[int] | None:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        return None
    try:
        first_row = input_ids[0] if hasattr(input_ids, "__getitem__") else input_ids
        if hasattr(first_row, "detach"):
            values = first_row.detach().cpu().tolist()
        elif hasattr(first_row, "tolist"):
            values = first_row.tolist()
        else:
            values = list(first_row)
        return [int(value) for value in values[:max_tokens]]
    except Exception:
        return None


def _summarize_batch_provenance(
    batch: dict[str, Any],
    *,
    step: int,
    loss_value: float,
    event: str,
    threshold: float | None = None,
    include_token_preview: bool = False,
    provenance_limit: int | None = None,
) -> dict[str, Any]:
    tokens_in_batch = _token_count_from_attention_mask(batch)
    provenance_entries = _parse_batch_provenance_entries(batch, limit=provenance_limit)
    source_name_values = [
        source_name
        for entry in provenance_entries
        for source_name in (
            entry.get("source_names", [])
            if isinstance(entry.get("source_names", []), list)
            else []
        )
    ]
    source_names = sorted(
        {
            str(source_name)
            for source_name in source_name_values
            if str(source_name)
        }
    )
    contributors = [
        dict(contributor)
        for entry in provenance_entries
        for contributor in (
            entry.get("contributors", [])
            if isinstance(entry.get("contributors", []), list)
            else []
        )
        if isinstance(contributor, dict)
    ]
    packed_document_count = sum(
        int(entry.get("packed_document_count", 0))
        for entry in provenance_entries
        if isinstance(entry.get("packed_document_count", 0), (int, float))
    )
    if packed_document_count <= 0 and contributors:
        packed_document_count = len(contributors)
    approximate_token_count = sum(
        int(entry.get("approximate_token_count", 0))
        for entry in provenance_entries
        if isinstance(entry.get("approximate_token_count", 0), (int, float))
    )
    if approximate_token_count <= 0 and tokens_in_batch is not None:
        approximate_token_count = tokens_in_batch
    payload: dict[str, Any] = {
        "event": event,
        "step": step,
        "loss": loss_value,
        "threshold": threshold,
        "source_names": source_names,
        "contributors": contributors[:20],
        "contributor_count": len(contributors),
        "packed_document_count": packed_document_count,
        "approximate_token_count": approximate_token_count,
        "tokens_in_batch": tokens_in_batch,
        "examples_in_batch": _infer_batch_size(batch),
        "provenance": provenance_entries[:5],
    }
    if include_token_preview:
        preview = _batch_token_id_preview(batch)
        if preview is not None:
            payload["token_id_preview"] = preview
    return payload


def _summarize_nonfinite_batch(batch: dict[str, Any], *, step: int, loss_value: float) -> dict[str, Any]:
    summary = _summarize_batch_provenance(
        batch,
        step=step,
        loss_value=loss_value,
        event="nonfinite_loss_batch_provenance",
        provenance_limit=3,
    )
    return {
        **summary,
        "step": step,
        "loss": loss_value,
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
    device = get_torch_device()
    model = model.to(device)
    scaler_context = autocast_if_available(torch, device=device, use_bf16=train_config.use_bf16)
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
            state.low_loss_event_count = int(
                persisted.get("low_loss_event_count", state.low_loss_event_count)
            )
            state.low_loss_event_steps = {
                int(step)
                for step in persisted.get("low_loss_event_steps", state.low_loss_event_steps)
            }
            state.low_loss_events_by_tier = {
                str(tier): int(count)
                for tier, count in dict(
                    persisted.get("low_loss_events_by_tier", state.low_loss_events_by_tier)
                ).items()
            }
            state.low_loss_events_by_source = {
                str(source): int(count)
                for source, count in dict(
                    persisted.get("low_loss_events_by_source", state.low_loss_events_by_source)
                ).items()
            }
            state.low_loss_events_by_source_by_tier = {
                str(source): {
                    str(tier): int(count)
                    for tier, count in dict(tier_counts).items()
                }
                for source, tier_counts in dict(
                    persisted.get(
                        "low_loss_events_by_source_by_tier",
                        state.low_loss_events_by_source_by_tier,
                    )
                ).items()
            }
            state.low_loss_events_by_contributor = {
                str(contributor): int(count)
                for contributor, count in dict(
                    persisted.get(
                        "low_loss_events_by_contributor",
                        state.low_loss_events_by_contributor,
                    )
                ).items()
            }
            min_low_loss_event = persisted.get("min_low_loss_event", state.min_low_loss_event)
            state.min_low_loss_event = (
                dict(min_low_loss_event) if isinstance(min_low_loss_event, dict) else None
            )
            state.min_low_loss_event_by_tier = {
                str(tier): dict(event)
                for tier, event in dict(
                    persisted.get(
                        "min_low_loss_event_by_tier",
                        state.min_low_loss_event_by_tier,
                    )
                ).items()
                if isinstance(event, dict)
            }
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
                "low_loss_event_count": state.low_loss_event_count,
                "low_loss_event_steps": sorted(state.low_loss_event_steps),
                "low_loss_events_by_tier": dict(state.low_loss_events_by_tier),
                "low_loss_events_by_source": dict(state.low_loss_events_by_source),
                "low_loss_events_by_source_by_tier": {
                    tier: dict(source_counts)
                    for tier, source_counts in state.low_loss_events_by_source_by_tier.items()
                },
                "low_loss_events_by_contributor": dict(state.low_loss_events_by_contributor),
                "min_low_loss_event": state.min_low_loss_event,
                "min_low_loss_event_by_tier": dict(state.min_low_loss_event_by_tier),
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

    def _read_selection_metadata(checkpoint_path: Path) -> dict[str, Any] | None:
        selection_path = checkpoint_path / "selection.json"
        if not selection_path.exists():
            return None
        try:
            payload = json.loads(selection_path.read_text())
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

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

    def _emit_train_payload(payload: dict[str, Any]) -> None:
        if train_event_printer is not None:
            train_event_printer(payload)
        else:
            print(dump_rounded_json(payload))

    def _record_low_loss_event(payload: dict[str, Any], *, tier: str) -> int:
        state.low_loss_event_count += 1
        state.low_loss_event_steps.add(int(payload.get("step", state.step)))
        tier_counts = Counter(state.low_loss_events_by_tier)
        tier_counts.update([tier])
        state.low_loss_events_by_tier = dict(tier_counts)
        source_names = payload.get("source_names")
        if isinstance(source_names, list) and source_names:
            sources = [str(source_name) for source_name in source_names if str(source_name)]
        else:
            sources = ["unknown"]
        source_counts = Counter(state.low_loss_events_by_source)
        source_counts.update(sources)
        state.low_loss_events_by_source = dict(source_counts)
        source_tier_counts = {
            str(raw_source): dict(raw_counts)
            for raw_source, raw_counts in state.low_loss_events_by_source_by_tier.items()
        }
        for source in sources:
            source_tier_counter = Counter(source_tier_counts.get(source, {}))
            source_tier_counter.update([tier])
            source_tier_counts[source] = dict(source_tier_counter)
        state.low_loss_events_by_source_by_tier = source_tier_counts
        contributor_counts = Counter(state.low_loss_events_by_contributor)
        contributors = payload.get("contributors")
        if isinstance(contributors, list):
            contributor_counts.update(
                _contributor_key(contributor)
                for contributor in contributors
                if isinstance(contributor, dict)
            )
        state.low_loss_events_by_contributor = dict(contributor_counts)
        if (
            state.min_low_loss_event is None
            or float(payload.get("loss", math.inf)) < float(state.min_low_loss_event.get("loss", math.inf))
        ):
            state.min_low_loss_event = dict(payload)
        min_for_tier = state.min_low_loss_event_by_tier.get(tier)
        if (
            min_for_tier is None
            or float(payload.get("loss", math.inf)) < float(min_for_tier.get("loss", math.inf))
        ):
            state.min_low_loss_event_by_tier[tier] = dict(payload)
        return int(state.low_loss_events_by_tier.get(tier, 0))

    def _emit_low_loss_summary() -> None:
        if not train_config.log_batch_provenance_extremes or not is_main_process():
            return
        _emit_train_payload(
            {
                "event": "low_loss_batch_provenance_summary",
                "thresholds": _low_loss_thresholds(train_config),
                **low_loss_summary_from_state(state),
            }
        )

    def _maybe_log_batch_provenance_extreme(
        batch: dict[str, Any],
        *,
        raw_loss_value: float,
        micro_step_index: int,
    ) -> None:
        if not train_config.log_batch_provenance_extremes or not is_main_process():
            return
        low_thresholds = _low_loss_thresholds(train_config)
        high_threshold = train_config.high_loss_probe_threshold
        event = None
        threshold: float | None = None
        low_loss_tier = _low_loss_tier_for_value(raw_loss_value, low_thresholds)
        if low_loss_tier is not None:
            tier, threshold = low_loss_tier
            event = _low_loss_event_name(tier)
        elif high_threshold is not None and raw_loss_value >= float(high_threshold):
            tier = None
            event = "high_loss_batch_provenance"
            threshold = float(high_threshold)
        if event is None:
            return
        payload = _summarize_batch_provenance(
            batch,
            step=state.step,
            loss_value=raw_loss_value,
            event=event,
            threshold=threshold,
            include_token_preview=low_loss_tier is not None and low_loss_tier[0] == "severe",
        )
        payload["micro_step"] = micro_step_index
        payload["gradient_accumulation_steps"] = train_config.gradient_accumulation_steps
        if low_loss_tier is not None:
            payload["tier"] = low_loss_tier[0]
            tier_count = _record_low_loss_event(payload, tier=low_loss_tier[0])
            if _should_emit_low_loss_event(low_loss_tier[0], tier_count):
                _emit_train_payload(payload)
            return
        _emit_train_payload(payload)

    def _selection_payload(
        *,
        step: int,
        metrics: dict[str, Any],
        payload: dict[str, Any],
        metric_key: str,
        selection_value: float,
        previous_best_value: float | None,
        previous_best_step: int | None,
        initial_eval: bool,
        final_eval: bool,
        replacement_reason: str,
    ) -> dict[str, Any]:
        selection_payload = {
            "stage": eval_control.stage_name if eval_control is not None else None,
            "step": step,
            "approx_epoch": _approx_epoch(step, initial_eval=initial_eval, final_eval=final_eval),
            "train_dataset_size": eval_control.train_dataset_size if eval_control is not None else None,
            "validation_dataset_size": eval_control.validation_dataset_size if eval_control is not None else None,
            "train_examples_seen": state.examples_seen,
            "validation_batches_evaluated": metrics.get("batches_evaluated"),
            "validation_examples_evaluated": metrics.get("examples_evaluated"),
            "eval_batches_used": payload.get("eval_batches_used"),
            "eval_sequences_estimated": payload.get("eval_sequences_estimated"),
            "validation_sequences_total": payload.get("validation_sequences_total"),
            "eval_coverage_percent": payload.get("eval_coverage_percent"),
            "eval_mode": payload.get("eval_mode"),
            "selection_eval_mode": payload.get("eval_mode"),
            "selection_eval_batches": payload.get("eval_batches_used"),
            "selection_eval_coverage_percent": payload.get("eval_coverage_percent"),
            "selected_from_interim_eval": not final_eval,
            "final_selection_confirmed": bool(final_eval),
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
            "replacement_reason": replacement_reason,
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
            "family_eval",
            "family_eval_coverage",
            "best_family",
            "worst_family",
        ):
            if key in payload:
                selection_payload[key] = payload[key]
        return selection_payload

    def _choose_final_selection(
        *,
        step: int,
        current_metrics: dict[str, Any],
        payload: dict[str, Any],
        metric_key: str,
        selection_value: float,
        evaluator: Callable[[Any, Any, int | None], dict[str, Any]],
    ) -> None:
        if eval_control is None or best_checkpoint_name is None or not is_main_process():
            return
        if math.isnan(selection_value):
            return
        best_path = Path(checkpoint_manager.output_dir) / best_checkpoint_name
        interim_selection = _read_selection_metadata(best_path) if best_path.exists() else None
        previous_best_value = None if math.isinf(state.best_eval_loss) else state.best_eval_loss
        previous_best_step = state.best_eval_step if state.best_eval_step >= 0 else None
        current_candidate_name = f"candidate-final-step-{step:08d}"
        current_candidate_path = checkpoint_manager.save_named(
            current_candidate_name,
            step=step,
            model=model,
            optimizer=None,
            scheduler=None,
            extra_state=_checkpoint_extra_state(),
        )
        interim_final_metrics = None
        interim_final_value = math.inf
        interim_step = (
            int(interim_selection["step"])
            if isinstance(interim_selection, dict) and interim_selection.get("step") is not None
            else previous_best_step
        )
        can_confirm_interim = best_path.exists() and hasattr(checkpoint_manager, "load")
        if can_confirm_interim:
            checkpoint_manager.load(str(best_path), model, strict=True)
            interim_final_metrics = evaluator(model, val_loader, payload.get("final_num_eval_batches"))
            interim_final_value = float(interim_final_metrics.get(metric_key, math.nan))
            if math.isnan(interim_final_value):
                interim_final_value = math.inf
            checkpoint_manager.load(str(current_candidate_path), model, strict=True)

        choose_interim = interim_final_value <= selection_value
        if choose_interim and interim_final_metrics is not None and interim_step is not None:
            state.best_eval_loss = interim_final_value
            state.best_eval_step = int(interim_step)
            final_selection_payload = dict(interim_selection or {})
            final_selection_payload.update(
                {
                    "stage": eval_control.stage_name,
                    "step": int(interim_step),
                    "selection_metric": metric_key,
                    "selection_value": interim_final_value,
                    "metrics": interim_final_metrics,
                    "selection_eval_mode": payload.get("eval_mode"),
                    "selection_eval_batches": payload.get("eval_batches_used"),
                    "selection_eval_coverage_percent": payload.get("eval_coverage_percent"),
                    "eval_mode": payload.get("eval_mode"),
                    "eval_batches_used": payload.get("eval_batches_used"),
                    "eval_sequences_estimated": payload.get("eval_sequences_estimated"),
                    "validation_sequences_total": payload.get("validation_sequences_total"),
                    "eval_coverage_percent": payload.get("eval_coverage_percent"),
                    "selected_from_interim_eval": True,
                    "final_selection_confirmed": True,
                    "selected_checkpoint_path": str(best_path),
                    "replacement_reason": "interim checkpoint confirmed by final selection eval",
                    "best_step_so_far": state.best_eval_step,
                    "best_interim_checkpoint": {
                        "path": str(best_path),
                        "step": int(interim_step),
                        "interim_selection_value": (
                            None if interim_selection is None else interim_selection.get("selection_value")
                        ),
                        "interim_eval_mode": (
                            None if interim_selection is None else interim_selection.get("selection_eval_mode")
                        ),
                        "final_metrics": interim_final_metrics,
                    },
                    "final_checkpoint": {
                        "path": str(current_candidate_path),
                        "step": step,
                        "final_metrics": current_metrics,
                        "selection_value": selection_value,
                    },
                    "best_final_eval_confirmed_checkpoint": {
                        "path": str(best_path),
                        "step": int(interim_step),
                        "metrics": interim_final_metrics,
                    },
                    "compared_checkpoints": [
                        {
                            "kind": "best_interim_checkpoint",
                            "path": str(best_path),
                            "step": int(interim_step),
                            "metrics": interim_final_metrics,
                        },
                        {
                            "kind": "final_checkpoint",
                            "path": str(current_candidate_path),
                            "step": step,
                            "metrics": current_metrics,
                        },
                    ],
                }
            )
            _write_selection_metadata(best_path, final_selection_payload)
            checkpoint_manager.load(str(current_candidate_path), model, strict=True)
            return

        state.best_eval_loss = selection_value
        state.best_eval_step = step
        final_selection_payload = _selection_payload(
            step=step,
            metrics=current_metrics,
            payload=payload,
            metric_key=metric_key,
            selection_value=selection_value,
            previous_best_value=previous_best_value,
            previous_best_step=previous_best_step,
            initial_eval=False,
            final_eval=True,
            replacement_reason="final checkpoint selected by final selection eval",
        )
        final_selection_payload.update(
            {
                "selected_from_interim_eval": False,
                "final_selection_confirmed": True,
                "selected_checkpoint_path": str(best_path),
                "best_interim_checkpoint": {
                    "path": str(best_path) if interim_selection is not None else None,
                    "step": interim_step,
                    "interim_selection_value": (
                        None if interim_selection is None else interim_selection.get("selection_value")
                    ),
                    "interim_eval_mode": (
                        None if interim_selection is None else interim_selection.get("selection_eval_mode")
                    ),
                    "final_metrics": interim_final_metrics,
                },
                "final_checkpoint": {
                    "path": str(current_candidate_path),
                    "step": step,
                    "final_metrics": current_metrics,
                    "selection_value": selection_value,
                },
                "best_final_eval_confirmed_checkpoint": {
                    "path": str(best_path),
                    "step": step,
                    "metrics": current_metrics,
                },
                "compared_checkpoints": [
                    {
                        "kind": "best_interim_checkpoint",
                        "path": str(best_path) if interim_selection is not None else None,
                        "step": interim_step,
                        "metrics": interim_final_metrics,
                    },
                    {
                        "kind": "final_checkpoint",
                        "path": str(current_candidate_path),
                        "step": step,
                        "metrics": current_metrics,
                    },
                ],
            }
        )
        best_path = checkpoint_manager.save_named(
            best_checkpoint_name,
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state=_checkpoint_extra_state(),
        )
        _write_selection_metadata(best_path, final_selection_payload)

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
        final_selection_pending = final_eval and eval_control is not None and best_checkpoint_name is not None
        if improved and not final_selection_pending:
            state.best_eval_loss = selection_value
            state.best_eval_step = step
            no_improvement_evals = 0
        elif not math.isnan(selection_value) and not final_selection_pending:
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
        if final_eval and eval_control is not None and best_checkpoint_name is not None:
            _choose_final_selection(
                step=step,
                current_metrics=metrics,
                payload=payload,
                metric_key=metric_key,
                selection_value=selection_value,
                evaluator=evaluator,
            )
            last_eval_step = step
            return
        best_path = None
        if improved:
            best_path = _save_best_checkpoint(step)
            if best_path is not None and eval_control is not None and is_main_process():
                selection_payload = _selection_payload(
                    step=step,
                    metrics=metrics,
                    payload=payload,
                    metric_key=metric_key,
                    selection_value=selection_value,
                    previous_best_value=previous_best_value,
                    previous_best_step=previous_best_step,
                    initial_eval=initial_eval,
                    final_eval=final_eval,
                    replacement_reason=(
                        "new best validation metric"
                        if previous_best_value is not None
                        else "first best validation checkpoint"
                    ),
                )
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
            _emit_train_payload(payload)

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
            _maybe_log_batch_provenance_extreme(
                batch,
                raw_loss_value=raw_loss_value,
                micro_step_index=micro_step,
            )
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
    _emit_low_loss_summary()
    return state
