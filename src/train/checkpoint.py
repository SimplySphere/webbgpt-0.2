from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    import torch

    return torch


def _fsdp_state_dict_context(model):
    torch = _require_torch()
    try:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )
    except Exception:
        return contextlib.nullcontext(), False
    if not isinstance(model, FullyShardedDataParallel):
        return contextlib.nullcontext(), False
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    return FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, config), True


@dataclass(slots=True)
class LoadedCheckpoint:
    step: int
    payload: dict[str, Any]


class CheckpointManager:
    def __init__(self, output_dir: str, keep_last_n: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def _checkpoint_dir(self, step: int) -> Path:
        return self.output_dir / f"step-{step:08d}"

    def _named_checkpoint_dir(self, name: str) -> Path:
        return self.output_dir / name

    def _rng_state(self) -> dict[str, Any]:
        torch = _require_torch()
        state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_rng_state(self, state: dict[str, Any]) -> None:
        torch = _require_torch()
        torch.random.set_rng_state(state["cpu"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save(
        self,
        step: int,
        model,
        optimizer=None,
        scheduler=None,
        dataloader=None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        torch = _require_torch()
        target = self._checkpoint_dir(step)
        return self._save_target(
            target,
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            extra_state=extra_state,
            cleanup_old=True,
        )

    def save_named(
        self,
        name: str,
        step: int,
        model,
        optimizer=None,
        scheduler=None,
        dataloader=None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        return self._save_target(
            self._named_checkpoint_dir(name),
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            extra_state=extra_state,
            cleanup_old=False,
        )

    def _save_target(
        self,
        target: Path,
        *,
        step: int,
        model,
        optimizer=None,
        scheduler=None,
        dataloader=None,
        extra_state: dict[str, Any] | None = None,
        cleanup_old: bool,
    ) -> Path:
        torch = _require_torch()
        tmp_target = target.with_suffix(".tmp")
        if tmp_target.exists():
            shutil.rmtree(tmp_target)
        tmp_target.mkdir(parents=True, exist_ok=True)
        state_context, _ = _fsdp_state_dict_context(model)
        with state_context:
            model_state = model.state_dict()

        payload = {
            "step": step,
            "model": model_state,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "dataloader": dataloader.state_dict() if dataloader is not None and hasattr(dataloader, "state_dict") else None,
            "rng": self._rng_state(),
            "extra_state": extra_state or {},
            "pid": os.getpid(),
        }
        torch.save(payload, tmp_target / "checkpoint.pt")
        checkpoint_metadata = payload["extra_state"].get("checkpoint_metadata")
        if checkpoint_metadata is not None:
            (tmp_target / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2))
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        tmp_target.replace(target)
        if cleanup_old:
            self._cleanup_old_checkpoints()
        return target

    def load(
        self,
        path: str,
        model,
        optimizer=None,
        scheduler=None,
        dataloader=None,
        strict: bool = True,
    ) -> LoadedCheckpoint:
        torch = _require_torch()
        payload = torch.load(Path(path) / "checkpoint.pt", map_location="cpu")
        state_context, _ = _fsdp_state_dict_context(model)
        with state_context:
            model.load_state_dict(payload["model"], strict=strict)
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if dataloader is not None and payload.get("dataloader") is not None and hasattr(dataloader, "load_state_dict"):
            dataloader.load_state_dict(payload["dataloader"])
        self._load_rng_state(payload["rng"])
        return LoadedCheckpoint(step=int(payload["step"]), payload=payload)

    def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(self.output_dir.glob("step-*"))
        if self.keep_last_n <= 0:
            return
        for path in checkpoints[:-self.keep_last_n]:
            shutil.rmtree(path, ignore_errors=True)


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any] | None:
    metadata_path = Path(path) / "checkpoint_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def load_stage_summary(path: str | Path) -> dict[str, Any] | None:
    summary_path = Path(path) / "stage_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def load_artifact_trust(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    for candidate in (
        target / "artifact_trust.json",
        target / "stage_summary.json",
        target / "checkpoint_metadata.json",
    ):
        if candidate.exists():
            payload = json.loads(candidate.read_text())
            return {
                "artifact_status": payload.get("artifact_status", "promotable"),
                "promotion_blockers": list(payload.get("promotion_blockers", [])),
                "promotion_eligible": bool(payload.get("promotion_eligible", False)),
                "source_path": str(candidate),
            }
    return {
        "artifact_status": "promotable",
        "promotion_blockers": [],
        "promotion_eligible": True,
        "source_path": "",
    }


def _infer_stage_from_path(path: str | Path) -> str | None:
    known_stages = {"pretrain", "continue", "sft", "dpo", "preference", "validation"}
    for part in Path(path).parts:
        if part in known_stages:
            return part
    return None


def resolve_parent_lineage(
    checkpoint_path: str | Path | None,
    *,
    nominal_parent_stage: str | None = None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "parent_stage": nominal_parent_stage,
            "parent_checkpoint_path": None,
        }
    target = Path(checkpoint_path)
    payload = load_checkpoint_metadata(target) or load_stage_summary(target) or {}
    actual_stage = payload.get("stage") or _infer_stage_from_path(target)
    lineage = {
        "parent_stage": actual_stage or nominal_parent_stage,
        "parent_checkpoint_path": str(target),
    }
    if nominal_parent_stage and actual_stage and nominal_parent_stage != actual_stage:
        lineage["lineage_via_skipped_stage"] = nominal_parent_stage
    return lineage
