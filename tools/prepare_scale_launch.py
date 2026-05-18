from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DataConfig, ModelConfig, TrainConfig, load_config  # noqa: E402


WORLD_SIZE_KEYS = (
    "world_size",
    "expected_world_size",
    "distributed_world_size",
)


class LaunchPreparationError(RuntimeError):
    """Raised when a dry-run scale launch config cannot be prepared."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int


@dataclass(frozen=True)
class HardwareInfo:
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    visible_gpu_count: int
    gpus: list[GpuInfo] = field(default_factory=list)
    disk_free_bytes: int | None = None

    @property
    def total_visible_vram_bytes(self) -> int:
        return sum(gpu.total_memory_bytes for gpu in self.gpus)


@dataclass(frozen=True)
class ConfigSummary:
    configured_global_batch_size: int
    configured_micro_batch_size: int
    configured_gradient_accumulation_steps: int
    configured_world_size: int | None
    inferred_world_size_from_batch: int | None
    expected_effective_batch_size: int | None
    sequence_length: int | None
    max_position_embeddings: int | None
    token_budget: int | None
    checkpoint_output_dir: str | None


@dataclass(frozen=True)
class LaunchPlan:
    hardware: HardwareInfo
    config_summary: ConfigSummary
    command: list[str]
    command_string: str
    train_config_path: Path
    auto_config_path: Path | None
    auto_config_written: bool
    adjusted_gradient_accumulation_steps: int | None
    warnings: list[str]
    standard_config_used: bool


def _bytes_to_gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024 ** 3):.2f} GiB"


def _disk_target(path: Path) -> Path:
    candidate = path
    if path.suffix:
        candidate = path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path.cwd()


def detect_hardware(run_dir: Path) -> HardwareInfo:
    disk_free_bytes: int | None = None
    try:
        disk_free_bytes = shutil.disk_usage(_disk_target(run_dir)).free
    except OSError:
        disk_free_bytes = None

    try:
        import torch
    except Exception:
        return HardwareInfo(
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            visible_gpu_count=0,
            gpus=[],
            disk_free_bytes=disk_free_bytes,
        )

    cuda_available = bool(torch.cuda.is_available())
    visible_gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpus: list[GpuInfo] = []
    if cuda_available:
        for index in range(visible_gpu_count):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=str(props.name),
                    total_memory_bytes=int(props.total_memory),
                )
            )

    return HardwareInfo(
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
        visible_gpu_count=visible_gpu_count,
        gpus=gpus,
        disk_free_bytes=disk_free_bytes,
    )


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_world_size(payload: dict[str, Any]) -> int | None:
    for key in WORLD_SIZE_KEYS:
        if key in payload and payload[key] is not None:
            return int(payload[key])
    return None


def _config_summary(
    *,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    train_payload: dict[str, Any],
) -> ConfigSummary:
    configured_world_size = _configured_world_size(train_payload)
    inferred_world_size = None
    denominator = int(train_config.micro_batch_size * train_config.gradient_accumulation_steps)
    if denominator > 0 and int(train_config.global_batch_size) % denominator == 0:
        inferred_world_size = int(train_config.global_batch_size) // denominator
    expected_effective_batch_size = None
    if configured_world_size is not None:
        expected_effective_batch_size = int(
            train_config.micro_batch_size
            * train_config.gradient_accumulation_steps
            * configured_world_size
        )
    return ConfigSummary(
        configured_global_batch_size=int(train_config.global_batch_size),
        configured_micro_batch_size=int(train_config.micro_batch_size),
        configured_gradient_accumulation_steps=int(train_config.gradient_accumulation_steps),
        configured_world_size=configured_world_size,
        inferred_world_size_from_batch=inferred_world_size,
        expected_effective_batch_size=expected_effective_batch_size,
        sequence_length=int(data_config.sequence_length),
        max_position_embeddings=int(model_config.max_position_embeddings),
        token_budget=(
            int(train_config.token_budget)
            if train_config.token_budget is not None
            else int(data_config.pretraining_token_budget)
        ),
        checkpoint_output_dir=train_config.checkpoint.output_dir,
    )


def _valid_global_batch_suggestions(
    *,
    desired_global_batch_size: int,
    micro_batch_size: int,
    visible_gpu_count: int,
    limit: int = 8,
) -> list[int]:
    step = micro_batch_size * visible_gpu_count
    if step <= 0:
        return []
    center = max(desired_global_batch_size // step, 1)
    candidates = {
        step * multiplier
        for multiplier in range(max(center - limit, 1), center + limit + 1)
        if step * multiplier > 0
    }
    candidates.add(step)
    return sorted(candidates, key=lambda value: (abs(value - desired_global_batch_size), value))[:limit]


def _command(
    *,
    nproc_per_node: int,
    model_config: Path,
    data_config: Path,
    train_config: Path,
) -> list[str]:
    return [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
        "src/cli.py",
        "train-pretrain",
        "--model-config",
        str(model_config),
        "--data-config",
        str(data_config),
        "--train-config",
        str(train_config),
    ]


def _shell_join(parts: list[str]) -> str:
    return " ".join(parts)


def _write_auto_config(
    *,
    original_payload: dict[str, Any],
    output_path: Path,
    source_train_config: Path,
    visible_gpu_count: int,
    recommended_gpus: int,
    desired_global_batch_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    payload = dict(original_payload)
    payload["global_batch_size"] = desired_global_batch_size
    payload["micro_batch_size"] = micro_batch_size
    payload["gradient_accumulation_steps"] = gradient_accumulation_steps
    for key in WORLD_SIZE_KEYS:
        if key in payload:
            payload[key] = visible_gpu_count
    payload["auto_generated_for_gpu_count"] = visible_gpu_count
    payload["auto_generated_recommended_gpus"] = recommended_gpus
    payload["auto_generated_from_train_config"] = str(source_train_config)
    payload["auto_generated_note"] = (
        "Generated by tools/prepare_scale_launch.py. Dry-run helper only; "
        "this file does not prove the model fits in memory."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_launch(args: argparse.Namespace, hardware: HardwareInfo | None = None) -> LaunchPlan:
    model_config_path = Path(args.model_config)
    data_config_path = Path(args.data_config)
    train_config_path = Path(args.train_config)
    run_dir = Path(args.run_dir)

    hardware = hardware or detect_hardware(run_dir)
    if not hardware.cuda_available or hardware.visible_gpu_count <= 0:
        raise LaunchPreparationError(
            "No visible CUDA GPUs were detected. This command is for GPU pretraining, "
            "not local CPU use."
        )

    model_config = load_config(model_config_path, ModelConfig)
    data_config = load_config(data_config_path, DataConfig)
    train_config = load_config(train_config_path, TrainConfig)
    train_payload = _load_payload(train_config_path)
    summary = _config_summary(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config,
        train_payload=train_payload,
    )

    visible_gpu_count = int(hardware.visible_gpu_count)
    recommended_gpus = int(args.recommended_gpus)
    desired_global_batch_size = int(
        args.desired_global_batch_size
        if args.desired_global_batch_size is not None
        else train_config.global_batch_size
    )
    micro_batch_size = int(
        args.micro_batch_size
        if args.micro_batch_size is not None
        else train_config.micro_batch_size
    )
    if desired_global_batch_size <= 0:
        raise LaunchPreparationError("--desired-global-batch-size must be positive.")
    if micro_batch_size <= 0:
        raise LaunchPreparationError("--micro-batch-size must be positive.")

    warnings = [
        "This helper does not guarantee the model fits in memory.",
        "3B/7B pretraining starter configs are 8-GPU recommended.",
        "Sequence length 8192 is expensive when used by the selected configs.",
        "If OOM occurs, consider shorter sequence length, activation checkpointing, FSDP/ZeRO, "
        "lower microbatch, or a smaller smoke config.",
        "Do not run 7B before validating the 3B path.",
    ]
    if visible_gpu_count == 1:
        warnings.append(
            "Only 1 CUDA GPU is visible. A generated config can preserve global batch size "
            "with high gradient accumulation, but 3B at sequence length 8192 may not fit "
            "or may be impractically slow."
        )

    standard_config_used = (
        visible_gpu_count == recommended_gpus
        and not bool(args.force_auto_config)
    )
    if standard_config_used:
        command = _command(
            nproc_per_node=visible_gpu_count,
            model_config=model_config_path,
            data_config=data_config_path,
            train_config=train_config_path,
        )
        return LaunchPlan(
            hardware=hardware,
            config_summary=summary,
            command=command,
            command_string=_shell_join(command),
            train_config_path=train_config_path,
            auto_config_path=None,
            auto_config_written=False,
            adjusted_gradient_accumulation_steps=None,
            warnings=warnings,
            standard_config_used=True,
        )

    if not bool(args.allow_auto_config) and not bool(args.force_auto_config):
        raise LaunchPreparationError(
            f"Visible CUDA GPU count is {visible_gpu_count}, but the recommended count is "
            f"{recommended_gpus}. Pass --allow-auto-config to write a run-specific train config."
        )

    denominator = micro_batch_size * visible_gpu_count
    if desired_global_batch_size % denominator != 0:
        suggestions = _valid_global_batch_suggestions(
            desired_global_batch_size=desired_global_batch_size,
            micro_batch_size=micro_batch_size,
            visible_gpu_count=visible_gpu_count,
        )
        raise LaunchPreparationError(
            "Cannot preserve the requested global batch size: "
            f"{desired_global_batch_size} is not divisible by "
            f"micro_batch_size({micro_batch_size}) * visible_gpu_count({visible_gpu_count}) "
            f"= {denominator}. Valid nearby global batch sizes: "
            f"{', '.join(str(value) for value in suggestions)}."
        )

    grad_accum = desired_global_batch_size // denominator
    auto_config_path = run_dir / f"train_config.auto-{visible_gpu_count}gpu.json"
    _write_auto_config(
        original_payload=train_payload,
        output_path=auto_config_path,
        source_train_config=train_config_path,
        visible_gpu_count=visible_gpu_count,
        recommended_gpus=recommended_gpus,
        desired_global_batch_size=desired_global_batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
    )
    command = _command(
        nproc_per_node=visible_gpu_count,
        model_config=model_config_path,
        data_config=data_config_path,
        train_config=auto_config_path,
    )
    return LaunchPlan(
        hardware=hardware,
        config_summary=summary,
        command=command,
        command_string=_shell_join(command),
        train_config_path=train_config_path,
        auto_config_path=auto_config_path,
        auto_config_written=True,
        adjusted_gradient_accumulation_steps=grad_accum,
        warnings=warnings,
        standard_config_used=False,
    )


def render_report(plan: LaunchPlan) -> str:
    hardware = plan.hardware
    summary = plan.config_summary
    lines = [
        "Scale launch dry run",
        "====================",
        "",
        "Hardware:",
        f"- torch_version: {hardware.torch_version or 'not importable'}",
        f"- cuda_available: {hardware.cuda_available}",
        f"- cuda_version: {hardware.cuda_version or 'unknown'}",
        f"- visible_gpu_count: {hardware.visible_gpu_count}",
    ]
    for gpu in hardware.gpus:
        lines.append(f"- gpu[{gpu.index}]: {gpu.name}, vram={_bytes_to_gib(gpu.total_memory_bytes)}")
    lines.extend(
        [
            f"- total_visible_vram: {_bytes_to_gib(hardware.total_visible_vram_bytes)}",
            f"- disk_free_near_run_dir: {_bytes_to_gib(hardware.disk_free_bytes)}",
            "",
            "Config:",
            f"- configured_global_batch_size: {summary.configured_global_batch_size}",
            f"- configured_micro_batch_size: {summary.configured_micro_batch_size}",
            f"- configured_gradient_accumulation_steps: {summary.configured_gradient_accumulation_steps}",
            f"- configured_or_expected_world_size: {summary.configured_world_size}",
            f"- inferred_world_size_from_batch: {summary.inferred_world_size_from_batch}",
            f"- configured_expected_effective_batch_size: {summary.expected_effective_batch_size}",
            f"- sequence_length: {summary.sequence_length}",
            f"- max_position_embeddings: {summary.max_position_embeddings}",
            f"- token_budget: {summary.token_budget}",
            f"- checkpoint_output_dir: {summary.checkpoint_output_dir}",
            "",
        ]
    )
    if plan.standard_config_used:
        lines.extend(
            [
                "Decision:",
                "- visible GPU count matches the recommended count; using the standard train config as-is.",
                f"- train_config: {plan.train_config_path}",
            ]
        )
    else:
        lines.extend(
            [
                "Decision:",
                "- visible GPU count differs from the recommended count, or auto config was forced.",
                f"- auto_train_config: {plan.auto_config_path}",
                f"- adjusted_gradient_accumulation_steps: {plan.adjusted_gradient_accumulation_steps}",
            ]
        )
    lines.extend(["", "Warnings:"])
    lines.extend(f"- {warning}" for warning in plan.warnings)
    lines.extend(["", "Command to run manually:", plan.command_string, ""])
    lines.append("Training was not started.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect CUDA hardware and prepare a dry-run 3B/7B pretraining launch command. "
            "This helper does not start training."
        )
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--recommended-gpus", type=int, default=8)
    parser.add_argument("--desired-global-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--allow-auto-config", action="store_true")
    parser.add_argument("--force-auto-config", action="store_true")
    return parser


def main(argv: list[str] | None = None, hardware: HardwareInfo | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = prepare_launch(args, hardware=hardware)
    except LaunchPreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(render_report(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
