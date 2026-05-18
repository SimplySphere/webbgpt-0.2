from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import TrainConfig, load_config
from tools.prepare_scale_launch import (
    GpuInfo,
    HardwareInfo,
    LaunchPreparationError,
    build_parser,
    main,
    prepare_launch,
)


GIB = 1024**3


def _hardware(gpu_count: int) -> HardwareInfo:
    return HardwareInfo(
        torch_version="2.6.0-test",
        cuda_available=gpu_count > 0,
        cuda_version="12.4",
        visible_gpu_count=gpu_count,
        gpus=[
            GpuInfo(index=index, name=f"Test GPU {index}", total_memory_bytes=80 * GIB)
            for index in range(gpu_count)
        ],
        disk_free_bytes=2_000 * GIB,
    )


def _write_configs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    model_config = tmp_path / "model-3b.json"
    data_config = tmp_path / "data-3b.json"
    train_config = tmp_path / "train-3b.json"
    run_dir = tmp_path / "run"

    model_config.write_text(
        json.dumps(
            {
                "version": "1.0",
                "name": "webbgpt-3b-test",
                "max_position_embeddings": 8192,
            }
        ),
        encoding="utf-8",
    )
    data_config.write_text(
        json.dumps(
            {
                "version": "1.0",
                "sequence_length": 8192,
                "pretraining_token_budget": 100_000_000,
            }
        ),
        encoding="utf-8",
    )
    train_config.write_text(
        json.dumps(
            {
                "version": "1.0",
                "run_name": "webbgpt-3b-smoke-test",
                "global_batch_size": 64,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "token_budget": 100_000_000,
                "checkpoint": {
                    "output_dir": str(run_dir / "checkpoints" / "pretrain"),
                    "save_every_steps": 100,
                    "keep_last_n": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return model_config, data_config, train_config, run_dir


def _args(
    model_config: Path,
    data_config: Path,
    train_config: Path,
    run_dir: Path,
    *extra: str,
):
    return build_parser().parse_args(
        [
            "--model-config",
            str(model_config),
            "--data-config",
            str(data_config),
            "--train-config",
            str(train_config),
            "--run-dir",
            str(run_dir),
            "--recommended-gpus",
            "8",
            "--desired-global-batch-size",
            "64",
            "--micro-batch-size",
            "1",
            *extra,
        ]
    )


def test_eight_gpu_case_uses_standard_config(tmp_path: Path) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    plan = prepare_launch(
        _args(model_config, data_config, train_config, run_dir),
        hardware=_hardware(8),
    )

    assert plan.standard_config_used is True
    assert plan.auto_config_written is False
    assert plan.auto_config_path is None
    assert plan.command[:4] == ["torchrun", "--nproc_per_node=8", "src/cli.py", "train-pretrain"]
    assert plan.command[-1] == str(train_config)
    assert not (run_dir / "train_config.auto-8gpu.json").exists()


def test_four_gpu_case_generates_grad_accum_16(tmp_path: Path) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    plan = prepare_launch(
        _args(model_config, data_config, train_config, run_dir, "--allow-auto-config"),
        hardware=_hardware(4),
    )

    assert plan.auto_config_written is True
    assert plan.adjusted_gradient_accumulation_steps == 16
    assert plan.auto_config_path == run_dir / "train_config.auto-4gpu.json"
    payload = json.loads(plan.auto_config_path.read_text(encoding="utf-8"))
    assert payload["global_batch_size"] == 64
    assert payload["micro_batch_size"] == 1
    assert payload["gradient_accumulation_steps"] == 16
    assert payload["auto_generated_for_gpu_count"] == 4
    loaded = load_config(plan.auto_config_path, TrainConfig)
    assert loaded.gradient_accumulation_steps == 16
    assert plan.command[-1] == str(plan.auto_config_path)


def test_two_gpu_case_generates_grad_accum_32(tmp_path: Path) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    plan = prepare_launch(
        _args(model_config, data_config, train_config, run_dir, "--allow-auto-config"),
        hardware=_hardware(2),
    )

    assert plan.adjusted_gradient_accumulation_steps == 32
    payload = json.loads(plan.auto_config_path.read_text(encoding="utf-8"))
    assert payload["gradient_accumulation_steps"] == 32
    assert load_config(plan.auto_config_path, TrainConfig).gradient_accumulation_steps == 32


def test_one_gpu_case_generates_grad_accum_64_and_warns(tmp_path: Path) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    plan = prepare_launch(
        _args(model_config, data_config, train_config, run_dir, "--allow-auto-config"),
        hardware=_hardware(1),
    )

    assert plan.adjusted_gradient_accumulation_steps == 64
    assert any("Only 1 CUDA GPU is visible" in warning for warning in plan.warnings)
    payload = json.loads(plan.auto_config_path.read_text(encoding="utf-8"))
    assert payload["gradient_accumulation_steps"] == 64


def test_no_cuda_case_fails_clearly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    exit_code = main(
        [
            "--model-config",
            str(model_config),
            "--data-config",
            str(data_config),
            "--train-config",
            str(train_config),
            "--run-dir",
            str(run_dir),
        ],
        hardware=_hardware(0),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "No visible CUDA GPUs were detected" in captured.err
    assert "not local CPU use" in captured.err


def test_non_divisible_global_batch_suggests_valid_sizes(tmp_path: Path) -> None:
    model_config, data_config, train_config, run_dir = _write_configs(tmp_path)
    args = build_parser().parse_args(
        [
            "--model-config",
            str(model_config),
            "--data-config",
            str(data_config),
            "--train-config",
            str(train_config),
            "--run-dir",
            str(run_dir),
            "--recommended-gpus",
            "8",
            "--desired-global-batch-size",
            "65",
            "--micro-batch-size",
            "1",
            "--allow-auto-config",
        ]
    )

    with pytest.raises(LaunchPreparationError) as exc_info:
        prepare_launch(args, hardware=_hardware(4))

    message = str(exc_info.value)
    assert "not divisible" in message
    assert "Valid nearby global batch sizes" in message
    assert "64" in message
    assert "68" in message
