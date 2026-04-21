from pathlib import Path

from config import CheckpointConfig, DataConfig, ModelConfig, TrainConfig
import posttrain.dpo as dpo_module
from posttrain.dpo import _dpo_scale_blockers


def test_dpo_scale_blockers_require_material_dataset_growth():
    stage_config = TrainConfig(
        dpo_min_train_examples=64,
        dpo_min_validation_examples=16,
    )

    blockers = _dpo_scale_blockers(
        train_examples=25,
        validation_examples=10,
        stage_config=stage_config,
    )

    assert "dpo_train_dataset_too_small" in blockers
    assert "dpo_validation_dataset_too_small" in blockers


def test_dpo_scale_blockers_allow_sufficient_dataset_sizes():
    stage_config = TrainConfig(
        dpo_min_train_examples=64,
        dpo_min_validation_examples=16,
    )

    blockers = _dpo_scale_blockers(
        train_examples=96,
        validation_examples=24,
        stage_config=stage_config,
    )

    assert blockers == []


def test_run_dpo_job_skip_summary_includes_parent_trust_blockers(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(dpo_module, "_require_torch", lambda: object())
    monkeypatch.setattr(dpo_module, "is_main_process", lambda: True)
    captured: dict[str, object] = {}

    class _FakeBuilder:
        def __init__(self, _config: DataConfig):
            pass

        def build_preference_split(self, **_kwargs):
            return list(range(25)), list(range(10))

        def validate_preference_datasets(self, _train_dataset, _validation_dataset):
            return {"promotion_blockers": [], "valid_for_promotion": True}

    monkeypatch.setattr(dpo_module, "DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr(
        dpo_module,
        "load_artifact_trust",
        lambda _path: {
            "artifact_status": "dev_only",
            "promotion_blockers": ["sft_behavior_collapse"],
            "promotion_eligible": False,
        },
    )
    monkeypatch.setattr(
        dpo_module,
        "save_stage_summary",
        lambda output_dir, payload: captured.update({"output_dir": output_dir, "payload": payload}),
    )

    dpo_module.run_dpo_job(
        ModelConfig(),
        DataConfig(),
        TrainConfig(
            dpo_min_train_examples=64,
            dpo_min_validation_examples=16,
            checkpoint=CheckpointConfig(output_dir=str(tmp_path / "dpo")),
        ),
        reference_checkpoint="artifacts/runs/local-mvp/checkpoints/sft/best",
    )

    summary = captured["payload"]
    assert summary["skip_reason"] == "insufficient_preference_scale"
    assert "parent_checkpoint_untrusted" in summary["promotion_blockers"]
    assert "dpo_train_dataset_too_small" in summary["promotion_blockers"]
    assert summary["parent_artifact_status"] == "dev_only"


def test_run_dpo_job_skip_summary_only_lists_scale_blockers_for_trusted_parent(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(dpo_module, "_require_torch", lambda: object())
    monkeypatch.setattr(dpo_module, "is_main_process", lambda: True)
    captured: dict[str, object] = {}

    class _FakeBuilder:
        def __init__(self, _config: DataConfig):
            pass

        def build_preference_split(self, **_kwargs):
            return list(range(25)), list(range(10))

        def validate_preference_datasets(self, _train_dataset, _validation_dataset):
            return {"promotion_blockers": [], "valid_for_promotion": True}

    monkeypatch.setattr(dpo_module, "DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr(
        dpo_module,
        "load_artifact_trust",
        lambda _path: {
            "artifact_status": "promotable",
            "promotion_blockers": [],
            "promotion_eligible": True,
        },
    )
    monkeypatch.setattr(
        dpo_module,
        "save_stage_summary",
        lambda output_dir, payload: captured.update({"output_dir": output_dir, "payload": payload}),
    )

    dpo_module.run_dpo_job(
        ModelConfig(),
        DataConfig(),
        TrainConfig(
            dpo_min_train_examples=64,
            dpo_min_validation_examples=16,
            checkpoint=CheckpointConfig(output_dir=str(tmp_path / "dpo")),
        ),
        reference_checkpoint="artifacts/runs/local-mvp/checkpoints/sft/best",
    )

    summary = captured["payload"]
    assert summary["skip_reason"] == "insufficient_preference_scale"
    assert "parent_checkpoint_untrusted" not in summary["promotion_blockers"]
