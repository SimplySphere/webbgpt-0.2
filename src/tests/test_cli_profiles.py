import json
import subprocess
import sys
from pathlib import Path

import cli
import pytest

from cli import (
    _apply_remote_run_preset,
    _artifact_trust,
    _can_reuse_tokenizer_corpus,
    _can_reuse_tokenizer_model,
    _prepare_manual_pretrain_train_config,
    _prepare_manual_stage_data_config,
    _prepare_manual_continue_train_config,
    _prepare_manual_dpo_train_config,
    _prepare_manual_sft_train_config,
    _shared_config_pack_profile,
    _require_trusted_artifact,
    _profile_config_paths,
    _stage_train_config,
    _validate_profile_hardware_fit,
    _validate_remote_profile,
)
from config import (
    CheckpointConfig,
    DataConfig,
    DataSourceConfig,
    EvalConfig,
    GroundingConfig,
    ModelConfig,
    ServeConfig,
    TokenizerConfig,
    TrainConfig,
)


def _valid_remote_data_config() -> DataConfig:
    return DataConfig(
        tokenizer_path="artifacts/tokenizer/webbgpt.model",
        pretrain_sources=[
            DataSourceConfig(
                name="pretrain",
                format="hf",
                dataset_name="HuggingFaceFW/fineweb-edu",
                dataset_config_name="sample-10BT",
                split="train",
                streaming=True,
                skip_records=2048,
            )
        ],
        validation_sources=[
            DataSourceConfig(
                name="validation",
                format="hf",
                dataset_name="HuggingFaceFW/fineweb-edu",
                dataset_config_name="sample-10BT",
                split="train",
                streaming=True,
                max_records=2048,
            )
        ],
        continued_pretrain_sources=[
            DataSourceConfig(name="education", path="data/domain/education_corpus.txt", format="text"),
            DataSourceConfig(name="advising", path="data/domain/advising_corpus.txt", format="text"),
        ],
        sft_sources=[
            DataSourceConfig(name="public_sft", path="data/posttrain/sft_public_seed.jsonl", format="jsonl"),
            DataSourceConfig(name="domain_sft", path="data/posttrain/sft_domain_synthetic.jsonl", format="jsonl"),
        ],
        preference_sources=[
            DataSourceConfig(name="public_pref", path="data/posttrain/preference_public_seed.jsonl", format="jsonl"),
            DataSourceConfig(name="domain_pref", path="data/posttrain/preference_domain_synthetic.jsonl", format="jsonl"),
        ],
    )


def _valid_remote_eval_config() -> EvalConfig:
    return EvalConfig(
        benchmark_paths=[
            "data/eval/chat_sanity.jsonl",
            "data/eval/assistant.jsonl",
            "data/eval/webb_course_present.responses",
            "data/eval/webb_course_missing.responses",
            "data/eval/webb_handbook_present.responses",
            "data/eval/webb_handbook_missing.responses",
        ],
        enforce_release_gates=True,
        grounding=GroundingConfig(
            dsn="sqlite:///artifacts/grounding/webbgpt-3b.db",
            seed_url_pack="data/webb/seed_urls_demo.json",
            handbook_url="data/webb/mock/handbook.txt",
            sync_on_start=True,
        ),
    )


def _valid_remote_serve_config() -> ServeConfig:
    return ServeConfig(
        checkpoint_path="artifacts/runs/remote-3b/export/final",
        grounding=GroundingConfig(
            dsn="sqlite:///artifacts/grounding/webbgpt-3b.db",
            seed_url_pack="data/webb/seed_urls_demo.json",
            handbook_url="data/webb/mock/handbook.txt",
        ),
    )


def test_profile_config_paths_cover_local_mvp_and_remote_7b():
    base = Path("/tmp/sample-configs")
    local = _profile_config_paths(base, "local-mvp")
    remote = _profile_config_paths(base, "remote-7b")

    assert local["model"] == base / "model-local-mvp.json"
    assert local["serve"] == base / "serve-local-mvp.json"
    assert remote["model"] == base / "model-7b.json"
    assert remote["tokenizer"] == base / "tokenizer-7b.json"


@pytest.mark.parametrize(
    ("command", "extra_args"),
    [
        ("train-pretrain", []),
        ("train-continue", []),
        ("train-sft", []),
        ("train-dpo", ["--reference-checkpoint", "artifacts/runs/local-mvp/checkpoints/sft/best"]),
    ],
)
def test_manual_train_commands_accept_force_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    extra_args: list[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webbgpt",
            command,
            "--model-config",
            "sample-configs/model-local-mvp.json",
            "--data-config",
            "sample-configs/data-local-mvp.json",
            "--train-config",
            "sample-configs/train-local-mvp.json",
            *extra_args,
            "--force-rebuild",
        ],
    )

    args = cli._parse_args()

    assert args.force_rebuild is True


def test_prepare_manual_stage_data_config_propagates_force_rebuild(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli, "_shared_config_pack_profile", lambda *_args: "local-mvp")
    monkeypatch.setattr(cli, "_uses_profile_runtime_layout", lambda _profile: True)
    monkeypatch.setattr(
        cli,
        "_manual_profile_prepared_manifest_keys",
        lambda *_args, **_kwargs: ["pretrain", "validation"],
    )

    def fake_materialize(
        profile: str,
        data_config: DataConfig,
        manifest_keys: list[str],
        *,
        force_rebuild: bool = False,
    ) -> dict[str, dict[str, object]]:
        captured["profile"] = profile
        captured["manifest_keys"] = list(manifest_keys)
        captured["force_rebuild"] = force_rebuild
        return {
            "pretrain": {"path": "artifacts/runs/local-mvp/prepared/pretrain.json", "manifest": {}},
            "validation": {"path": "artifacts/runs/local-mvp/prepared/validation.json", "manifest": {}},
        }

    monkeypatch.setattr(cli, "_materialize_profile_prepared_manifests", fake_materialize)

    prepared_config = cli._prepare_manual_stage_data_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        DataConfig(),
        TrainConfig(),
        stage_name="pretrain",
        force_rebuild=True,
    )

    assert captured == {
        "profile": "local-mvp",
        "manifest_keys": ["pretrain", "validation"],
        "force_rebuild": True,
    }
    assert prepared_config.pretrain_sources[0].format == "prepared"
    assert prepared_config.validation_sources[0].format == "prepared"


def test_shared_config_pack_profile_accepts_local_mvp_alternate_train_config_name():
    assert (
        _shared_config_pack_profile(
            "sample-configs/model-local-mvp.json",
            "sample-configs/data-local-mvp.json",
            "sample-configs/train-local-mvp-sft-from-pretrain.json",
        )
        == "local-mvp"
    )


def test_prepare_manual_stage_data_config_uses_profile_manifests_for_local_mvp_alt_train_config(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    data_config = DataConfig(
        sft_sources=[DataSourceConfig(name="train", path="data/local/sft.jsonl", format="jsonl")],
        sft_validation_sources=[
            DataSourceConfig(name="validation", path="data/local/sft_validation.jsonl", format="jsonl")
        ],
    )

    def fake_materialize(
        profile: str,
        data_config: DataConfig,
        manifest_keys: list[str],
        *,
        force_rebuild: bool = False,
    ) -> dict[str, dict[str, object]]:
        captured["profile"] = profile
        captured["manifest_keys"] = list(manifest_keys)
        captured["force_rebuild"] = force_rebuild
        return {
            "sft": {"path": "artifacts/runs/local-mvp/prepared/sft.json", "manifest": {}},
            "sft_validation": {
                "path": "artifacts/runs/local-mvp/prepared/sft_validation.json",
                "manifest": {},
            },
        }

    monkeypatch.setattr(cli, "_materialize_profile_prepared_manifests", fake_materialize)

    prepared_config = cli._prepare_manual_stage_data_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp-sft-from-pretrain.json",
        data_config,
        TrainConfig(),
        stage_name="sft",
        force_rebuild=True,
    )

    assert captured == {
        "profile": "local-mvp",
        "manifest_keys": ["sft", "sft_validation"],
        "force_rebuild": True,
    }
    assert prepared_config.sft_sources[0].format == "prepared"
    assert prepared_config.sft_validation_sources[0].format == "prepared"


def test_require_trusted_artifact_rejects_dev_only_without_force(tmp_path: Path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "artifact_status": "dev_only",
                "promotion_blockers": ["generic_refusal_collapse"],
                "promotion_eligible": False,
            }
        )
    )

    trust = _artifact_trust(artifact_dir)
    assert trust["artifact_status"] == "dev_only"
    with pytest.raises(RuntimeError, match="Refusing to export"):
        _require_trusted_artifact(artifact_dir, action="export")


def test_require_trusted_artifact_allows_force_for_debug(tmp_path: Path, capsys):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "artifact_status": "blocked",
                "promotion_blockers": ["lm_health_regressed"],
                "promotion_eligible": False,
            }
        )
    )

    trust = _require_trusted_artifact(artifact_dir, action="serve", force_untrusted=True)

    assert trust["artifact_status"] == "blocked"
    assert "forcing serve" in capsys.readouterr().err.lower()


def test_validate_remote_profile_rejects_placeholder_data():
    data_config = _valid_remote_data_config()
    data_config.sft_sources[0] = DataSourceConfig(
        name="placeholder",
        path="data/local/sft.jsonl",
        format="jsonl",
    )
    with pytest.raises(RuntimeError, match="debug placeholder data"):
        _validate_remote_profile(
            TokenizerConfig(),
            ModelConfig(),
            data_config,
            _valid_remote_eval_config(),
            _valid_remote_serve_config(),
        )


def test_validate_remote_profile_rejects_overlapping_validation():
    data_config = _valid_remote_data_config()
    data_config.validation_sources[0] = DataSourceConfig(
        name="overlap",
        format="hf",
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config_name="sample-10BT",
        split="train",
        streaming=True,
        skip_records=2048,
        max_records=4096,
    )
    with pytest.raises(RuntimeError, match="must not overlap"):
        _validate_remote_profile(
            TokenizerConfig(),
            ModelConfig(),
            data_config,
            _valid_remote_eval_config(),
            _valid_remote_serve_config(),
        )


def test_stage_train_config_preserves_initialize_from_lineage():
    train_config = TrainConfig(run_name="webbgpt-3b")
    stage_config = _stage_train_config(
        train_config,
        stage_name="sft",
        output_dir="artifacts/runs/remote-3b/checkpoints/sft",
        initialize_from="artifacts/runs/remote-3b/checkpoints/continue/step-00050000",
        max_steps=20_000,
    )
    assert stage_config.run_name == "webbgpt-3b-sft"
    assert stage_config.checkpoint.output_dir.endswith("/sft")
    assert stage_config.checkpoint.initialize_from.endswith("/continue/step-00050000")
    assert stage_config.checkpoint.resume_from is None


def test_stage_train_config_applies_continue_overrides():
    train_config = TrainConfig(
        run_name="webbgpt-local-mvp",
        learning_rate=5e-4,
        min_learning_rate=5e-5,
        warmup_steps=200,
        max_steps=20_000,
        continued_learning_rate=1e-4,
        continued_min_learning_rate=1e-5,
        continued_warmup_steps=25,
        continued_max_steps=250,
    )
    stage_config = _stage_train_config(
        train_config,
        stage_name="continue",
        output_dir="artifacts/runs/local-mvp/checkpoints/continue",
        initialize_from="artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000",
    )
    assert stage_config.learning_rate == 1e-4
    assert stage_config.min_learning_rate == 1e-5
    assert stage_config.warmup_steps == 25
    assert stage_config.max_steps == 250


def test_prepare_manual_pretrain_train_config_uses_local_mvp_stage_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    stale_pretrain_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain/step-00001000"
    stale_pretrain_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-115900")

    stage_config = _prepare_manual_pretrain_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(
            run_name="webbgpt-local-mvp",
            checkpoint=CheckpointConfig(output_dir="artifacts/checkpoints-local-mvp"),
        ),
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/pretrain"
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain.stale-20260324-115900/step-00001000"
    ).exists()


def test_prepare_manual_pretrain_train_config_uses_remote_3b_stage_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    stale_pretrain_dir = tmp_path / "artifacts/runs/remote-3b/checkpoints/pretrain/step-00025000"
    stale_pretrain_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-115930")

    stage_config = _prepare_manual_pretrain_train_config(
        "sample-configs/model-3b.json",
        "sample-configs/data-3b.json",
        "sample-configs/train-3b.json",
        TrainConfig(
            run_name="webbgpt-3b",
            checkpoint=CheckpointConfig(output_dir="artifacts/checkpoints"),
        ),
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/remote-3b/checkpoints/pretrain"
    assert (
        tmp_path / "artifacts/runs/remote-3b/checkpoints/pretrain.stale-20260324-115930/step-00025000"
    ).exists()


def test_prepare_manual_continue_train_config_uses_latest_local_mvp_pretrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    pretrain_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    stale_continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue/step-00001050"
    stale_continue_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120000")

    stage_config = _prepare_manual_continue_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp"),
    )

    assert stage_config.checkpoint.initialize_from == "artifacts/runs/local-mvp/checkpoints/pretrain/step-00020000"
    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/continue"
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/continue.stale-20260324-120000/step-00001050"
    ).exists()


def test_prepare_manual_continue_train_config_requires_explicit_lineage_for_legacy_local_mvp_pretrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    legacy_pretrain_dir = tmp_path / "artifacts/checkpoints-local-mvp/step-00019975"
    legacy_pretrain_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="explicit lineage"):
        _prepare_manual_continue_train_config(
            "sample-configs/model-local-mvp.json",
            "sample-configs/data-local-mvp.json",
            "sample-configs/train-local-mvp.json",
            TrainConfig(
                run_name="webbgpt-local-mvp",
                checkpoint=CheckpointConfig(output_dir="artifacts/runs/local-mvp/checkpoints/pretrain"),
            ),
        )


def test_prepare_manual_continue_train_config_requires_explicit_lineage_for_legacy_remote_3b_pretrain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    legacy_pretrain_dir = tmp_path / "artifacts/checkpoints/step-00050000"
    legacy_pretrain_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="explicit lineage"):
        _prepare_manual_continue_train_config(
            "sample-configs/model-3b.json",
            "sample-configs/data-3b.json",
            "sample-configs/train-3b.json",
            TrainConfig(
                run_name="webbgpt-3b",
                checkpoint=CheckpointConfig(output_dir="artifacts/runs/remote-3b/checkpoints/pretrain"),
            ),
        )


def test_prepare_manual_pretrain_train_config_uses_remote_7b_stage_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    stale_pretrain_dir = tmp_path / "artifacts/runs/remote-7b/checkpoints/pretrain/step-00060000"
    stale_pretrain_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-115945")

    stage_config = _prepare_manual_pretrain_train_config(
        "sample-configs/model-7b.json",
        "sample-configs/data-7b.json",
        "sample-configs/train-7b.json",
        TrainConfig(
            run_name="webbgpt-7b",
            checkpoint=CheckpointConfig(output_dir="artifacts/checkpoints-7b"),
        ),
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/remote-7b/checkpoints/pretrain"
    assert (
        tmp_path / "artifacts/runs/remote-7b/checkpoints/pretrain.stale-20260324-115945/step-00060000"
    ).exists()


def test_prepare_manual_sft_train_config_uses_latest_local_mvp_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue/step-00000145"
    continue_dir.mkdir(parents=True, exist_ok=True)
    stale_sft_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/sft/step-00000075"
    stale_sft_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120100")

    stage_config = _prepare_manual_sft_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp", sft_max_steps=3_000),
    )

    assert stage_config.checkpoint.initialize_from == "artifacts/runs/local-mvp/checkpoints/continue/step-00000145"
    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/sft"
    assert stage_config.max_steps == 3_000
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/sft.stale-20260324-120100/step-00000075"
    ).exists()


def test_prepare_manual_sft_train_config_rotates_existing_history_for_alt_local_mvp_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    pretrain_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain/step-00019975"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue"
    continue_dir.mkdir(parents=True, exist_ok=True)
    (continue_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "stage": "continue",
                "skipped": True,
                "skip_reason": "continue_readiness_failed",
                "artifact_status": "dev_only",
                "promotion_blockers": ["continue_readiness_failed"],
                "promotion_eligible": False,
            }
        )
    )
    stage_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/sft"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "eval_history.jsonl").write_text("{}\n")
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120130")

    stage_config = _prepare_manual_sft_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp-sft-from-pretrain.json",
        TrainConfig(
            run_name="webbgpt-local-mvp",
            sft_max_steps=3_000,
            checkpoint=CheckpointConfig(
                initialize_from="artifacts/runs/local-mvp/checkpoints/pretrain/step-00019975",
                output_dir="artifacts/runs/local-mvp/checkpoints/sft",
            ),
        ),
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/sft"
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/sft.stale-20260324-120130/eval_history.jsonl"
    ).exists()


def test_prepare_manual_sft_train_config_falls_back_to_pretrain_when_continue_was_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    pretrain_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/pretrain/step-00019975"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    continue_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/continue"
    continue_dir.mkdir(parents=True, exist_ok=True)
    (continue_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "stage": "continue",
                "skipped": True,
                "skip_reason": "continue_readiness_failed",
                "artifact_status": "dev_only",
                "promotion_blockers": ["continue_readiness_failed"],
                "promotion_eligible": False,
            }
        )
    )

    stage_config = _prepare_manual_sft_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp", sft_max_steps=3_000),
    )

    assert stage_config.checkpoint.initialize_from == "artifacts/runs/local-mvp/checkpoints/pretrain/step-00019975"
    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/sft"
    assert stage_config.max_steps == 3_000


def test_prepare_manual_sft_train_config_requires_explicit_lineage_when_continue_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="Set checkpoint.initialize_from explicitly"):
        _prepare_manual_sft_train_config(
            "sample-configs/model-local-mvp.json",
            "sample-configs/data-local-mvp.json",
            "sample-configs/train-local-mvp.json",
            TrainConfig(run_name="webbgpt-local-mvp", sft_max_steps=3_000),
        )


def test_prepare_manual_dpo_train_config_uses_stage_output_and_dpo_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    stale_dpo_dir = tmp_path / "artifacts/runs/local-mvp/checkpoints/dpo/step-00000050"
    stale_dpo_dir.mkdir(parents=True, exist_ok=True)
    reference_checkpoint = tmp_path / "artifacts/runs/local-mvp/checkpoints/sft/step-00003000"
    reference_checkpoint.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli.time, "strftime", lambda _: "20260324-120200")

    stage_config = _prepare_manual_dpo_train_config(
        "sample-configs/model-local-mvp.json",
        "sample-configs/data-local-mvp.json",
        "sample-configs/train-local-mvp.json",
        TrainConfig(run_name="webbgpt-local-mvp", max_steps=20_000, dpo_max_steps=1_500),
        reference_checkpoint=str(reference_checkpoint.relative_to(tmp_path)),
    )

    assert stage_config.checkpoint.output_dir == "artifacts/runs/local-mvp/checkpoints/dpo"
    assert stage_config.max_steps == 1_500
    assert stage_config.checkpoint.initialize_from is None
    assert (
        tmp_path / "artifacts/runs/local-mvp/checkpoints/dpo.stale-20260324-120200/step-00000050"
    ).exists()


def test_prepare_manual_dpo_train_config_rejects_missing_reference_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="does not exist"):
        _prepare_manual_dpo_train_config(
            "sample-configs/model-local-mvp.json",
            "sample-configs/data-local-mvp.json",
            "sample-configs/train-local-mvp.json",
            TrainConfig(run_name="webbgpt-local-mvp", max_steps=20_000, dpo_max_steps=1_500),
            reference_checkpoint="artifacts/runs/local-mvp/checkpoints/sft/missing",
        )


def test_prepare_manual_stage_data_config_uses_profile_prepared_manifests(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict[str, object] = {}

    def fake_materialize(profile: str, data_config: DataConfig, manifest_keys: list[str], *, force_rebuild: bool = False):
        recorded["profile"] = profile
        recorded["manifest_keys"] = list(manifest_keys)
        recorded["force_rebuild"] = force_rebuild
        return {
            key: {
                "path": f"artifacts/runs/{profile}/prepared/{key}.json",
                "manifest": {"kind": "stub"},
            }
            for key in manifest_keys
        }

    monkeypatch.setattr(cli, "_materialize_profile_prepared_manifests", fake_materialize)

    data_config = DataConfig(
        preference_sources=[DataSourceConfig(name="pref", path="data/pref.jsonl", format="jsonl")],
        preference_validation_sources=[
            DataSourceConfig(name="pref-val", path="data/pref-val.jsonl", format="jsonl")
        ],
        validation_sources=[DataSourceConfig(name="val", path="data/val.txt", format="text")],
    )
    prepared_config = _prepare_manual_stage_data_config(
        "sample-configs/model-3b.json",
        "sample-configs/data-3b.json",
        "sample-configs/train-3b.json",
        data_config,
        TrainConfig(dpo_enable_lm_health_eval=True),
        stage_name="dpo",
    )

    assert recorded["profile"] == "remote-3b"
    assert recorded["manifest_keys"] == ["preference", "preference_validation", "validation"]
    assert prepared_config.preference_sources[0].format == "prepared"
    assert prepared_config.preference_validation_sources[0].format == "prepared"
    assert prepared_config.validation_sources[0].format == "prepared"


def test_apply_remote_run_preset_mvp_reduces_serious_budgets():
    data_config, train_config, eval_config = _apply_remote_run_preset(
        _valid_remote_data_config(),
        TrainConfig(run_name="webbgpt-3b", max_steps=400_000, continued_max_steps=50_000, sft_max_steps=20_000, dpo_max_steps=10_000),
        _valid_remote_eval_config(),
        preset="mvp",
    )
    assert data_config.pretraining_token_budget == 2_500_000_000
    assert data_config.continued_pretraining_token_budget == 350_000_000
    assert train_config.max_steps == 25_000
    assert train_config.continued_max_steps == 8_000
    assert train_config.sft_max_steps == 6_000
    assert train_config.dpo_max_steps == 3_000
    assert eval_config.release_gates.chat_sanity_pass_rate_min == 0.5
    assert train_config.checkpoint.save_every_steps == 250


def test_validate_profile_hardware_fit_rejects_remote_3b_on_low_memory_mac(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli, "_system_memory_bytes", lambda: 16 * 1024**3)

    with pytest.raises(RuntimeError, match="local-mvp"):
        _validate_profile_hardware_fit("remote-3b", ModelConfig())


def test_validate_profile_hardware_fit_rejects_remote_7b_on_non_linux(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli, "_system_memory_bytes", lambda: 512 * 1024**3)

    with pytest.raises(RuntimeError, match="Linux multi-GPU"):
        _validate_profile_hardware_fit(
            "remote-7b",
            ModelConfig(
                name="webbgpt-7b",
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=4096,
            ),
        )


def test_can_reuse_tokenizer_corpus_when_config_matches(tmp_path: Path):
    config = cli.TokenizerCorpusConfig(output_path=str(tmp_path / "corpus.txt"))
    output_path = Path(config.output_path)
    output_path.write_text("hello\n")
    output_path.with_suffix(".txt.meta.json").write_text(
        cli.json.dumps({"config": config.to_dict(), "documents_written": 1})
    )

    assert _can_reuse_tokenizer_corpus(config) is True


def test_can_reuse_tokenizer_model_when_config_matches(tmp_path: Path):
    config = cli.TokenizerConfig(model_prefix=str(tmp_path / "tokenizer"))
    Path(f"{config.model_prefix}.model").write_text("stub")
    Path(f"{config.model_prefix}.vocab").write_text("stub")
    Path(f"{config.model_prefix}.tokenizer.json").write_text(cli.json.dumps(config.to_dict()))

    assert _can_reuse_tokenizer_model(config) is True


def test_main_test_command_invokes_pytest(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert check is False
        assert "env" in kwargs
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sys, "argv", ["webbgpt", "test", "-q", "src/tests/test_config.py"])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main() == 0
    assert calls == [[sys.executable, "-m", "pytest", "-q", "src/tests/test_config.py"]]


def test_main_test_command_uses_safe_subset_when_torch_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:3] == [sys.executable, "-c", "import torch"]:
            return subprocess.CompletedProcess(cmd, 1)
        assert check is False
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sys, "argv", ["webbgpt", "test"])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main() == 0
    assert calls[0] == [sys.executable, "-c", "import torch"]
    assert calls[1] == [sys.executable, "-m", "pytest", *cli.SAFE_DEFAULT_TEST_PATHS]
