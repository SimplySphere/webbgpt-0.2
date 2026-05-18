from pathlib import Path
from types import SimpleNamespace

from config import CheckpointConfig, DataConfig, DataSourceConfig, ModelConfig, TrainConfig
import train.entrypoints as entrypoints


def test_run_continued_pretraining_skips_cleanly_when_readiness_fails(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    readiness = {
        "passed": False,
        "failures": ["insufficient_source_families", "single_source_share_too_high"],
        "required_clean_tokens": 4000,
        "audit": {
            "stage": "continue",
            "total_documents": 10,
            "total_clean_tokens": 1000,
            "source_reports": [],
            "source_family_count": 1,
            "max_single_source_token_share": 1.0,
            "max_repeat_rate": 0.0,
            "top_repeated_phrases": [],
        },
    }

    class _FakeBuilder:
        def __init__(self, _config: DataConfig):
            pass

        def assess_continue_readiness(self):
            return readiness

    captured: dict[str, object] = {}

    monkeypatch.setattr(entrypoints, "DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr(entrypoints, "is_main_process", lambda: True)
    monkeypatch.setattr(
        entrypoints,
        "save_stage_summary",
        lambda output_dir, payload: captured.update(
            {"output_dir": output_dir, "payload": payload}
        ),
    )

    data_config = DataConfig(
        tokenizer_path="artifacts/tokenizer/webbgpt.model",
        continued_pretrain_sources=[
            DataSourceConfig(
                name="prepared-continue",
                path="artifacts/runs/local-mvp/prepared/continue.json",
                format="prepared",
            )
        ],
        validation_sources=[
            DataSourceConfig(
                name="prepared-validation",
                path="artifacts/runs/local-mvp/prepared/validation.json",
                format="prepared",
            )
        ],
    )
    train_config = TrainConfig(
        checkpoint=CheckpointConfig(
            output_dir=str(tmp_path / "continue"),
            initialize_from="artifacts/runs/local-mvp/checkpoints/pretrain/step-00019975",
        )
    )

    summary = entrypoints.run_continued_pretraining(ModelConfig(), data_config, train_config)

    assert summary["skipped"] is True
    assert summary["skip_reason"] == "continue_readiness_failed"
    assert summary["continue_readiness"] == readiness
    assert summary["promotion_blockers"] == ["continue_readiness_failed"]
    assert captured["output_dir"] == str(tmp_path / "continue")
    assert captured["payload"] == summary
    assert "skipping continued pretraining because the continue corpus failed readiness checks" in capsys.readouterr().err


def test_run_pretraining_wires_eval_history_control(monkeypatch, tmp_path: Path):
    class _FakeDataset:
        def __init__(self, size: int):
            self.size = size

        def __len__(self):
            return self.size

    class _FakeBuilder:
        def __init__(self, _config: DataConfig):
            pass

        def build_pretrain(self):
            return _FakeDataset(7)

        def build_validation(self):
            return _FakeDataset(3)

    captured: dict[str, object] = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens_seen=128,
            examples_seen=8,
            best_eval_loss=1.25,
            best_eval_step=200,
            nonfinite_loss_steps=0,
            nonfinite_event_samples=[],
            run_mode="max_steps_limited",
            progress_mode="steps",
            scheduler_max_steps=400_000,
            effective_optimizer_steps=400_000,
            prepared_token_target=None,
            prepared_sequence_target=None,
            prepared_token_progress_percent=None,
            prepared_sequence_progress_percent=None,
            final_partial_accumulation_flushed=False,
            final_partial_microbatches=0,
            dataloader_passes_completed=0,
        )

    monkeypatch.setattr(entrypoints, "DatasetBuilder", _FakeBuilder)
    monkeypatch.setattr(entrypoints, "CausalTransformer", lambda _config: object())
    monkeypatch.setattr(entrypoints, "maybe_wrap_fsdp", lambda model, _config: model)
    monkeypatch.setattr(entrypoints, "CheckpointManager", lambda **_kwargs: object())
    monkeypatch.setattr(entrypoints, "build_optimizer", lambda _model, _config: object())
    monkeypatch.setattr(entrypoints, "build_scheduler", lambda _optimizer, _config, **_kwargs: object())
    monkeypatch.setattr(entrypoints, "build_dataloader", lambda dataset, **_kwargs: [dataset])
    monkeypatch.setattr(entrypoints, "run_training", fake_run_training)
    monkeypatch.setattr(entrypoints, "init_distributed", lambda: (0, 1, 0))
    monkeypatch.setattr(entrypoints, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(entrypoints, "is_main_process", lambda: True)
    monkeypatch.setattr(entrypoints, "seed_everything", lambda _seed: {"seed": 52})
    monkeypatch.setattr(entrypoints, "snapshot_configs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(entrypoints, "save_run_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(entrypoints, "save_stage_summary", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "checkpoints" / "pretrain"
    data_config = DataConfig(
        tokenizer_path="artifacts/tokenizer/webbgpt.model",
        pretrain_sources=[
            DataSourceConfig(name="pretrain", path="data/pretrain.jsonl", format="jsonl")
        ],
        validation_sources=[
            DataSourceConfig(name="validation", path="data/validation.jsonl", format="jsonl")
        ],
    )
    train_config = TrainConfig(
        eval_every_steps=200,
        num_eval_batches=8,
        checkpoint=CheckpointConfig(output_dir=str(output_dir)),
    )

    summary = entrypoints.run_pretraining(ModelConfig(), data_config, train_config)

    eval_control = captured["eval_control"]
    assert eval_control.stage_name == "pretrain"
    assert eval_control.evaluate_at_start is False
    assert eval_control.eval_interval_steps == 200
    assert eval_control.validation_max_batches == 8
    assert eval_control.final_validation_max_batches == 8
    assert eval_control.final_eval_full_validation is False
    assert eval_control.train_dataset_size == 7
    assert eval_control.validation_dataset_size == 3
    assert eval_control.eval_history_path == str(output_dir / "eval_history.jsonl")
    assert captured["best_checkpoint_name"] == "best-pretrain"
    assert captured["eval_event_printer"] == entrypoints.print_lm_eval_event
    assert captured["eval_payload_callback"] is not None
    assert captured["run_control"].run_mode == "max_steps_limited"
    assert summary["validation_enabled"] is True
    assert summary["validation_dataset_size"] == 3
    assert summary["best_checkpoint_path"] == str(output_dir / "best-pretrain")


def test_pretrain_eval_payload_uses_raw_lm_sampler(monkeypatch):
    captured: dict[str, object] = {"raw_calls": []}

    def fake_raw_sampler(model, tokenizer_path, **kwargs):
        captured["raw_model"] = model
        captured["raw_tokenizer_path"] = tokenizer_path
        captured["raw_calls"].append(kwargs)
        return [
            {
                "id": "neutral_expository_01",
                "bucket": "neutral_expository_prose",
                "probe_type": "general_legibility",
                "prompt": "A raw prefix",
                "clean_response": "continued text",
            }
        ]

    def fake_chat_sampler(*_args, **_kwargs):
        raise AssertionError("pretrain eval must not use chat-formatted qualitative sampling")

    monkeypatch.setattr(entrypoints, "generate_raw_lm_qualitative_samples", fake_raw_sampler)
    monkeypatch.setattr(entrypoints, "generate_qualitative_samples", fake_chat_sampler)
    monkeypatch.setattr(
        entrypoints,
        "evaluate_pretrain_family_holdouts",
        lambda *_args, **_kwargs: {
            "families": {
                "general_clean_prose": {
                    "loss": 1.0,
                    "examples_evaluated": 100,
                    "windows_evaluated": 100,
                    "coverage_percent": 100.0,
                }
            },
            "best_family": "general_clean_prose",
            "worst_family": "general_clean_prose",
            "coverage": {
                "family_count": 1,
                "total_examples_evaluated": 100,
                "total_windows_evaluated": 100,
                "coverage_percent": 100.0,
                "sequence_length": 512,
            },
        },
    )

    best_family_eval: dict[str, object] = {}
    callback = entrypoints._lm_eval_payload_callback(
        "tokenizer.model",
        regression_path="data/eval/pretrain_general_regression.jsonl",
        stage_name="pretrain",
        sequence_length=512,
        best_family_eval=best_family_eval,
    )

    payload = callback(
        object(),
        200,
        False,
        SimpleNamespace(best_eval_step=200),
        {"loss": 1.0},
    )

    assert captured["raw_tokenizer_path"] == "tokenizer.model"
    assert captured["raw_calls"][0]["regression_path"] == "data/eval/pretrain_general_regression.jsonl"
    assert captured["raw_calls"][0]["limit"] is None
    assert captured["raw_calls"][0]["temperature"] == 0.4
    assert captured["raw_calls"][0]["top_p"] == 0.9
    assert captured["raw_calls"][0]["max_new_tokens"] == 48
    assert captured["raw_calls"][1]["temperature"] == 0.7
    assert captured["raw_calls"][1]["top_p"] == 0.95
    assert captured["raw_calls"][1]["max_new_tokens"] == 128
    assert payload["sample_mode"] == "raw_lm"
    assert payload["sample_decode"]["stable_profile"] == {"temperature": 0.4, "top_p": 0.9, "max_new_tokens": 48}
    assert payload["sample_decode"]["stress_profile"] == {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 128}
    assert payload["samples"] == [
        {
            "id": "neutral_expository_01",
            "bucket": "neutral_expository_prose",
            "probe_type": "general_legibility",
            "prompt": "A raw prefix",
            "response": "continued text",
        }
    ]
    assert payload["short_stable_samples"] == payload["samples"]
    assert payload["long_stress_samples"] == payload["samples"]
    assert payload["raw_lm_quality_gate_passed"] is False
    assert payload["model_quality_status"] == "weak_raw_lm"
    assert payload["qualitative_rubric"] == entrypoints.PRETRAIN_QUALITATIVE_RUBRIC
    assert payload["best_family"] == "general_clean_prose"
    assert payload["family_eval_coverage"]["total_examples_evaluated"] == 100
    assert payload["family_eval_coverage"]["total_windows_evaluated"] == 100
    assert best_family_eval["best_family"] == "general_clean_prose"
    assert best_family_eval["coverage"]["sequence_length"] == 512


def test_non_pretrain_eval_payload_keeps_chat_sampler(monkeypatch):
    captured: dict[str, object] = {}

    def fake_raw_sampler(*_args, **_kwargs):
        raise AssertionError("non-pretrain eval should not use the pretrain raw-LM sampler")

    def fake_chat_sampler(model, tokenizer_path, **kwargs):
        captured["chat_model"] = model
        captured["chat_tokenizer_path"] = tokenizer_path
        captured["chat_kwargs"] = kwargs
        return [{"prompt": "A chat prompt", "clean_response": "assistant text"}]

    monkeypatch.setattr(entrypoints, "generate_raw_lm_qualitative_samples", fake_raw_sampler)
    monkeypatch.setattr(entrypoints, "generate_qualitative_samples", fake_chat_sampler)

    callback = entrypoints._lm_eval_payload_callback(
        "tokenizer.model",
        regression_path="data/eval/continue_regression.jsonl",
        stage_name="continue",
        sequence_length=512,
    )

    payload = callback(
        object(),
        200,
        False,
        SimpleNamespace(best_eval_step=200),
        {"loss": 1.0},
    )

    assert captured["chat_tokenizer_path"] == "tokenizer.model"
    assert captured["chat_kwargs"]["regression_path"] == "data/eval/continue_regression.jsonl"
    assert captured["chat_kwargs"]["temperature"] == 0.0
    assert captured["chat_kwargs"]["top_p"] == 1.0
    assert payload["sample_mode"] == "chat"
    assert payload["sample_decode"] == {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
    assert payload["samples"] == [{"prompt": "A chat prompt", "response": "assistant text"}]
