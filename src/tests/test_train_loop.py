import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from config import CheckpointConfig, TrainConfig


class _FakeTorch:
    class Tensor:
        pass

    def compile(self, model):
        raise RuntimeError("Dynamo is not supported on Python 3.12+")

    class cuda:
        @staticmethod
        def is_available():
            return False

    class nn:
        class utils:
            @staticmethod
            def clip_grad_norm_(_params, _max_norm):
                return None

    @staticmethod
    def device(name):
        return name

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False


class _FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeMask:
    def __init__(self, tokens):
        self.tokens = tokens

    def sum(self):
        return _FakeValue(self.tokens)


class _FakeLoss:
    def __init__(self, value):
        self.value = value

    def __truediv__(self, divisor):
        return _FakeLoss(self.value / divisor)

    def backward(self):
        return None

    def item(self):
        return self.value


class _FakeParameter:
    def __init__(self):
        self.device = "cpu"


class _FakeModel:
    def __init__(self):
        self._parameter = _FakeParameter()
        self.active_checkpoint = "current"

    def parameters(self):
        return iter([self._parameter])

    def to(self, _device):
        return self

    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, **_batch):
        return types.SimpleNamespace(loss=_FakeLoss(1.0))


class _FakeNonFiniteModel(_FakeModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __call__(self, **_batch):
        self.calls += 1
        if self.calls == 1:
            return types.SimpleNamespace(loss=_FakeLoss(float("nan")))
        return types.SimpleNamespace(loss=_FakeLoss(1.0))


class _FakeLossSequenceModel(_FakeModel):
    def __init__(self, losses):
        super().__init__()
        self.losses = list(losses)

    def __call__(self, **_batch):
        if not self.losses:
            return types.SimpleNamespace(loss=_FakeLoss(1.0))
        return types.SimpleNamespace(loss=_FakeLoss(self.losses.pop(0)))


class _FakeOptimizer:
    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        return None


class _FakeScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return None

    def get_last_lr(self):
        return [1e-4]


class _FakeCheckpointManager:
    def __init__(self):
        self.saved_steps = []
        self.named_saves = []

    def save(self, step, model, optimizer=None, scheduler=None, extra_state=None):
        self.saved_steps.append((step, extra_state))

    def save_named(self, name, step, model, optimizer=None, scheduler=None, extra_state=None):
        self.named_saves.append((name, step, extra_state))
        target = Path("/tmp") / f"webbgpt-{name}"
        target.mkdir(parents=True, exist_ok=True)
        return target


class _StatefulFakeCheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.named_saves = []
        self.loads = []

    def save_named(self, name, step, model, optimizer=None, scheduler=None, extra_state=None):
        self.named_saves.append((name, step, getattr(model, "active_checkpoint", "current"), extra_state))
        target = self.output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save(self, step, model, optimizer=None, scheduler=None, extra_state=None):
        target = self.output_dir / f"step-{step:08d}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def load(self, path, model, optimizer=None, scheduler=None, dataloader=None, strict=True):
        del optimizer, scheduler, dataloader, strict
        target = Path(path)
        self.loads.append(str(target))
        if target.name == "best":
            model.active_checkpoint = "best"
            step = 2
        else:
            model.active_checkpoint = "current"
            step = 5
        return types.SimpleNamespace(step=step, payload={"extra_state": {}})


def test_maybe_compile_model_falls_back_when_dynamo_is_unsupported(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)

    model = object()
    result = train_loop.maybe_compile_model(model, enabled=True)

    assert result is model
    assert "skipping torch.compile" in capsys.readouterr().err


def test_validate_effective_batch_size_rejects_mismatched_global_batch(monkeypatch):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)

    config = TrainConfig(global_batch_size=16, micro_batch_size=1, gradient_accumulation_steps=4)

    with pytest.raises(ValueError, match="global_batch_size does not match"):
        train_loop.validate_effective_batch_size(config)


def test_run_training_emits_terminal_eval_for_short_token_budget_stage(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    eval_calls = []

    def fake_evaluate(_model, _loader, _max_batches):
        eval_calls.append(True)
        return {"loss": 1.5, "perplexity": 4.48}

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "evaluate_language_model", fake_evaluate)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    train_config = TrainConfig(
        run_name="test-short-continue",
        global_batch_size=1,
        max_steps=250,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        warmup_steps=25,
        eval_every_steps=200,
        log_every_steps=10,
        num_eval_batches=2,
        token_budget=3,
        checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test", save_every_steps=1000),
    )

    train_loader = [
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
    ]

    checkpoint_manager = _FakeCheckpointManager()
    state = train_loop.run_training(
        model=_FakeModel(),
        train_loader=train_loader,
        train_config=train_config,
        checkpoint_manager=checkpoint_manager,
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        val_loader=[{"attention_mask": _FakeMask(2)}],
    )

    stdout_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in stdout_lines]

    assert state.step == 2
    assert len(eval_calls) == 1
    assert payloads[-1]["step"] == 2
    assert payloads[-1]["final_eval"] is True
    assert "progress_summary" in payloads[-1]
    assert "elapsed" in payloads[-1]["progress_summary"]
    assert "left" in payloads[-1]["progress_summary"]


def test_run_training_merges_eval_payload_and_saves_best_checkpoint(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    eval_losses = iter([2.0, 1.0, 1.5])

    def fake_evaluate(_model, _loader, _max_batches):
        loss = next(eval_losses)
        return {"loss": loss, "perplexity": loss * 10}

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "evaluate_language_model", fake_evaluate)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    train_config = TrainConfig(
        run_name="test-sft",
        global_batch_size=1,
        max_steps=6,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        warmup_steps=1,
        eval_every_steps=2,
        log_every_steps=10,
        num_eval_batches=2,
        checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test", save_every_steps=1000),
    )
    train_loader = [
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
        {"attention_mask": _FakeMask(2)},
    ]
    checkpoint_manager = _FakeCheckpointManager()

    train_loop.run_training(
        model=_FakeModel(),
        train_loader=train_loader,
        train_config=train_config,
        checkpoint_manager=checkpoint_manager,
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        val_loader=[{"attention_mask": _FakeMask(2)}],
        best_checkpoint_name="best",
        eval_payload_callback=lambda _model, step, final_eval, _state, _metrics: {
            "qualitative_samples": [
                {"prompt": f"p{step}", "raw_response": "r", "clean_response": "r"}
            ],
            "final_eval_seen": final_eval,
        },
    )

    stdout_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    payloads = [json.loads(line) for line in stdout_lines if '"eval"' in line]

    assert [(name, step) for name, step, _extra in checkpoint_manager.named_saves] == [
        ("best", 2),
        ("best", 4),
    ]
    first_train_state = checkpoint_manager.named_saves[0][2]["train_state"]
    second_train_state = checkpoint_manager.named_saves[1][2]["train_state"]
    assert first_train_state["tokens_seen"] == 6
    assert first_train_state["examples_seen"] == 3
    assert first_train_state["best_eval_loss"] == 2.0
    assert first_train_state["best_eval_step"] == 2
    assert first_train_state["run_mode"] == "max_steps_limited"
    assert second_train_state["tokens_seen"] == 10
    assert second_train_state["examples_seen"] == 5
    assert second_train_state["best_eval_loss"] == 1.0
    assert second_train_state["best_eval_step"] == 4
    assert payloads[-1]["final_eval"] is True
    assert payloads[-1]["qualitative_samples"] == [{"prompt": "p6", "raw_response": "r", "clean_response": "r"}]
    assert payloads[-1]["final_eval_seen"] is True
    train_payloads = [json.loads(line) for line in stdout_lines if '"loss"' in line and '"eval"' not in line]
    assert train_payloads
    assert "tokens_seen" in train_payloads[0]
    assert "step_time_sec" in train_payloads[0]
    assert "progress_summary" in train_payloads[0]


def test_final_selection_confirms_interim_best_with_larger_eval(monkeypatch, tmp_path):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    eval_calls = []

    def fake_evaluate(model, _loader, max_batches):
        eval_calls.append((getattr(model, "active_checkpoint", "current"), max_batches))
        if max_batches == 2:
            loss = 0.7 if len(eval_calls) == 1 else 0.9
            return {"loss": loss, "perplexity": loss * 10, "batches_evaluated": 2, "examples_evaluated": 2}
        if getattr(model, "active_checkpoint", "current") == "best":
            return {"loss": 0.6, "perplexity": 6.0, "batches_evaluated": 20, "examples_evaluated": 20}
        return {"loss": 0.8, "perplexity": 8.0, "batches_evaluated": 20, "examples_evaluated": 20}

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    output_dir = tmp_path / "final-selection"
    checkpoint_manager = _StatefulFakeCheckpointManager(output_dir)
    model = _FakeModel()
    state = train_loop.run_training(
        model=model,
        train_loader=[
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
        ],
        train_config=TrainConfig(
            run_name="test-final-selection",
            global_batch_size=1,
            max_steps=5,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_steps=1,
            eval_every_steps=2,
            log_every_steps=100,
            num_eval_batches=2,
            checkpoint=CheckpointConfig(output_dir=str(output_dir), save_every_steps=1000),
        ),
        checkpoint_manager=checkpoint_manager,
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        val_loader=[{"attention_mask": _FakeMask(2)}],
        best_checkpoint_name="best",
        eval_fn=fake_evaluate,
        eval_control=train_loop.EvalControl(
            stage_name="pretrain",
            eval_interval_steps=2,
            validation_max_batches=2,
            final_validation_max_batches=20,
            validation_dataset_size=20,
        ),
        eval_payload_callback=lambda *_args, **_kwargs: {
            "family_eval": {
                "catalog_grounding_prose": {
                    "loss": 1.2,
                    "examples_evaluated": 100,
                    "windows_evaluated": 100,
                    "coverage_percent": 100.0,
                }
            },
            "family_eval_coverage": {
                "family_count": 1,
                "total_examples_evaluated": 100,
                "total_windows_evaluated": 100,
                "coverage_percent": 100.0,
                "sequence_length": 512,
            },
            "best_family": "catalog_grounding_prose",
            "worst_family": "catalog_grounding_prose",
        },
    )

    selection = json.loads((output_dir / "best" / "selection.json").read_text())

    assert state.best_eval_step == 2
    assert state.best_eval_loss == 0.6
    assert model.active_checkpoint == "current"
    assert selection["selection_eval_mode"] == "final_subset"
    assert selection["selection_eval_batches"] == 20
    assert selection["selection_eval_coverage_percent"] == 100.0
    assert selection["selected_from_interim_eval"] is True
    assert selection["final_selection_confirmed"] is True
    assert selection["family_eval_coverage"]["total_windows_evaluated"] == 100
    assert selection["family_eval"]["catalog_grounding_prose"]["examples_evaluated"] == 100
    assert selection["best_interim_checkpoint"]["final_metrics"]["loss"] == 0.6
    assert selection["final_checkpoint"]["final_metrics"]["loss"] == 0.8
    assert [call[1] for call in eval_calls] == [2, 2, 20, 20]


def test_run_training_skips_nonfinite_losses(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    train_config = TrainConfig(
        run_name="test-nonfinite",
        global_batch_size=1,
        max_steps=1,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        warmup_steps=1,
        log_every_steps=1,
        checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test", save_every_steps=1000),
    )
    train_loader = [
        {
            "attention_mask": _FakeMask(2),
            "provenance_json": [
                json.dumps(
                    {
                        "shard_index": 0,
                        "row_index": 0,
                        "source_names": ["general_clean_prose"],
                    }
                )
            ],
        },
        {"attention_mask": _FakeMask(2)},
    ]
    checkpoint_manager = _FakeCheckpointManager()

    state = train_loop.run_training(
        model=_FakeNonFiniteModel(),
        train_loader=train_loader,
        train_config=train_config,
        checkpoint_manager=checkpoint_manager,
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        save_final_checkpoint=True,
        checkpoint_metadata={
            "stage": "pretrain",
            "artifact_status": "promotable",
            "promotion_blockers": [],
            "promotion_eligible": True,
        },
    )

    assert state.step == 1
    assert state.nonfinite_loss_steps == 1
    assert state.nonfinite_event_samples[0]["step"] == 0
    assert state.nonfinite_event_samples[0]["examples_in_batch"] == 1
    assert state.nonfinite_event_samples[0]["provenance"][0]["source_names"] == ["general_clean_prose"]
    saved_metadata = checkpoint_manager.saved_steps[-1][1]["checkpoint_metadata"]
    assert saved_metadata["artifact_status"] == "dev_only"
    assert saved_metadata["promotion_blockers"] == ["nonfinite_loss_seen"]
    assert saved_metadata["nonfinite_loss_steps"] == 1
    assert saved_metadata["nonfinite_event_samples"][0]["step"] == 0
    assert saved_metadata["nonfinite_event_samples"][0]["provenance"][0]["source_names"] == [
        "general_clean_prose"
    ]
    assert "skipping non-finite training loss" in capsys.readouterr().err


def test_to_device_preserves_non_tensor_metadata(monkeypatch):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)

    batch = {
        "attention_mask": _FakeMask(2),
        "provenance_json": ['{"source_names":["general_clean_prose"]}'],
    }

    moved = train_loop._to_device(batch, "cpu")

    assert moved["provenance_json"] == ['{"source_names":["general_clean_prose"]}']


def test_run_training_logs_tiered_low_and_high_loss_batch_provenance(monkeypatch):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    events: list[dict[str, object]] = []
    train_loader = [
        {
            "attention_mask": _FakeMask(11),
            "provenance_json": [
                json.dumps(
                    {
                        "source_names": ["catalog_domain_fixture"],
                        "contributors": [
                            {
                                "source": "catalog_domain_fixture",
                                "family": "catalog_grounding_prose",
                                "document_id": "doc-low",
                            }
                        ],
                        "packed_document_count": 1,
                        "approximate_token_count": 11,
                    }
                )
            ],
        },
        {
            "attention_mask": _FakeMask(13),
            "provenance_json": [
                json.dumps(
                    {
                        "source_names": ["advising_domain_fixture"],
                        "contributors": [
                            {
                                "source": "advising_domain_fixture",
                                "family": "advising_planning_prose",
                                "document_id": "doc-suspicious",
                            }
                        ],
                        "packed_document_count": 1,
                        "approximate_token_count": 13,
                    }
                )
            ],
        },
        {
            "attention_mask": _FakeMask(15),
            "provenance_json": [
                json.dumps(
                    {
                        "source_names": ["domain_lm_fixture"],
                        "contributors": [
                            {
                                "source": "domain_lm_fixture",
                                "family": "webb_domain_seed_prose",
                                "document_id": "doc-broad",
                            }
                        ],
                        "packed_document_count": 1,
                        "approximate_token_count": 15,
                    }
                )
            ],
        },
        {
            "attention_mask": _FakeMask(17),
            "provenance_json": [
                json.dumps(
                    {
                        "source_names": ["fineweb_extension_corpus"],
                        "contributors": [
                            {
                                "source": "fineweb_extension_corpus",
                                "family": "public_prose",
                                "document_id": "doc-high",
                            }
                        ],
                        "packed_document_count": 1,
                        "approximate_token_count": 17,
                    }
                )
            ],
        },
    ]

    state = train_loop.run_training(
        model=_FakeLossSequenceModel([0.04, 0.08, 0.2, 2.5]),
        train_loader=train_loader,
        train_config=TrainConfig(
            run_name="test-provenance-extremes",
            global_batch_size=1,
            max_steps=4,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_steps=1,
            log_every_steps=1000,
            log_batch_provenance_extremes=True,
            severe_low_loss_threshold=0.05,
            suspicious_low_loss_threshold=0.1,
            broad_low_loss_threshold=0.5,
            high_loss_probe_threshold=2.0,
            checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test", save_every_steps=1000),
        ),
        checkpoint_manager=_FakeCheckpointManager(),
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        train_event_printer=events.append,
    )

    provenance_events = [event for event in events if str(event.get("event", "")).endswith("_loss_batch_provenance")]
    assert state.step == 4
    assert [event["event"] for event in provenance_events] == [
        "severe_low_loss_batch_provenance",
        "suspicious_low_loss_batch_provenance",
        "high_loss_batch_provenance",
    ]
    assert provenance_events[0]["loss"] == 0.04
    assert provenance_events[0]["tier"] == "severe"
    assert provenance_events[0]["threshold"] == 0.05
    assert provenance_events[0]["source_names"] == ["catalog_domain_fixture"]
    assert provenance_events[0]["contributors"][0]["document_id"] == "doc-low"
    assert provenance_events[0]["packed_document_count"] == 1
    assert provenance_events[0]["approximate_token_count"] == 11
    assert provenance_events[1]["loss"] == 0.08
    assert provenance_events[1]["tier"] == "suspicious"
    assert provenance_events[1]["threshold"] == 0.1
    assert provenance_events[1]["source_names"] == ["advising_domain_fixture"]
    assert provenance_events[1]["contributors"][0]["document_id"] == "doc-suspicious"
    assert provenance_events[2]["loss"] == 2.5
    assert provenance_events[2]["threshold"] == 2.0
    assert provenance_events[2]["source_names"] == ["fineweb_extension_corpus"]
    assert provenance_events[2]["contributors"][0]["document_id"] == "doc-high"
    assert provenance_events[2]["approximate_token_count"] == 17
    assert state.low_loss_event_count == 3
    assert state.low_loss_events_by_tier == {"severe": 1, "suspicious": 1, "broad": 1}
    assert state.low_loss_events_by_source == {
        "advising_domain_fixture": 1,
        "catalog_domain_fixture": 1,
        "domain_lm_fixture": 1,
    }
    assert state.min_low_loss_event["contributors"][0]["document_id"] == "doc-low"
    summary_events = [
        event for event in events if event.get("event") == "low_loss_batch_provenance_summary"
    ]
    assert len(summary_events) == 1
    assert summary_events[0]["low_loss_event_count"] == 3
    assert summary_events[0]["unique_low_loss_steps"] == 3
    assert summary_events[0]["low_loss_events_by_tier"] == {
        "severe": 1,
        "suspicious": 1,
        "broad": 1,
    }
    assert summary_events[0]["low_loss_events_by_source_by_tier"]["domain_lm_fixture"] == {
        "broad": 1
    }
    assert summary_events[0]["min_low_loss_event"]["contributors"][0]["document_id"] == "doc-low"
    assert summary_events[0]["top_low_loss_sources"] == [
        {"source": "advising_domain_fixture", "count": 1},
        {"source": "catalog_domain_fixture", "count": 1},
        {"source": "domain_lm_fixture", "count": 1},
    ]
    assert summary_events[0]["top_low_loss_contributors"][0]["document_id"] == "doc-broad"


def test_run_training_honors_qualitative_stop_request(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)
    monkeypatch.setattr(
        train_loop,
        "evaluate_language_model",
        lambda _model, _loader, _max_batches: {"loss": 1.0, "perplexity": 2.71},
    )

    state = train_loop.run_training(
        model=_FakeModel(),
        train_loader=[{"attention_mask": _FakeMask(2)}],
        train_config=TrainConfig(
            run_name="test-stop",
            global_batch_size=1,
            max_steps=5,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_steps=1,
            log_every_steps=1,
            checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test-stop", save_every_steps=1000),
        ),
        checkpoint_manager=_FakeCheckpointManager(),
        optimizer=_FakeOptimizer(),
        scheduler=_FakeScheduler(),
        val_loader=[{"attention_mask": _FakeMask(2)}],
        eval_control=train_loop.EvalControl(stage_name="sft", evaluate_at_start=True),
        eval_payload_callback=lambda *_args, **_kwargs: {"should_stop_training": True},
    )

    assert state.step == 0
    assert "qualitative gate requested termination" in capsys.readouterr().err


def test_run_training_one_prepared_pass_does_not_repeat_and_flushes_partial(monkeypatch, capsys):
    fake_checkpoint = types.ModuleType("train.checkpoint")
    fake_checkpoint.CheckpointManager = object
    fake_distributed = types.ModuleType("train.distributed")
    fake_distributed.barrier = lambda: None
    fake_distributed.is_main_process = lambda: True

    monkeypatch.setitem(sys.modules, "train.checkpoint", fake_checkpoint)
    monkeypatch.setitem(sys.modules, "train.distributed", fake_distributed)
    sys.modules.pop("train.loop", None)

    train_loop = importlib.import_module("train.loop")

    def fake_require_torch():
        return _FakeTorch(), None, None

    monkeypatch.setattr(train_loop, "_require_torch", fake_require_torch)
    monkeypatch.setattr(train_loop, "_to_device", lambda batch, _device: batch)
    monkeypatch.setattr(train_loop, "barrier", lambda: None)
    monkeypatch.setattr(train_loop, "is_main_process", lambda: True)

    scheduler = _FakeScheduler()
    state = train_loop.run_training(
        model=_FakeModel(),
        train_loader=[
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
            {"attention_mask": _FakeMask(2)},
        ],
        train_config=TrainConfig(
            run_name="test-one-pass",
            global_batch_size=2,
            max_steps=100,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            warmup_steps=1,
            log_every_steps=1,
            checkpoint=CheckpointConfig(output_dir="/tmp/webbgpt-test-one-pass", save_every_steps=1000),
        ),
        checkpoint_manager=_FakeCheckpointManager(),
        optimizer=_FakeOptimizer(),
        scheduler=scheduler,
        run_control=train_loop.TrainingRunControl(
            run_mode="one_prepared_pass",
            progress_mode="prepared_tokens",
            prepared_token_target=10,
            prepared_sequence_target=5,
            stop_after_one_pass=True,
            flush_final_partial_accumulation=True,
            scheduler_max_steps=3,
            effective_optimizer_steps=3,
        ),
    )

    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert state.step == 3
    assert scheduler.steps == 3
    assert state.tokens_seen == 10
    assert state.examples_seen == 5
    assert state.dataloader_passes_completed == 1
    assert state.final_partial_accumulation_flushed is True
    assert state.final_partial_microbatches == 1
    assert state.prepared_token_progress_percent == 100.0
    assert state.prepared_sequence_progress_percent == 100.0
    assert payloads[-1]["run_mode"] == "one_prepared_pass"
    assert payloads[-1]["prepared_token_progress_percent"] == 100.0
