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


class _FakeOptimizer:
    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        return None


class _FakeScheduler:
    def step(self):
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
    assert "final_eval" not in payloads[-1]
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

    assert checkpoint_manager.named_saves == [
        (
            "best",
            2,
            {
                "train_state": {
                    "tokens_seen": 6,
                    "examples_seen": 3,
                    "best_eval_loss": 2.0,
                    "best_eval_step": 2,
                    "nonfinite_loss_steps": 0,
                    "nonfinite_event_samples": [],
                }
            },
        ),
        (
            "best",
            4,
            {
                "train_state": {
                    "tokens_seen": 10,
                    "examples_seen": 5,
                    "best_eval_loss": 1.0,
                    "best_eval_step": 4,
                    "nonfinite_loss_steps": 0,
                    "nonfinite_event_samples": [],
                }
            },
        ),
    ]
    assert "final_eval" not in payloads[-1]
    assert payloads[-1]["qualitative_samples"] == [{"prompt": "p6", "raw_response": "r", "clean_response": "r"}]
    assert payloads[-1]["final_eval_seen"] is True
    train_payloads = [json.loads(line) for line in stdout_lines if '"loss"' in line and '"eval"' not in line]
    assert train_payloads
    assert "tokens_seen" in train_payloads[0]
    assert "step_time_sec" in train_payloads[0]
    assert "progress_summary" in train_payloads[0]


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
