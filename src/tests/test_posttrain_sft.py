import pytest

from config import TrainConfig
from posttrain.sft import (
    _compute_sft_schedule,
    _is_severe_sft_qualitative_failure,
    _sft_grounded_stop_warmup_step,
    _should_count_sft_collapse_gate_hit,
    _should_include_sft_samples,
)


@pytest.mark.parametrize(
    ("train_loader_steps", "gradient_accumulation_steps", "max_steps", "expected"),
    [
        (6, 1, 20, (6, 20, 25, 10)),
        (27, 4, 200, (7, 35, 25, 10)),
        (10, 1, 20_000, (10, 50, 25, 10)),
        (21, 8, 30_000, (3, 15, 25, 10)),
    ],
)
def test_compute_sft_schedule_uses_epoch_cap_and_min_eval_interval(
    train_loader_steps: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    expected: tuple[int, int, int, int],
):
    stage_config = TrainConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sft_max_epochs=5,
        sft_evals_per_epoch=4,
        sft_min_eval_interval_steps=25,
    )

    assert _compute_sft_schedule(
        train_loader_steps=train_loader_steps,
        stage_config=stage_config,
    ) == expected


def test_should_include_sft_samples_for_initial_best_periodic_and_final_evals():
    assert _should_include_sft_samples(
        step=0,
        final_eval=False,
        best_eval_step=0,
        sample_every_steps=100,
    )
    assert _should_include_sft_samples(
        step=25,
        final_eval=False,
        best_eval_step=25,
        sample_every_steps=100,
    )
    assert _should_include_sft_samples(
        step=100,
        final_eval=False,
        best_eval_step=25,
        sample_every_steps=100,
    )
    assert _should_include_sft_samples(
        step=35,
        final_eval=True,
        best_eval_step=25,
        sample_every_steps=100,
    )
    assert not _should_include_sft_samples(
        step=10,
        final_eval=False,
        best_eval_step=25,
        sample_every_steps=100,
    )


def test_sft_grounded_stop_warmup_uses_half_epoch_floor():
    assert _sft_grounded_stop_warmup_step(steps_per_epoch=224) == 112
    assert _sft_grounded_stop_warmup_step(steps_per_epoch=1) == 1


def test_severe_sft_failures_still_count_immediately():
    sample_behavior = {
        "collapse_detected": True,
        "blank_count": 0,
        "generic_refusal_count": 0,
        "repetitive_count": 2,
        "wrong_source_attribution_count": 0,
        "unsupported_source_tag_count": 0,
        "grounded_abstention_fail_count": 1,
    }

    assert _is_severe_sft_qualitative_failure(sample_behavior)
    assert _should_count_sft_collapse_gate_hit(
        step=10,
        steps_per_epoch=224,
        sample_behavior=sample_behavior,
    )


def test_grounded_abstention_failures_wait_for_sft_warmup():
    sample_behavior = {
        "collapse_detected": True,
        "blank_count": 0,
        "generic_refusal_count": 0,
        "repetitive_count": 0,
        "wrong_source_attribution_count": 0,
        "unsupported_source_tag_count": 0,
        "grounded_abstention_fail_count": 1,
    }

    assert not _is_severe_sft_qualitative_failure(sample_behavior)
    assert not _should_count_sft_collapse_gate_hit(
        step=56,
        steps_per_epoch=224,
        sample_behavior=sample_behavior,
    )
    assert _should_count_sft_collapse_gate_hit(
        step=112,
        steps_per_epoch=224,
        sample_behavior=sample_behavior,
    )
