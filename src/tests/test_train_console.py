import json

from train.console import (
    dump_rounded_json,
    format_scalar,
    print_dpo_eval_event,
    print_dpo_train_event,
    round_output_numbers,
)


def test_format_scalar_uses_five_decimals_for_loss_and_eight_for_learning_rate():
    assert format_scalar(0.1234567, key="loss") == "0.12346"
    assert format_scalar(0.000123456, key="lr") == "0.00012346"
    assert format_scalar(0.000123456, key="learning_rate") == "0.00012346"


def test_format_scalar_uses_two_decimals_for_other_numbers():
    assert format_scalar(12.3456, key="perplexity") == "12.35"
    assert format_scalar(98.7654, key="progress_percent") == "98.77"


def test_round_output_numbers_keeps_loss_and_lr_at_their_separate_precisions():
    payload = {
        "loss": 0.1234567,
        "lr": 0.000123456,
        "progress_percent": 98.7654,
        "eval": {
            "loss": 1.234567,
            "perplexity": 12.34567,
        },
        "lm_health": {
            "loss": 0.987654,
        },
        "validation_loss_delta": -0.0012345,
    }

    rounded = round_output_numbers(payload)

    assert rounded["loss"] == 0.12346
    assert rounded["lr"] == 0.00012346
    assert rounded["progress_percent"] == 98.77
    assert rounded["eval"]["loss"] == 1.23457
    assert rounded["eval"]["perplexity"] == 12.35
    assert rounded["lm_health"]["loss"] == 0.98765
    assert rounded["validation_loss_delta"] == -0.00123


def test_dump_rounded_json_applies_the_same_rounding_rules():
    payload = {
        "loss": 0.1234567,
        "lr": 0.000123456,
        "examples": 12,
        "perplexity": 4.56789,
    }

    dumped = dump_rounded_json(payload, indent=2)
    parsed = json.loads(dumped)

    assert parsed == {
        "loss": 0.12346,
        "lr": 0.00012346,
        "examples": 12,
        "perplexity": 4.57,
    }


def test_print_dpo_train_event_uses_semicolon_console_format(capsys):
    print_dpo_train_event(
        {
            "step": 0,
            "loss": 0.69315,
            "lr": 0.00001,
            "train_examples_seen": 8,
            "progress_percent": 85.5,
            "stage_elapsed_sec": 845.77,
            "stage_eta_sec": 143.43,
        }
    )

    captured = capsys.readouterr()

    assert captured.out == (
        "step: 0; loss: 0.69315; lr: 0.00001; train_examples_seen: 8; "
        "progress_percent: 85.5; stage_elapsed_sec: 845.77; stage_eta_sec: 143.43\n"
    )


def test_print_dpo_eval_event_uses_semicolon_console_format_with_samples(capsys):
    print_dpo_eval_event(
        {
            "step": 10,
            "eval": {
                "val_dpo_loss": 4.62134,
                "preference_accuracy": 0.78,
                "mean_margin": 0.143,
                "examples_evaluated": 128,
            },
            "approx_epoch": 0.5,
            "train_dataset_size": 1024,
            "validation_dataset_size": 128,
            "train_examples_seen": 88,
            "validation_examples_evaluated": 128,
            "progress_percent": 99.0,
            "stage_elapsed_sec": 977.14,
            "stage_eta_sec": 9.87,
            "lm_health": {
                "loss": 2.28741,
                "perplexity": 9.85,
            },
            "best_step_so_far": 198,
            "qualitative_samples": [
                {
                    "prompt": "I missed class and need a plan for catching up this week.",
                    "clean_response": "...",
                },
                {
                    "prompt": "What should you ask before recommending between two majors?",
                    "clean_response": "...",
                },
                {
                    "prompt": "If the catalog does not list ECON 404, how should you respond?",
                    "clean_response": "...",
                },
            ],
        }
    )

    captured = capsys.readouterr()

    assert captured.out == (
        "step: 10; eval: {loss: 4.62134; preference_accuracy: 0.78; mean_margin: 0.143; examples_evaluated: 128}; "
        "approx_epoch: 0.5; train_dataset_size: 1024; validation_dataset_size: 128; "
        "train_examples_seen: 88; validation_examples_evaluated: 128; progress_percent: 99.0; "
        "stage_elapsed_sec: 977.14; stage_eta_sec: 9.87; llm_health: {loss: 2.28741; perplexity: 9.85}; "
        "best_step: 198; samples: [\n"
        "  sample1: {prompt: \"I missed class and need a plan for catching up this week.\"; response: \"...\"};\n"
        "  sample2: {prompt: \"What should you ask before recommending between two majors?\"; response: \"...\"};\n"
        "  sample3: {prompt: \"If the catalog does not list ECON 404, how should you respond?\"; response: \"...\"}\n"
        "]\n"
    )
