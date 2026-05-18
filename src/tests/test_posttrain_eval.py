from __future__ import annotations

import json
from pathlib import Path
import types

import pytest
import torch

from config import DataConfig, DataSourceConfig, TokenizerConfig
from data.dataset import DatasetBuilder, IndexedDataset, split_dataset_for_validation
from posttrain.eval import (
    _clean_generated_response,
    assess_raw_lm_sample_behavior,
    assess_sample_behavior,
    ensure_no_regression_prompt_overlap,
    evaluate_pretrain_family_holdouts,
    raw_lm_quality_status,
)
from tokenizer.spm import train_tokenizer


def _make_jsonl_file(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return str(path)


def _base_config(tokenizer_path: str) -> DataConfig:
    return DataConfig(
        tokenizer_path=tokenizer_path,
        sequence_length=32,
        prepared_shard_size=2,
        min_document_chars=1,
        max_document_chars=10000,
    )


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    tokenizer_dir = tmp_path_factory.mktemp("posttrain-eval-tokenizer")
    corpus_path = tokenizer_dir / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "WebbGPT explains courses, planning, and grounded answers clearly.",
                "Students benefit from concise explanations and honest uncertainty.",
                "Preference tuning should reward chosen responses over rejected ones.",
                "Good assistant fine-tuning focuses on helpful and understandable responses.",
            ]
        )
        + "\n"
    )
    return str(
        train_tokenizer(
            [str(corpus_path)],
            TokenizerConfig(
                model_prefix=str(tokenizer_dir / "test-tokenizer"),
                vocab_size=320,
                sample_input_sentence_size=1000,
                max_sentence_length=2048,
            ),
        )
    )


class _DummyDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _HoldoutEvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        loss = input_ids.float().mean() / 100.0
        return types.SimpleNamespace(loss=loss)


def test_split_dataset_for_validation_is_deterministic_and_disjoint():
    dataset = _DummyDataset(10)

    train_a, val_a = split_dataset_for_validation(
        dataset,
        stage_name="sft",
        seed=52,
        validation_fraction=0.2,
        validation_min_examples=2,
    )
    train_b, val_b = split_dataset_for_validation(
        dataset,
        stage_name="sft",
        seed=52,
        validation_fraction=0.2,
        validation_min_examples=2,
    )

    assert isinstance(train_a, IndexedDataset)
    assert isinstance(val_a, IndexedDataset)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert set(train_a.indices).isdisjoint(set(val_a.indices))
    assert sorted(train_a.indices + val_a.indices) == list(range(10))


def test_split_dataset_for_validation_fails_when_validation_is_too_small():
    dataset = _DummyDataset(3)

    with pytest.raises(RuntimeError, match="below the required minimum"):
        split_dataset_for_validation(
            dataset,
            stage_name="preference",
            seed=52,
            validation_fraction=0.25,
            validation_min_examples=4,
        )


def test_split_dataset_for_validation_allows_weak_validation_when_enabled():
    dataset = _DummyDataset(3)

    train_dataset, val_dataset = split_dataset_for_validation(
        dataset,
        stage_name="preference",
        seed=52,
        validation_fraction=0.25,
        validation_min_examples=4,
        allow_weak_validation=True,
    )

    assert isinstance(train_dataset, IndexedDataset)
    assert isinstance(val_dataset, IndexedDataset)
    assert len(train_dataset) == 1
    assert len(val_dataset) == 2


def test_build_sft_split_supports_raw_and_prepared_sources(tmp_path: Path, tokenizer_path: str):
    sft_path = _make_jsonl_file(
        tmp_path / "sft.jsonl",
        [
            {"messages": [{"role": "user", "content": f"Question {index}?"}, {"role": "assistant", "content": f"Answer {index}."}]}
            for index in range(6)
        ],
    )
    config = _base_config(tokenizer_path)
    config.sft_sources = [
        DataSourceConfig(name="sft", path=sft_path, format="jsonl", quality_filter=False, deduplicate=False, pii_scrub=False)
    ]
    builder = DatasetBuilder(config)

    raw_train, raw_val = builder.build_sft_split(
        seed=52,
        validation_fraction=0.25,
        validation_min_examples=2,
        allow_weak_validation=False,
    )

    assert len(raw_train) == 4
    assert len(raw_val) == 2
    assert {example.example_id for example in raw_train.examples}.isdisjoint(
        {example.example_id for example in raw_val.examples}
    )

    manifest_path = tmp_path / "sft-prepared.json"
    builder.prepare_stage("sft", str(manifest_path), force_rebuild=True)
    prepared_config = _base_config(tokenizer_path)
    prepared_config.sft_sources = [DataSourceConfig(name="prepared-sft", path=str(manifest_path), format="prepared")]
    prepared_builder = DatasetBuilder(prepared_config)

    with pytest.raises(RuntimeError, match="require explicit sft_validation_sources"):
        prepared_builder.build_sft_split(
            seed=52,
            validation_fraction=0.25,
            validation_min_examples=2,
            allow_weak_validation=False,
        )

    sft_val_path = _make_jsonl_file(
        tmp_path / "sft-val.jsonl",
        [
            {"messages": [{"role": "user", "content": f"Validation {index}?"}, {"role": "assistant", "content": f"Val answer {index}."}]}
            for index in range(2)
        ],
    )
    prepared_config.sft_validation_sources = [
        DataSourceConfig(
            name="sft-val",
            path=sft_val_path,
            format="jsonl",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    prepared_builder = DatasetBuilder(prepared_config)

    prepared_train, prepared_val = prepared_builder.build_sft_split(
        seed=52,
        validation_fraction=0.25,
        validation_min_examples=2,
        allow_weak_validation=False,
    )

    assert len(prepared_train) == 6
    assert len(prepared_val) == 2


def test_build_preference_split_supports_raw_and_prepared_sources(tmp_path: Path, tokenizer_path: str):
    preference_path = _make_jsonl_file(
        tmp_path / "preference.jsonl",
        [
            {
                "prompt": f"Prompt {index}",
                "chosen": f"Chosen answer {index}",
                "rejected": f"Rejected answer {index}",
            }
            for index in range(6)
        ],
    )
    config = _base_config(tokenizer_path)
    config.preference_sources = [
        DataSourceConfig(
            name="preference",
            path=preference_path,
            format="jsonl",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)

    raw_train, raw_val = builder.build_preference_split(
        seed=52,
        validation_fraction=0.25,
        validation_min_examples=2,
        allow_weak_validation=False,
    )

    assert len(raw_train) == 4
    assert len(raw_val) == 2
    assert {example.example_id for example in raw_train.examples}.isdisjoint(
        {example.example_id for example in raw_val.examples}
    )

    manifest_path = tmp_path / "preference-prepared.json"
    builder.prepare_stage("preference", str(manifest_path), force_rebuild=True)
    prepared_config = _base_config(tokenizer_path)
    prepared_config.preference_sources = [
        DataSourceConfig(name="prepared-preference", path=str(manifest_path), format="prepared")
    ]
    prepared_builder = DatasetBuilder(prepared_config)

    with pytest.raises(RuntimeError, match="require explicit preference_validation_sources"):
        prepared_builder.build_preference_split(
            seed=52,
            validation_fraction=0.25,
            validation_min_examples=2,
            allow_weak_validation=False,
        )

    preference_val_path = _make_jsonl_file(
        tmp_path / "preference-val.jsonl",
        [
            {
                "prompt": f"Validation prompt {index}",
                "chosen": f"Chosen validation answer {index}",
                "rejected": f"Rejected validation answer {index}",
            }
            for index in range(2)
        ],
    )
    prepared_config.preference_validation_sources = [
        DataSourceConfig(
            name="preference-val",
            path=preference_val_path,
            format="jsonl",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    prepared_builder = DatasetBuilder(prepared_config)

    prepared_train, prepared_val = prepared_builder.build_preference_split(
        seed=52,
        validation_fraction=0.25,
        validation_min_examples=2,
        allow_weak_validation=False,
    )

    assert len(prepared_train) == 6
    assert len(prepared_val) == 2


def test_evaluate_dpo_model_reports_loss_accuracy_and_margin(monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    from posttrain import dpo as dpo_module

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *args, **kwargs):
            raise AssertionError("forward should not be called when sequence log probs are monkeypatched")

    policy_model = _DummyModel()
    reference_model = _DummyModel()
    policy_outputs = iter(
        [
            torch.tensor([5.0, 2.0]),
            torch.tensor([1.0, 1.0]),
        ]
    )
    reference_outputs = iter(
        [
            torch.tensor([2.0, 1.0]),
            torch.tensor([1.0, 0.5]),
        ]
    )

    def fake_sequence_log_probs(model, input_ids, attention_mask):
        if model is policy_model:
            return next(policy_outputs)
        return next(reference_outputs)

    monkeypatch.setattr(dpo_module, "_sequence_log_probs", fake_sequence_log_probs)
    batch = {
        "chosen_input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]], dtype=torch.long),
        "rejected_input_ids": torch.tensor([[1, 2, 5], [1, 2, 6]], dtype=torch.long),
        "chosen_attention_mask": torch.ones((2, 3), dtype=torch.long),
        "rejected_attention_mask": torch.ones((2, 3), dtype=torch.long),
    }

    metrics = dpo_module.evaluate_dpo_model(
        policy_model,
        reference_model,
        [batch],
        max_batches=8,
        beta=0.1,
    )

    assert metrics["val_dpo_loss"] >= 0.0
    assert metrics["preference_accuracy"] == 1.0
    assert metrics["mean_margin"] == pytest.approx(1.75)


def test_regression_overlap_guard_rejects_matching_prompt(tmp_path: Path):
    regression_path = _make_jsonl_file(
        tmp_path / "regression.jsonl",
        [{"messages": [{"role": "user", "content": "What is a prerequisite?"}], "tags": ["definition"]}],
    )

    class _Example:
        def __init__(self, messages):
            self.messages = messages

    train_examples = [_Example([{"role": "user", "content": "What is a prerequisite?"}])]
    validation_examples = [_Example([{"role": "user", "content": "Different prompt"}])]

    with pytest.raises(RuntimeError, match="overlaps the training data"):
        ensure_no_regression_prompt_overlap(
            stage_name="sft",
            train_examples=train_examples,
            validation_examples=validation_examples,
            regression_path=regression_path,
        )


def test_assess_sample_behavior_flags_grounded_failures():
    samples = [
        {
            "prompt": "If the catalog does not list ECON 404, how should you respond?",
            "clean_response": "The handbook says ECON 404 is available. [source: handbook]",
            "expected_mode": "abstain",
            "allowed_source_labels": ["course catalog"],
            "forbidden_source_labels": ["handbook"],
            "requires_source_label": False,
        },
        {
            "prompt": "Before comparing two majors for a student, what background questions matter most?",
            "clean_response": "The best major is economics.",
            "expected_mode": "clarify",
            "allowed_source_labels": [],
            "forbidden_source_labels": [],
            "requires_source_label": False,
        },
    ]

    behavior = assess_sample_behavior(samples)

    assert behavior["wrong_source_attribution_count"] == 1
    assert behavior["grounded_abstention_fail_count"] == 1
    assert behavior["clarification_missing_count"] == 1
    assert "source_attribution_failures" in behavior["promotion_blockers"]
    assert "grounded_abstention_failures" in behavior["promotion_blockers"]
    assert behavior["collapse_detected"] is True


def test_assess_raw_lm_sample_behavior_catches_sentence_shaped_drift():
    samples = [
        {
            "id": "domain_catalog_01",
            "bucket": "domain_catalog_prose",
            "probe_type": "domain_readiness",
            "prompt": "This semester course carries one-half credit and is open to juniors and seniors who",
            "clean_response": "have the ability to read and read. The catalog entry should be read for any placement or departmental approval notes. The catalog entry should be read for any placement or departmental approval notes.",
        },
        {
            "id": "narrative_descriptive_01",
            "bucket": "narrative_descriptive_prose",
            "probe_type": "general_legibility",
            "prompt": "By the time the bus reached the edge of town, the rain had",
            "clean_response": "been in the early 1990s when the United States population was first known for the largest city in the world.",
        },
        {
            "id": "everyday_practical_01",
            "bucket": "everyday_practical_prose",
            "probe_type": "general_legibility",
            "prompt": "When a person is trying to recover from a chaotic week, one useful first step is",
            "clean_response": "to improve body health, diet, doctor visits, blood pressure, and symptoms for a child in the home.",
        },
    ]

    behavior = assess_raw_lm_sample_behavior(samples)

    assert behavior["raw_lm_quality_gate_passed"] is False
    assert "prompt_topic_retention_too_low" in behavior["raw_lm_quality_gate_reasons"]
    assert "domain_boilerplate_repetition" in behavior["raw_lm_quality_gate_reasons"]
    assert "boilerplate_repetition" in behavior["raw_lm_quality_gate_reasons"]
    assert "domain_collapse" not in behavior["raw_lm_quality_gate_reasons"]
    assert "narrative_to_expository_drift" in behavior["raw_lm_quality_gate_reasons"]
    assert "everyday_to_medical_drift" in behavior["raw_lm_quality_gate_reasons"]
    assert behavior["domain_phrase_accuracy"] < 1.0
    assert behavior["failure_mode_counts"]["genre_collapse"] >= 2
    assert behavior["per_sample_quality"][0]["max_repeated_4gram_count"] >= 2


def test_assess_raw_lm_sample_behavior_flags_semantic_loops_and_malformed_tokens():
    samples = [
        {
            "id": "domain_catalog_01",
            "bucket": "domain_catalog_prose",
            "probe_type": "domain_readiness",
            "prompt": "This semester course carries one-half credit and is open to juniors and seniors who",
            "clean_response": "This course is an important course for students and the course is a course about the same important thing.",
        },
        {
            "id": "narrative_descriptive_01",
            "bucket": "narrative_descriptive_prose",
            "probe_type": "general_legibility",
            "prompt": "When she opened the old notebook, the first thing she noticed was",
            "clean_response": "the article described the history of the United States population and the largest city in the century.",
        },
        {
            "id": "everyday_practical_01",
            "bucket": "everyday_practical_prose",
            "probe_type": "general_legibility",
            "prompt": "A grocery list usually works better when items are grouped by section of the store, because",
            "clean_response": "the product computer research study data project showed that body health diet infection heart results were important.",
        },
        {
            "id": "neutral_expository_01",
            "bucket": "neutral_expository_prose",
            "probe_type": "general_legibility",
            "prompt": "A short explanation is most useful when it defines the term, gives an example, and",
            "clean_response": "scheduleite Berled inology engagingable pronting over-the-jay unforgetable 'a' or 'b' or 'c' or 'a' or 'b' or 'c'.",
        },
        {
            "id": "neutral_expository_02",
            "bucket": "neutral_expository_prose",
            "probe_type": "general_legibility",
            "prompt": "Clear writing becomes easier to follow when each paragraph stays focused on one main idea and",
            "clean_response": "water river land city war water river land city war water river land city war repeats without adding an idea.",
        },
        {
            "id": "neutral_expository_03",
            "bucket": "neutral_expository_prose",
            "probe_type": "general_legibility",
            "prompt": "A useful explanation should define the main term before adding an example and",
            "clean_response": "the course catalog entry should be read for prerequisites, one-half credit, and departmental approval.",
        },
    ]

    behavior = assess_raw_lm_sample_behavior(samples)
    reasons = set(behavior["raw_lm_quality_gate_reasons"])

    assert behavior["raw_lm_quality_gate_passed"] is False
    assert "domain_collapse" not in reasons
    assert "catalog_advising_drift_into_unrelated_prompt" in reasons
    assert "semantic_repetition" in reasons
    assert "narrative_to_expository_drift" in reasons
    assert "everyday_to_medical_drift" in reasons
    assert "malformed_token_rate_high" in reasons
    assert behavior["semantic_loop_detected"] is True
    assert behavior["repeated_content_bigram_rate"] > 0.0
    assert behavior["repeated_content_trigram_rate"] > 0.0
    assert behavior["malformed_token_sample_count"] >= 1
    assert behavior["per_sample_quality"][3]["malformed_token_count"] >= 7
    assert behavior["per_sample_quality"][4]["semantic_loop_detected"] is True
    assert behavior["per_sample_quality"][5]["catalog_domain_drift"] is True
    assert behavior["failure_mode_counts"]["malformed_generation"] >= 1
    assert behavior["failure_mode_counts"]["semantic_loop"] >= 1


def test_raw_lm_quality_status_caps_major_gate_failures_at_weak():
    almost_improving = {
        "raw_lm_quality_gate_passed": False,
        "raw_lm_quality_gate_reasons": ["semantic_repetition"],
        "aggregate_quality_metrics": {
            "first_40_tokens_legible_rate": 0.9,
            "prompt_topic_retention_rate": 0.8,
            "semantic_drift_score": 0.2,
            "generic_attractor_rate": 0.1,
        },
    }
    clean_long = {
        "raw_lm_quality_gate_passed": True,
        "raw_lm_quality_gate_reasons": [],
        "aggregate_quality_metrics": {
            "first_40_tokens_legible_rate": 0.9,
            "prompt_topic_retention_rate": 0.8,
            "semantic_drift_score": 0.2,
            "generic_attractor_rate": 0.1,
        },
    }

    assert raw_lm_quality_status(almost_improving, clean_long) == "weak_raw_lm"


def test_evaluate_pretrain_family_holdouts_reports_family_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_path: str,
):
    monkeypatch.setattr(
        "posttrain.eval.load_pretrain_family_holdouts",
        lambda *_args, **_kwargs: {
            "general_clean_prose": [
                "Students learn through clear explanations and examples.",
                "A concise paragraph can name the main idea before adding details.",
            ],
            "catalog_grounding_prose": [
                "Prerequisites describe required preparation, while recommendations describe useful background."
            ],
        },
    )

    family_eval = evaluate_pretrain_family_holdouts(
        _HoldoutEvalModel(),
        tokenizer_path,
        sequence_length=32,
    )

    assert set(family_eval["families"]) == {
        "general_clean_prose",
        "catalog_grounding_prose",
    }
    assert family_eval["best_family"] in family_eval["families"]
    assert family_eval["worst_family"] in family_eval["families"]
    assert all("loss" in metrics for metrics in family_eval["families"].values())
    assert family_eval["families"]["general_clean_prose"]["examples_evaluated"] == 2
    assert family_eval["families"]["catalog_grounding_prose"]["examples_evaluated"] == 1
    assert all("windows_evaluated" in metrics for metrics in family_eval["families"].values())
    assert all(metrics["coverage_percent"] == 100.0 for metrics in family_eval["families"].values())
    assert family_eval["coverage"]["family_count"] == 2
    assert family_eval["coverage"]["total_examples_evaluated"] == 3
    assert family_eval["coverage"]["total_windows_evaluated"] >= 3
    assert family_eval["coverage"]["coverage_percent"] == 100.0
    assert family_eval["coverage"]["sequence_length"] == 32


def test_pretrain_family_holdout_files_are_large_clean_raw_prose():
    holdouts_path = Path("data/eval/pretrain_family_holdouts_general.json")
    holdouts = json.loads(holdouts_path.read_text())
    blocked_policy_phrases = [
        "assistant should",
        "the assistant should",
        "grounded answer should",
        "grounded answers should",
        "assistant behavior",
        "assistant-policy",
        "student-facing explanation",
    ]

    assert set(holdouts) == {
        "general_clean_prose",
        "everyday_practical_prose",
        "narrative_descriptive_prose",
        "school_academic_prose",
    }
    for family, path_text in holdouts.items():
        path = Path(path_text)
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        lowered = "\n".join(lines).lower()
        assert len(lines) >= 100, family
        assert not any(phrase in lowered for phrase in blocked_policy_phrases), family


def test_clean_generated_response_keeps_raw_and_strips_special_tokens():
    class _FakeTokenizer:
        pieces = {0: "<pad>", 1: "Hello", 2: "</s>", 3: "<|assistant|>", 4: "world"}

        def token_to_id(self, token):
            for token_id, piece in self.pieces.items():
                if piece == token:
                    return token_id
            raise KeyError(token)

        def id_to_token(self, token_id):
            return self.pieces[token_id]

        def decode(self, token_ids):
            return " ".join(self.pieces[token_id] for token_id in token_ids)

    raw_response, clean_response = _clean_generated_response(_FakeTokenizer(), [1, 3, 4, 2, 4])

    assert raw_response == "Hello <|assistant|> world </s> world"
    assert clean_response == "Hello world"
