from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import data.dataset as dataset_module
from config import DataConfig, DataSourceConfig, TokenizerConfig
from data.dataset import DatasetBuilder
from tokenizer.spm import train_tokenizer


def _make_text_file(path: Path, rows: list[str]) -> str:
    path.write_text("\n".join(rows) + "\n")
    return str(path)


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
    tokenizer_dir = tmp_path_factory.mktemp("resume-tokenizer")
    corpus_path = tokenizer_dir / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "WebbGPT helps students think clearly about courses and planning.",
                "Catalog answers should be grounded, specific, and understandable.",
                "Academic advising benefits from calm explanations and honest uncertainty.",
                "Good assistants avoid gibberish, filler loops, and repeated punctuation.",
                "CS101 usually introduces fundamentals, practice, and steady progression.",
                "Preference tuning should reward clarity, abstention, and grounded behavior.",
            ]
        )
        + "\n"
    )
    model_path = train_tokenizer(
        [str(corpus_path)],
        TokenizerConfig(
            model_prefix=str(tokenizer_dir / "test-tokenizer"),
            vocab_size=320,
            sample_input_sentence_size=1000,
            max_sentence_length=2048,
        ),
    )
    return str(model_path)


def _load_manifest_arrays(manifest: dict) -> list[tuple[np.ndarray, ...]]:
    outputs: list[tuple[np.ndarray, ...]] = []
    for shard in manifest["shards"]:
        if "path" in shard:
            outputs.append((np.load(shard["path"]),))
        elif "input_ids_path" in shard:
            outputs.append((np.load(shard["input_ids_path"]), np.load(shard["labels_path"])))
        else:
            outputs.append(
                (
                    np.load(shard["chosen_input_ids_path"]),
                    np.load(shard["rejected_input_ids_path"]),
                )
            )
    return outputs


def _load_manifest_metadata(manifest: dict) -> list[str]:
    rows: list[str] = []
    for shard in manifest["shards"]:
        metadata_path = shard.get("metadata_path")
        if metadata_path:
            rows.extend(Path(metadata_path).read_text().splitlines())
    return rows


def _assert_same_outputs(left: dict, right: dict) -> None:
    assert left["kind"] == right["kind"]
    assert left["input_fingerprint"] == right["input_fingerprint"]
    assert len(left["shards"]) == len(right["shards"])
    left_arrays = _load_manifest_arrays(left)
    right_arrays = _load_manifest_arrays(right)
    assert len(left_arrays) == len(right_arrays)
    for left_group, right_group in zip(left_arrays, right_arrays):
        assert len(left_group) == len(right_group)
        for left_array, right_array in zip(left_group, right_group):
            assert np.array_equal(left_array, right_array)
    assert _load_manifest_metadata(left) == _load_manifest_metadata(right)


def _assert_same_lm_manifest_summary(left: dict, right: dict) -> None:
    for key in (
        "num_tokens",
        "num_sequences",
        "pad_token_id",
        "eos_token_id",
        "sequence_length",
        "diagnostics",
        "prepare_warnings",
        "domain_realization_gate",
        "corpus_quality_gate",
        "broad_source_quality_gate",
    ):
        assert left.get(key) == right.get(key)


def test_prepare_stage_reuses_completed_manifest(tmp_path: Path, tokenizer_path: str):
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "WebbGPT helps students think through course planning and requirements.",
            "Course catalogs describe prerequisites, credits, and scheduling expectations.",
            "Academic advising conversations should stay clear, grounded, and understandable.",
        ],
    )
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    manifest = builder.prepare_stage("pretrain", str(output_path))
    shard_path = Path(manifest["shards"][0]["path"])
    manifest_mtime = output_path.stat().st_mtime_ns
    shard_mtime = shard_path.stat().st_mtime_ns

    reused = builder.prepare_stage("pretrain", str(output_path))

    assert reused["input_fingerprint"] == manifest["input_fingerprint"]
    assert output_path.stat().st_mtime_ns == manifest_mtime
    assert shard_path.stat().st_mtime_ns == shard_mtime


def test_parallel_prepare_matches_serial_preprocessing_and_tokenization(
    tmp_path: Path,
    tokenizer_path: str,
):
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "WebbGPT helps students think through course planning and requirements with clear grounded prose.",
            "tiny",
            "Course catalogs describe prerequisites, credits, scheduling expectations, and careful advising choices.",
            "WebbGPT helps students think through course planning and requirements with clear grounded prose.",
            "Academic advising conversations should stay clear, grounded, understandable, and specific.",
        ],
    )
    source = DataSourceConfig(
        name="docs",
        path=text_path,
        format="text",
        quality_filter=True,
        deduplicate=True,
        pii_scrub=False,
    )
    serial_config = _base_config(tokenizer_path)
    serial_config.min_document_chars = 20
    serial_config.pretrain_sources = [source]

    parallel_config = DataConfig.from_dict(serial_config.to_dict())
    parallel_config.num_workers = 2

    serial_manifest = DatasetBuilder(serial_config).prepare_stage(
        "pretrain",
        str(tmp_path / "serial" / "pretrain.json"),
        force_rebuild=True,
    )
    parallel_manifest = DatasetBuilder(parallel_config).prepare_stage(
        "pretrain",
        str(tmp_path / "parallel" / "pretrain.json"),
        force_rebuild=True,
    )

    _assert_same_outputs(serial_manifest, parallel_manifest)
    _assert_same_lm_manifest_summary(serial_manifest, parallel_manifest)
    dropped = serial_manifest["diagnostics"]["per_source"][0]["dropped_reasons"]
    assert dropped["too_short"] == 1
    assert dropped["duplicate"] == 1


def test_lm_worker_exception_mentions_source_and_record():
    builder = DatasetBuilder(DataConfig())
    source = DataSourceConfig(name="docs")
    result = dataset_module.LMDocumentProcessResult(
        record_index=7,
        is_text=True,
        error="ValueError: broken tokenization",
    )

    with pytest.raises(RuntimeError, match="source 'docs'.*raw record 7.*broken tokenization"):
        builder._raise_lm_document_worker_error(source, result)


def test_prepare_stage_fails_early_when_completed_manifest_references_missing_shard(
    tmp_path: Path,
    tokenizer_path: str,
):
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "WebbGPT helps students think through course planning and requirements.",
            "Course catalogs describe prerequisites, credits, and scheduling expectations.",
            "Academic advising conversations should stay clear, grounded, and understandable.",
        ],
    )
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    manifest = builder.prepare_stage("pretrain", str(output_path))
    Path(manifest["shards"][0]["path"]).unlink()

    with pytest.raises(RuntimeError, match="Re-run with --force-rebuild or restore the shard directory"):
        builder.prepare_stage("pretrain", str(output_path))


def test_force_rebuild_discards_completed_manifest_with_missing_shard(
    tmp_path: Path,
    tokenizer_path: str,
):
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "WebbGPT helps students think through course planning and requirements.",
            "Course catalogs describe prerequisites, credits, and scheduling expectations.",
            "Academic advising conversations should stay clear, grounded, and understandable.",
        ],
    )
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    manifest = builder.prepare_stage("pretrain", str(output_path))
    missing_path = Path(manifest["shards"][0]["path"])
    missing_path.unlink()

    rebuilt = builder.prepare_stage("pretrain", str(output_path), force_rebuild=True)

    assert rebuilt["kind"] == "packed_lm"
    assert Path(rebuilt["shards"][0]["path"]).exists()


def test_prepare_stage_treats_empty_output_file_as_fresh_target(tmp_path: Path, tokenizer_path: str):
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "WebbGPT helps students think through course planning and requirements.",
            "Course catalogs describe prerequisites, credits, and scheduling expectations.",
            "Academic advising conversations should stay clear, grounded, and understandable.",
        ],
    )
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    output_path.write_text("")

    manifest = builder.prepare_stage("pretrain", str(output_path))

    assert manifest["kind"] == "packed_lm"
    assert output_path.read_text().strip().startswith("{")


def test_prepare_stage_fails_for_legacy_partial_outputs(tmp_path: Path, tokenizer_path: str):
    text_path = _make_text_file(tmp_path / "docs.txt", ["alpha beta gamma delta epsilon"] * 4)
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    shard_dir = output_path.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "legacy-partial.npy").write_bytes(b"legacy")

    with pytest.raises(RuntimeError, match="legacy partial shards are not resumable"):
        builder.prepare_stage("pretrain", str(output_path))


def test_force_rebuild_discards_legacy_partial_outputs(tmp_path: Path, tokenizer_path: str):
    text_path = _make_text_file(tmp_path / "docs.txt", ["alpha beta gamma delta epsilon"] * 6)
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=False,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    shard_dir = output_path.with_suffix("")
    shard_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = shard_dir / "legacy-partial.npy"
    legacy_path.write_bytes(b"legacy")

    manifest = builder.prepare_stage("pretrain", str(output_path), force_rebuild=True)

    assert output_path.exists()
    assert manifest["kind"] == "packed_lm"
    assert not legacy_path.exists()


def test_prepare_stage_resumes_interrupted_packed_lm_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_path: str,
):
    monkeypatch.setattr(dataset_module, "PREPARE_DOC_SNAPSHOT_INTERVAL", 1)
    text_path = _make_text_file(
        tmp_path / "docs.txt",
        [
            "Duplicate text should only appear once in a deduplicated corpus." * 2,
            "Duplicate text should only appear once in a deduplicated corpus." * 2,
            "Academic planning requires understanding prerequisites and sequencing." * 2,
            "Students need grounded explanations that avoid jargon loops." * 2,
            "Catalog grounded assistants should cite concrete course information." * 2,
            "Good conversational behavior includes greeting, clarity, and restraint." * 2,
        ],
    )
    config = _base_config(tokenizer_path)
    config.pretrain_sources = [
        DataSourceConfig(
            name="docs",
            path=text_path,
            format="text",
            quality_filter=False,
            deduplicate=True,
            pii_scrub=False,
        )
    ]
    builder = DatasetBuilder(config)
    output_path = tmp_path / "pretrain.json"
    interrupted = {"done": False}
    original = DatasetBuilder._load_source_records

    def flaky_load(self, source, *, raw_records_consumed: int = 0):
        for index, item in enumerate(original(self, source, raw_records_consumed=raw_records_consumed)):
            if not interrupted["done"] and raw_records_consumed == 0 and index == 2:
                interrupted["done"] = True
                raise KeyboardInterrupt()
            yield item

    monkeypatch.setattr(DatasetBuilder, "_load_source_records", flaky_load)

    with pytest.raises(KeyboardInterrupt):
        builder.prepare_stage("pretrain", str(output_path))

    assert output_path.with_suffix(".resume.json").exists()

    resumed = builder.prepare_stage("pretrain", str(output_path))

    clean_output = tmp_path / "pretrain-clean.json"
    fresh = DatasetBuilder(config).prepare_stage("pretrain", str(clean_output))

    _assert_same_outputs(resumed, fresh)
    assert not output_path.with_suffix(".resume.json").exists()


@pytest.mark.parametrize(
    ("stage", "source_builder"),
    [
        (
            "sft",
            lambda path: (
                "jsonl",
                _make_jsonl_file(
                    path,
                    [
                        {"prompt": "Say hi", "response": "Hi, Harry."},
                        {"prompt": "Explain CS101", "response": "CS101 introduces the basics and expects steady practice."},
                        {"prompt": "What is advising?", "response": "Advising helps students make informed course decisions."},
                        {"prompt": "Good morning", "response": "Good morning. How can I help today?"},
                    ],
                ),
            ),
        ),
        (
            "preference",
            lambda path: (
                "jsonl",
                _make_jsonl_file(
                    path,
                    [
                        {"prompt": "Say hi", "chosen": "Hi, Harry.", "rejected": ",,,,, and and and"},
                        {"prompt": "Explain CS101", "chosen": "CS101 usually covers foundations and builds week by week.", "rejected": "CS101 is and, and, and"},
                        {"prompt": "What if a catalog page is missing?", "chosen": "I’m not seeing that course in the catalog data, so I can’t confirm it.", "rejected": "It definitely exists and meets on Fridays."},
                        {"prompt": "How are you?", "chosen": "I’m doing well and ready to help.", "rejected": ",,,,,,,,,,"},
                    ],
                ),
            ),
        ),
    ],
)
def test_prepare_stage_resumes_interrupted_structured_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    source_builder,
    tokenizer_path: str,
):
    monkeypatch.setattr(dataset_module, "PREPARE_EXAMPLE_SNAPSHOT_INTERVAL", 1)
    data_format, source_path = source_builder(tmp_path / f"{stage}.jsonl")
    config = _base_config(tokenizer_path)
    source = DataSourceConfig(
        name=f"{stage}_source",
        path=source_path,
        format=data_format,
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
    )
    if stage == "sft":
        config.sft_sources = [source]
    else:
        config.preference_sources = [source]

    builder = DatasetBuilder(config)
    output_path = tmp_path / f"{stage}.json"
    interrupted = {"done": False}
    original = DatasetBuilder._load_source_records

    def flaky_load(self, source, *, raw_records_consumed: int = 0):
        for index, item in enumerate(original(self, source, raw_records_consumed=raw_records_consumed)):
            if not interrupted["done"] and raw_records_consumed == 0 and index == 1:
                interrupted["done"] = True
                raise KeyboardInterrupt()
            yield item

    monkeypatch.setattr(DatasetBuilder, "_load_source_records", flaky_load)

    with pytest.raises(KeyboardInterrupt):
        builder.prepare_stage(stage, str(output_path))

    resumed = builder.prepare_stage(stage, str(output_path))
    fresh = DatasetBuilder(config).prepare_stage(stage, str(tmp_path / f"{stage}-fresh.json"))

    _assert_same_outputs(resumed, fresh)
    assert not output_path.with_suffix(".resume.json").exists()
