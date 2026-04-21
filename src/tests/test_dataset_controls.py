import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from config import DataConfig, DataSourceConfig
import data.dataset as dataset_module
from data.dataset import DatasetBuilder
from data.schemas import DocumentRecord
from data.schemas import PreferenceExample


def _normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _configured_text_window_hashes(source: DataSourceConfig) -> set[str]:
    rows: list[str] = []
    paths = list(source.paths) if source.paths else [source.path]
    for path in paths:
        rows.extend(line.strip() for line in Path(path).read_text(errors="replace").splitlines() if line.strip())
    start = min(int(source.skip_records), len(rows))
    stop = len(rows) if source.max_records is None else min(start + int(source.max_records), len(rows))
    return {_normalized_text_hash(row) for row in rows[start:stop]}


def test_local_mvp_pretrain_sources_have_zero_exact_validation_overlap():
    data_config = DataConfig.from_dict(json.loads(Path("sample-configs/data-local-mvp.json").read_text()))
    pretrain_hashes: set[str] = set()
    validation_hashes: set[str] = set()

    for source in data_config.pretrain_sources:
        if source.format == "text":
            pretrain_hashes.update(_configured_text_window_hashes(source))
    for source in data_config.validation_sources:
        if source.format == "text":
            validation_hashes.update(_configured_text_window_hashes(source))

    assert pretrain_hashes
    assert validation_hashes
    assert pretrain_hashes & validation_hashes == set()


def test_local_mvp_large_pretrain_sources_do_not_contain_each_other():
    data_config = DataConfig.from_dict(json.loads(Path("sample-configs/data-local-mvp.json").read_text()))
    large_sources = {
        source.name: _configured_text_window_hashes(source)
        for source in data_config.pretrain_sources
        if source.format == "text" and not source.paths
    }
    large_sources = {name: hashes for name, hashes in large_sources.items() if len(hashes) >= 1_000}

    assert set(large_sources) >= {"local_mvp_pretrain_corpus", "fineweb_extension_corpus"}
    for left_name, left_hashes in large_sources.items():
        for right_name, right_hashes in large_sources.items():
            if left_name == right_name:
                continue
            overlap = len(left_hashes & right_hashes)
            smaller_size = min(len(left_hashes), len(right_hashes))
            assert overlap / max(smaller_size, 1) < 0.98


def test_source_hits_share_cap_trips_for_dominant_source():
    builder = DatasetBuilder(
        DataConfig(lm_weighted_source_token_budget=100, lm_max_source_token_share=0.6)
    )
    dominant = {
        "source": DataSourceConfig(name="dominant", format="text", weight=4.0),
        "audit": {"kept_tokens": 80},
    }
    smaller = {
        "source": DataSourceConfig(name="smaller", format="text", weight=1.0),
        "audit": {"kept_tokens": 20},
    }

    assert builder._source_hits_share_cap(dominant, [dominant, smaller]) is True
    assert builder._source_hits_share_cap(smaller, [dominant, smaller]) is False


def test_source_hits_repeat_cap_trips_for_repeating_source():
    builder = DatasetBuilder(
        DataConfig(lm_max_source_repeat_rate=0.1)
    )
    repeating = {
        "source": DataSourceConfig(name="repeating", format="text", weight=2.0),
        "audit": {"kept_documents": 20, "repeated_documents": 3},
    }
    other = {
        "source": DataSourceConfig(name="other", format="text", weight=1.0),
        "audit": {"kept_documents": 10, "repeated_documents": 0},
    }

    assert builder._source_hits_repeat_cap(repeating, [repeating, other]) is True
    assert builder._source_hits_repeat_cap(other, [repeating, other]) is False


def test_weighted_iter_prefers_under_target_source_share(monkeypatch):
    builder = DatasetBuilder(DataConfig())
    dominant = DataSourceConfig(name="dominant", format="text", weight=3.0)
    smaller = DataSourceConfig(name="smaller", format="text", weight=1.0)

    def build_state(source: DataSourceConfig, docs: int):
        audit = {"kept_tokens": 0, "kept_documents": 0, "repeated_documents": 0}

        def iterator():
            for index in range(docs):
                audit["kept_documents"] += 1
                audit["kept_tokens"] += 10
                yield (
                    DocumentRecord(text=f"{source.name}-{index}", source=source.name),
                    [index] * 10,
                )

        return {"source": source, "audit": audit, "iterator": iter(iterator())}

    states = [build_state(dominant, 12), build_state(smaller, 12)]
    monkeypatch.setattr(builder, "_weighted_lm_source_iterators", lambda *args, **kwargs: states)

    emitted = [
        source.name
        for source, _document, _token_ids, _audit in builder._iter_weighted_tokenized_documents(
            [dominant, smaller],
            tokenizer=None,  # type: ignore[arg-type]
        )
    ][:12]
    dominant_share = emitted.count("dominant") / max(len(emitted), 1)

    assert len(emitted) == 12
    assert dominant_share >= 0.5
    assert dominant_share < 0.85


def test_weighted_iter_restarts_underfilled_source_within_repeat_cap(monkeypatch):
    builder = DatasetBuilder(DataConfig(lm_max_source_repeat_rate=0.5))
    dominant = DataSourceConfig(name="dominant", format="text", weight=1.0)
    minor = DataSourceConfig(name="minor", format="text", weight=1.0)

    def build_state(source: DataSourceConfig, docs: int, target_share: float):
        audit = {
            "kept_tokens": 0,
            "kept_documents": 0,
            "repeated_documents": 0,
            "target_share": target_share,
            "unique_document_ids": set(),
            "restart_count": 0,
        }
        progress = {"raw_records_consumed": 0, "accepted_records": 0, "restart_count": 0}

        def iterator_factory(*, raw_records_consumed: int, allow_reentry: bool, max_kept_documents=None):
            del allow_reentry

            def iterator():
                kept_in_cycle = 0
                for index in range(raw_records_consumed, docs):
                    document_id = f"{source.name}-{index}"
                    progress["raw_records_consumed"] = index + 1
                    progress["accepted_records"] += 1
                    audit["kept_documents"] += 1
                    audit["kept_tokens"] += 10
                    if document_id in audit["unique_document_ids"]:
                        audit["repeated_documents"] += 1
                    else:
                        audit["unique_document_ids"].add(document_id)
                    yield (
                        DocumentRecord(
                            text=f"{source.name}-{index}",
                            source=source.name,
                            document_id=document_id,
                        ),
                        [index] * 10,
                    )
                    kept_in_cycle += 1
                    if max_kept_documents is not None and kept_in_cycle >= max_kept_documents:
                        return

            return iterator()

        return {
            "source": source,
            "audit": audit,
            "progress": progress,
            "iterator_factory": iterator_factory,
            "iterator": iter(
                iterator_factory(
                    raw_records_consumed=0,
                    allow_reentry=False,
                    max_kept_documents=None,
                )
            ),
        }

    states = [build_state(dominant, 12, 0.5), build_state(minor, 4, 0.5)]
    monkeypatch.setattr(builder, "_weighted_lm_source_iterators", lambda *args, **kwargs: states)

    emitted = [
        source.name
        for source, _document, _token_ids, _audit in builder._iter_weighted_tokenized_documents(
            [dominant, minor],
            tokenizer=None,  # type: ignore[arg-type]
        )
    ][:16]
    minor_state = next(state for state in states if state["source"].name == "minor")

    assert emitted.count("minor") > 4
    assert emitted.count("minor") <= 8
    assert minor_state["audit"]["restart_count"] == 1
    assert minor_state["audit"]["repeated_documents"] == emitted.count("minor") - 4
    assert minor_state["audit"]["repeated_documents"] / minor_state["audit"]["kept_documents"] <= 0.5


def test_weighted_iter_does_not_restart_when_repeat_budget_is_exhausted(monkeypatch):
    builder = DatasetBuilder(DataConfig(lm_max_source_repeat_rate=0.1))
    dominant = DataSourceConfig(name="dominant", format="text", weight=1.0)
    minor = DataSourceConfig(name="minor", format="text", weight=1.0)

    def build_state(source: DataSourceConfig, docs: int, target_share: float):
        audit = {
            "kept_tokens": 0,
            "kept_documents": 0,
            "repeated_documents": 0,
            "target_share": target_share,
            "unique_document_ids": set(),
            "restart_count": 0,
        }
        progress = {"raw_records_consumed": 0, "accepted_records": 0, "restart_count": 0}

        def iterator_factory(*, raw_records_consumed: int, allow_reentry: bool, max_kept_documents=None):
            del allow_reentry

            def iterator():
                kept_in_cycle = 0
                for index in range(raw_records_consumed, docs):
                    document_id = f"{source.name}-{index}"
                    progress["raw_records_consumed"] = index + 1
                    progress["accepted_records"] += 1
                    audit["kept_documents"] += 1
                    audit["kept_tokens"] += 10
                    if document_id in audit["unique_document_ids"]:
                        audit["repeated_documents"] += 1
                    else:
                        audit["unique_document_ids"].add(document_id)
                    yield (
                        DocumentRecord(
                            text=f"{source.name}-{index}",
                            source=source.name,
                            document_id=document_id,
                        ),
                        [index] * 10,
                    )
                    kept_in_cycle += 1
                    if max_kept_documents is not None and kept_in_cycle >= max_kept_documents:
                        return

            return iterator()

        return {
            "source": source,
            "audit": audit,
            "progress": progress,
            "iterator_factory": iterator_factory,
            "iterator": iter(
                iterator_factory(
                    raw_records_consumed=0,
                    allow_reentry=False,
                    max_kept_documents=None,
                )
            ),
        }

    states = [build_state(dominant, 12, 0.5), build_state(minor, 4, 0.5)]
    monkeypatch.setattr(builder, "_weighted_lm_source_iterators", lambda *args, **kwargs: states)

    emitted = [
        source.name
        for source, _document, _token_ids, _audit in builder._iter_weighted_tokenized_documents(
            [dominant, minor],
            tokenizer=None,  # type: ignore[arg-type]
        )
    ][:12]
    minor_state = next(state for state in states if state["source"].name == "minor")

    assert emitted.count("minor") == 4
    assert minor_state["audit"]["restart_count"] == 0
    assert minor_state["audit"]["repeated_documents"] == 0


def test_assess_continue_readiness_reports_failures(monkeypatch):
    builder = DatasetBuilder(DataConfig(continued_pretraining_token_budget=1_000))
    monkeypatch.setattr(
        builder,
        "audit_lm_stage",
        lambda _stage: {
            "stage": "continue",
            "total_documents": 25,
            "total_clean_tokens": 100,
            "source_reports": [],
            "source_family_count": 1,
            "max_single_source_token_share": 0.9,
            "max_repeat_rate": 0.5,
            "top_repeated_phrases": [],
        },
    )

    readiness = builder.assess_continue_readiness()

    assert readiness["passed"] is False
    assert "insufficient_clean_tokens" in readiness["failures"]
    assert "insufficient_documents" in readiness["failures"]
    assert "insufficient_source_families" in readiness["failures"]
    assert "single_source_share_too_high" in readiness["failures"]
    assert "repeat_rate_too_high" in readiness["failures"]


def test_assess_continue_readiness_supports_prepared_continue_manifest(tmp_path: Path):
    manifest_path = tmp_path / "continue.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "stage": "continue",
                "kind": "packed_lm",
                "diagnostics": {
                    "per_source": [
                        {
                            "source": "domain_mix",
                            "family": "domain_mix",
                            "weight": 2.5,
                            "raw_records": 20,
                            "kept_documents": 5,
                            "kept_tokens": 400,
                            "repeated_documents": 0,
                            "repeat_rate": 0.0,
                            "dropped_reasons": {},
                        },
                        {
                            "source": "general_refresh",
                            "family": "general_refresh",
                            "weight": 1.0,
                            "raw_records": 100,
                            "kept_documents": 10,
                            "kept_tokens": 600,
                            "repeated_documents": 0,
                            "repeat_rate": 0.0,
                            "dropped_reasons": {},
                        },
                    ],
                    "source_family_count": 2,
                    "max_single_source_token_share": 0.6,
                    "max_repeat_rate": 0.0,
                    "top_repeated_phrases": [
                        {"phrase": "example phrase", "count": 3},
                    ],
                },
            }
        )
    )
    builder = DatasetBuilder(
        DataConfig(
            continued_pretraining_token_budget=1_000,
            continue_readiness_min_clean_token_fraction=0.5,
            continue_readiness_min_documents=10,
            continue_readiness_min_source_families=2,
            continue_readiness_max_single_source_share=0.7,
            continue_readiness_max_repeat_rate=0.1,
            continued_pretrain_sources=[
                DataSourceConfig(
                    name="prepared-continue",
                    path=str(manifest_path),
                    format="prepared",
                )
            ],
        )
    )

    readiness = builder.assess_continue_readiness()

    assert readiness["passed"] is True
    assert readiness["failures"] == []
    assert readiness["audit"]["total_clean_tokens"] == 1_000
    assert readiness["audit"]["total_documents"] == 15
    assert readiness["audit"]["source_family_count"] == 2
    assert readiness["audit"]["max_single_source_token_share"] == 0.6


def test_lm_source_diagnostics_include_target_share_and_gap():
    builder = DatasetBuilder(DataConfig())
    diagnostics = builder._lm_source_diagnostics(
        [
            {
                "source": "dominant",
                "family": "general",
                "weight": 3.0,
                "target_share": 0.75,
                "raw_records": 10,
                "kept_documents": 8,
                "kept_tokens": 80,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
            {
                "source": "minor",
                "family": "domain",
                "weight": 1.0,
                "target_share": 0.25,
                "raw_records": 10,
                "kept_documents": 2,
                "kept_tokens": 20,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
        ],
        total_tokens=100,
        total_documents=10,
    )

    reports = {row["source"]: row for row in diagnostics["per_source"]}
    assert reports["dominant"]["target_share"] == 0.75
    assert reports["dominant"]["share_gap"] == -0.05
    assert reports["minor"]["target_share"] == 0.25
    assert reports["minor"]["share_gap"] == 0.05


def test_lm_source_diagnostics_include_realizability_adjustment_for_tiny_sources():
    builder = DatasetBuilder(DataConfig(lm_max_source_repeat_rate=0.1))
    diagnostics = builder._lm_source_diagnostics(
        [
            {
                "source": "dominant",
                "family": "general",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 100,
                "kept_documents": 90,
                "kept_tokens": 900,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
            {
                "source": "tiny",
                "family": "seed",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 3,
                "kept_documents": 3,
                "kept_tokens": 30,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
        ],
        total_tokens=930,
        total_documents=93,
    )

    reports = {row["source"]: row for row in diagnostics["per_source"]}
    assert reports["tiny"]["target_share"] == 0.5
    assert reports["tiny"]["effective_target_share"] < 0.1
    assert reports["tiny"]["realizable_token_capacity"] == 30
    assert reports["tiny"]["realizability_limited"] is True
    assert reports["tiny"]["effective_share_gap"] == 0.0
    assert reports["dominant"]["effective_target_share"] > reports["dominant"]["target_share"]


def test_lm_source_diagnostics_warn_when_pretrain_domain_contribution_is_too_low():
    builder = DatasetBuilder(DataConfig())
    diagnostics = builder._lm_source_diagnostics(
        [
            {
                "source": "local_mvp_pretrain_corpus",
                "family": "general_clean_prose",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 100,
                "kept_documents": 100,
                "kept_tokens": 999_000,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
            {
                "source": "catalog_expanded_corpus",
                "family": "catalog_grounding_prose",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 10,
                "kept_documents": 10,
                "kept_tokens": 1_000,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
        ],
        total_tokens=1_000_000,
        total_documents=110,
        stage="pretrain",
    )

    contribution = diagnostics["domain_contribution"]
    assert contribution["passed"] is False
    assert contribution["severity"] == "warning"
    assert contribution["domain_tokens"] == 1_000
    assert contribution["token_share"] == 0.001
    assert contribution["minimum_token_share"] == 0.01
    assert contribution["minimum_tokens"] == 500_000
    assert contribution["failures"] == ["domain_tokens_too_low", "domain_share_too_low"]


def test_lm_source_diagnostics_pass_when_pretrain_domain_contribution_is_material():
    builder = DatasetBuilder(DataConfig())
    diagnostics = builder._lm_source_diagnostics(
        [
            {
                "source": "local_mvp_pretrain_corpus",
                "family": "general_clean_prose",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 100,
                "kept_documents": 100,
                "kept_tokens": 900_000,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
            {
                "source": "catalog_expanded_corpus",
                "family": "catalog_grounding_prose",
                "weight": 1.0,
                "target_share": 0.5,
                "raw_records": 1_000,
                "kept_documents": 1_000,
                "kept_tokens": 600_000,
                "dropped_reasons": Counter(),
                "repeated_documents": 0,
                "restart_count": 0,
                "phrase_counter": Counter(),
            },
        ],
        total_tokens=1_500_000,
        total_documents=1_100,
        stage="pretrain",
    )

    contribution = diagnostics["domain_contribution"]
    assert contribution["passed"] is True
    assert contribution["failures"] == []
    assert contribution["domain_tokens"] == 600_000
    assert contribution["token_share"] == 0.4
    assert contribution["source_names"] == ["catalog_expanded_corpus"]


def test_prepared_pretrain_audit_includes_domain_contribution_guardrail(tmp_path: Path):
    manifest_path = tmp_path / "pretrain.json"
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "packed_lm",
                "stage": "pretrain",
                "diagnostics": {
                    "per_source": [
                        {
                            "source": "local_mvp_pretrain_corpus",
                            "family": "general_clean_prose",
                            "weight": 1.0,
                            "target_share": 0.5,
                            "raw_records": 100,
                            "kept_documents": 100,
                            "kept_tokens": 999_000,
                            "repeated_documents": 0,
                            "restart_count": 0,
                            "dropped_reasons": {},
                        },
                        {
                            "source": "catalog_expanded_corpus",
                            "family": "catalog_grounding_prose",
                            "weight": 1.0,
                            "target_share": 0.5,
                            "raw_records": 10,
                            "kept_documents": 10,
                            "kept_tokens": 1_000,
                            "repeated_documents": 0,
                            "restart_count": 0,
                            "dropped_reasons": {},
                        },
                    ],
                    "top_repeated_phrases": [],
                },
            }
        )
    )
    builder = DatasetBuilder(
        DataConfig(
            pretrain_sources=[
                DataSourceConfig(
                    name="prepared-pretrain",
                    path=str(manifest_path),
                    format="prepared",
                )
            ],
        )
    )

    audit = builder.audit_lm_stage("pretrain")

    contribution = audit["domain_contribution"]
    assert contribution["passed"] is False
    assert contribution["failures"] == ["domain_tokens_too_low", "domain_share_too_low"]
    assert contribution["domain_tokens"] == 1_000
    assert contribution["token_share"] == 0.001


def test_iter_tokenized_documents_for_source_drops_too_short_tokenized_rows(monkeypatch):
    builder = DatasetBuilder(DataConfig())
    source = DataSourceConfig(
        name="docs",
        format="text",
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
    )
    audit_state = builder._new_lm_audit_state(source)
    progress_state = {"raw_records_consumed": 0, "accepted_records": 0, "restart_count": 0}

    monkeypatch.setattr(
        builder,
        "_load_source_records",
        lambda *_args, **_kwargs: iter([{"text": "pathological row"}]),
    )

    class DummyTokenizer:
        def encode(self, _text, add_bos=True, add_eos=True):
            assert add_bos is True
            assert add_eos is True
            return [2]

    rows = list(
        builder._iter_tokenized_documents_for_source(
            source,
            tokenizer=DummyTokenizer(),  # type: ignore[arg-type]
            seen_hashes=set(),
            audit_state=audit_state,
            progress_state=progress_state,
        )
    )

    assert rows == []
    assert audit_state["dropped_reasons"]["too_short_tokenized"] == 1
    assert progress_state["accepted_records"] == 0


def test_prepare_packed_stage_drops_one_token_packed_windows(tmp_path: Path, monkeypatch):
    class DummyTokenizer:
        def token_to_id(self, token: str) -> int:
            return {"<pad>": 0, "</s>": 9}[token]

    source = DataSourceConfig(
        name="docs",
        format="text",
        path=str(tmp_path / "unused.txt"),
        quality_filter=False,
        deduplicate=False,
        pii_scrub=False,
    )
    builder = DatasetBuilder(
        DataConfig(
            tokenizer_path="dummy.model",
            sequence_length=4,
            prepared_shard_size=8,
            pretraining_token_budget=100,
            pretrain_sources=[source],
        )
    )

    monkeypatch.setattr(dataset_module, "SentencePieceTokenizer", lambda _path: DummyTokenizer())

    def fake_iter_tokenized_documents_for_source(_source, **kwargs):
        audit_state = kwargs["audit_state"]
        progress_state = kwargs["progress_state"]
        audit_state["raw_records"] += 1
        audit_state["kept_documents"] += 1
        audit_state["kept_tokens"] += 4
        progress_state["accepted_records"] += 1
        progress_state["raw_records_consumed"] += 1
        yield DocumentRecord(text="doc", source="docs", document_id="doc-1"), [1, 2, 3, 4]

    monkeypatch.setattr(
        builder,
        "_iter_tokenized_documents_for_source",
        fake_iter_tokenized_documents_for_source,
    )

    manifest = builder.prepare_stage(
        "pretrain",
        str(tmp_path / "prepared" / "pretrain.json"),
        force_rebuild=True,
    )

    np = __import__("numpy")
    rows = np.load(manifest["shards"][0]["path"])
    assert manifest["num_sequences"] == 1
    assert manifest["diagnostics"]["too_short_packed_sequences"] == 1
    assert all(int((row != 0).sum()) > 1 for row in rows)


def test_should_keep_sft_example_filters_generic_refusal_and_short_abstention():
    builder = DatasetBuilder(DataConfig())

    keep, reason = builder._should_keep_sft_example(
        bucket="hard_refusal",
        label_token_count=8,
        assistant_text="I can't say that.",
    )
    assert keep is False
    assert reason == "generic_refusal"

    keep, reason = builder._should_keep_sft_example(
        bucket="informative_abstention",
        label_token_count=12,
        assistant_text="I do not see that in the catalog.",
    )
    assert keep is False
    assert reason == "too_short_abstention"


def test_select_sft_candidate_indices_globally_trims_dominant_bucket():
    builder = DatasetBuilder(DataConfig())
    candidate_metadata_rows = []
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "constructive_direct",
            "label_token_count": 48,
            "source": "dominant",
            "prompt_signature_hash": f"constructive-{index}",
        }
        for index in range(220)
    )
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "clarifying_question",
            "label_token_count": 28,
            "source": "clarifying",
            "prompt_signature_hash": f"clarifying-{index}",
        }
        for index in range(48)
    )
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "informative_abstention",
            "label_token_count": 32,
            "source": "abstention",
            "prompt_signature_hash": f"abstention-{index}",
        }
        for index in range(8)
    )
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "hard_refusal",
            "label_token_count": 26,
            "source": "refusal",
            "prompt_signature_hash": f"refusal-{index}",
        }
        for index in range(4)
    )

    selected_indices, planner = builder._select_sft_candidate_indices(candidate_metadata_rows)

    candidate_total = len(candidate_metadata_rows)
    selected_total = len(selected_indices)
    candidate_constructive_share = 220 / candidate_total
    selected_constructive_share = planner["selected_bucket_counts"]["constructive_direct"] / selected_total

    assert planner["planned_total_examples"] < candidate_total
    assert selected_total == planner["planned_total_examples"]
    assert planner["distribution_reject_counts"]["constructive_direct"] > 0
    assert planner["selected_bucket_counts"]["clarifying_question"] == 48
    assert planner["selected_bucket_counts"]["informative_abstention"] == 8
    assert selected_constructive_share < candidate_constructive_share
    assert planner["bucket_targets"]["constructive_direct"]["distribution_rejects"] > 0


def test_select_sft_candidate_indices_prefers_latest_informative_abstentions():
    builder = DatasetBuilder(DataConfig())
    candidate_metadata_rows = []
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "constructive_direct",
            "label_token_count": 48,
            "source": "dominant",
            "prompt_signature_hash": f"constructive-{index}",
        }
        for index in range(220)
    )
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "clarifying_question",
            "label_token_count": 28,
            "source": "clarifying",
            "prompt_signature_hash": f"clarifying-{index}",
        }
        for index in range(48)
    )
    abstention_start = len(candidate_metadata_rows)
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "informative_abstention",
            "label_token_count": 32,
            "source": "abstention",
            "prompt_signature_hash": f"abstention-{index}",
        }
        for index in range(40)
    )
    candidate_metadata_rows.extend(
        {
            "behavior_bucket": "hard_refusal",
            "label_token_count": 26,
            "source": "refusal",
            "prompt_signature_hash": f"refusal-{index}",
        }
        for index in range(4)
    )

    selected_indices, planner = builder._select_sft_candidate_indices(candidate_metadata_rows)

    selected_abstention_indices = [
        index
        for index in selected_indices
        if candidate_metadata_rows[index]["behavior_bucket"] == "informative_abstention"
    ]

    abstention_keep_count = planner["selected_bucket_counts"]["informative_abstention"]

    assert abstention_keep_count < 40
    assert selected_abstention_indices == list(
        range(
            abstention_start + 40 - abstention_keep_count,
            abstention_start + 40,
        )
    )


def test_validate_preference_datasets_rejects_invalid_metadata():
    builder = DatasetBuilder(DataConfig())
    dataset = SimpleNamespace(
        examples=[
            PreferenceExample(
                prompt=[{"role": "user", "content": "Hi"}],
                chosen="hello",
                rejected="bad",
                source="test",
                metadata={
                    "chosen_quality_tier": "model_unreviewed",
                    "negative_type": "unspecified",
                },
            )
        ]
    )

    validation = builder.validate_preference_datasets(dataset)

    assert validation["valid_for_promotion"] is False
    assert "invalid_chosen_quality_tier" in validation["promotion_blockers"]
    assert "invalid_negative_type" in validation["promotion_blockers"]
