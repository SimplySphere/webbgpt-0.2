import json
from pathlib import Path

from data.prepared import (
    PreparedPackedDataset,
    PreparedPreferenceDataset,
    PreparedSFTDataset,
    save_buffer_rows,
    save_metadata_rows,
    save_prepared_manifest,
)


def test_prepared_packed_dataset_exposes_loss_probe_provenance(tmp_path: Path):
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / "shard-00000.npy"
    metadata_path = shard_dir / "metadata-00000.jsonl"
    save_buffer_rows(shard_path, [[1, 2, 3, 0]])
    save_metadata_rows(
        metadata_path,
        [
            {
                "source_names": ["catalog_domain_fixture"],
                "contributors": [
                    {
                        "source": "catalog_domain_fixture",
                        "family": "catalog_grounding_prose",
                        "document_id": "catalog-1",
                    }
                ],
                "packed_document_count": 1,
            }
        ],
    )
    manifest_path = tmp_path / "packed.json"
    save_prepared_manifest(
        manifest_path,
        {
            "version": "1.0",
            "stage": "pretrain",
            "kind": "packed_lm",
            "input_fingerprint": "fingerprint",
            "tokenizer_path": "artifacts/tokenizer/webbgpt.model",
            "sequence_length": 4,
            "pad_token_id": 0,
            "num_sequences": 1,
            "num_tokens": 3,
            "source_snapshots": [],
            "shards": [
                {
                    "path": str(shard_path),
                    "metadata_path": str(metadata_path),
                    "rows": 1,
                }
            ],
        },
    )

    item = PreparedPackedDataset(manifest_path)[0]
    provenance = json.loads(item["provenance_json"])

    assert provenance["source_names"] == ["catalog_domain_fixture"]
    assert provenance["contributors"][0]["document_id"] == "catalog-1"
    assert provenance["packed_document_count"] == 1
    assert provenance["approximate_token_count"] == 3


def test_prepared_sft_manifest_v2_exposes_examples_and_prompt_hashes(tmp_path: Path):
    shard_dir = tmp_path / "sft"
    shard_dir.mkdir(parents=True, exist_ok=True)
    input_path = shard_dir / "input_ids-00000.npy"
    label_path = shard_dir / "labels-00000.npy"
    metadata_path = shard_dir / "metadata-00000.jsonl"
    save_buffer_rows(input_path, [[1, 2, 3, 0]])
    save_buffer_rows(label_path, [[-100, 2, 3, -100]])
    metadata_path.write_text(
        json.dumps(
            {
                "example_id": "ex-1",
                "split_group_id": "grp-1",
                "source": "curated",
                "prompt_signature_hash": "abc123",
                "behavior_bucket": "constructive_direct_answer",
                "quality_tier": "human_curated",
                "label_token_count": 2,
            }
        )
        + "\n"
    )
    manifest_path = tmp_path / "sft.json"
    save_prepared_manifest(
        manifest_path,
        {
            "version": "2.0",
            "stage": "sft",
            "kind": "sft",
            "input_fingerprint": "fingerprint",
            "tokenizer_path": "artifacts/tokenizer/webbgpt.model",
            "sequence_length": 4,
            "pad_token_id": 0,
            "num_examples": 1,
            "num_label_tokens": 2,
            "source_snapshots": [],
            "diagnostics": {},
            "trust": {"artifact_status": "promotable", "promotion_blockers": []},
            "shards": [
                {
                    "input_ids_path": str(input_path),
                    "labels_path": str(label_path),
                    "metadata_path": str(metadata_path),
                    "rows": 1,
                }
            ],
        },
    )

    dataset = PreparedSFTDataset(manifest_path)

    assert dataset.examples is not None
    assert dataset.trust_flags == []
    assert dataset.examples[0].metadata["prompt_signature_hash"] == "abc123"


def test_prepared_preference_manifest_v1_stays_untrusted(tmp_path: Path):
    shard_dir = tmp_path / "preference"
    shard_dir.mkdir(parents=True, exist_ok=True)
    chosen_path = shard_dir / "chosen_input_ids-00000.npy"
    rejected_path = shard_dir / "rejected_input_ids-00000.npy"
    save_buffer_rows(chosen_path, [[1, 2, 3, 0]])
    save_buffer_rows(rejected_path, [[1, 4, 3, 0]])
    manifest_path = tmp_path / "preference.json"
    save_prepared_manifest(
        manifest_path,
        {
            "version": "1.0",
            "stage": "preference",
            "kind": "preference",
            "input_fingerprint": "fingerprint",
            "tokenizer_path": "artifacts/tokenizer/webbgpt.model",
            "sequence_length": 4,
            "pad_token_id": 0,
            "num_examples": 1,
            "source_snapshots": [],
            "shards": [
                {
                    "chosen_input_ids_path": str(chosen_path),
                    "rejected_input_ids_path": str(rejected_path),
                    "rows": 1,
                }
            ],
        },
    )

    dataset = PreparedPreferenceDataset(manifest_path)

    assert dataset.examples is None
    assert "behavior_eval_untrusted" in dataset.trust_flags
    assert "overlap_guard_skipped" in dataset.trust_flags
