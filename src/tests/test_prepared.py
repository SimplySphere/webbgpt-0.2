from pathlib import Path

import pytest

from config import TokenizerConfig
from data.prepared import (
    encode_preference_example,
    encode_sft_messages,
    save_buffer_rows,
    save_prepared_manifest,
    validate_prepared_manifest_artifacts,
)
from tokenizer.spm import SentencePieceTokenizer, train_tokenizer


class FakeTokenizer:
    def __init__(self):
        self._special_ids = {
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "<|assistant|>": 4,
            "<|user|>": 5,
            "<|system|>": 6,
            "<|tool|>": 7,
        }
        self._next_id = 100
        self._vocab: dict[str, int] = {}

    def token_to_id(self, token: str) -> int:
        return self._special_ids[token]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens: list[int] = [self._special_ids["<s>"]] if add_bos else []
        index = 0
        specials = sorted(self._special_ids.keys(), key=len, reverse=True)
        while index < len(text):
            matched = False
            for special in specials:
                if text.startswith(special, index):
                    tokens.append(self._special_ids[special])
                    index += len(special)
                    matched = True
                    break
            if matched:
                continue
            char = text[index]
            if char.isspace():
                tokens.append(9)
                index += 1
                continue
            token = []
            while index < len(text) and not text[index].isspace():
                if any(text.startswith(special, index) for special in specials):
                    break
                token.append(text[index])
                index += 1
            piece = "".join(token)
            if piece not in self._vocab:
                self._vocab[piece] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[piece])
        if add_eos:
            tokens.append(self._special_ids["</s>"])
        return tokens


def test_encode_sft_messages_masks_non_assistant_tokens():
    tokenizer = FakeTokenizer()
    input_ids, labels = encode_sft_messages(
        [
            {"role": "system", "content": "You are WebbGPT."},
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning, Harry."},
        ],
        tokenizer,
        sequence_length=64,
    )
    first_labeled_index = next(index for index, value in enumerate(labels) if value != -100)
    assistant_prefix = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
    assistant_token_positions = [index for index, value in enumerate(input_ids) if value == tokenizer.token_to_id("<|assistant|>")]
    assert assistant_token_positions
    assert first_labeled_index >= assistant_token_positions[0] + len(assistant_prefix)
    assert any(value != -100 for value in labels[first_labeled_index:])
    assert all(value == -100 for value in labels[:assistant_token_positions[0] + len(assistant_prefix)])


def _train_real_tokenizer(tmp_path: Path) -> SentencePieceTokenizer:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "WebbGPT explains courses and planning clearly.",
                "Students benefit from grounded answers and honest uncertainty.",
                "Preference tuning should reward helpful chosen responses.",
                "Good assistant messages should end cleanly.",
            ]
        )
        + "\n"
    )
    model_path = train_tokenizer(
        [str(corpus_path)],
        TokenizerConfig(
            model_prefix=str(tmp_path / "test-tokenizer"),
            vocab_size=320,
            sample_input_sentence_size=1000,
            max_sentence_length=2048,
        ),
    )
    return SentencePieceTokenizer(model_path)


def test_real_sentencepiece_tokenizer_encodes_literal_eos_as_special_id(tmp_path: Path):
    tokenizer = _train_real_tokenizer(tmp_path)

    token_ids = tokenizer.encode("<|assistant|>\nhello\n</s>", add_bos=True, add_eos=False)

    assert tokenizer.token_to_id("<|assistant|>") in token_ids
    assert tokenizer.token_to_id("</s>") in token_ids


def test_encode_preference_example_appends_real_eos_token(tmp_path: Path):
    tokenizer = _train_real_tokenizer(tmp_path)

    token_ids = encode_preference_example(
        [{"role": "user", "content": "Say hi"}],
        "Hi there.",
        tokenizer,
        sequence_length=64,
    )

    assert tokenizer.token_to_id("</s>") in token_ids


def test_validate_prepared_manifest_artifacts_accepts_valid_packed_manifest(tmp_path: Path):
    shard_dir = tmp_path / "pretrain"
    shard_dir.mkdir()
    shard_path = shard_dir / "shard-00000.npy"
    save_buffer_rows(shard_path, [[1, 2, 3, 0]])
    manifest_path = tmp_path / "pretrain.json"
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
            "shards": [{"path": str(shard_path), "rows": 1}],
        },
    )

    manifest = validate_prepared_manifest_artifacts(manifest_path, expected_kind="packed_lm")

    assert manifest["kind"] == "packed_lm"


def test_validate_prepared_manifest_artifacts_fails_for_missing_shard(tmp_path: Path):
    manifest_path = tmp_path / "pretrain.json"
    missing_shard = tmp_path / "pretrain" / "shard-00000.npy"
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
            "shards": [{"path": str(missing_shard), "rows": 1}],
        },
    )

    with pytest.raises(RuntimeError, match="references missing shard artifact"):
        validate_prepared_manifest_artifacts(manifest_path, expected_kind="packed_lm")
