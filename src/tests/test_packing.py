from data.packing import PackedSequencePacker, pack_token_sequences


def test_pack_token_sequences_pads_to_length():
    sequences = [[1, 2, 3], [4, 5]]
    packed = pack_token_sequences(sequences, sequence_length=6, pad_token_id=0, eos_token_id=9)
    assert all(len(row) == 6 for row in packed)
    assert packed[0][-1] == 0


def test_pack_token_sequences_splits_long_documents():
    sequences = [[1, 2, 3, 4, 5, 6, 7]]
    packed = pack_token_sequences(sequences, sequence_length=4, pad_token_id=0, eos_token_id=9)
    assert len(packed) == 2


def test_pack_token_sequences_drops_one_token_tail_window():
    sequences = [[1, 2, 3, 4]]
    packed = pack_token_sequences(sequences, sequence_length=4, pad_token_id=0, eos_token_id=9)

    assert packed == [[1, 2, 3, 4]]
    assert all(sum(token != 0 for token in row) >= 3 for row in packed)


def test_pack_token_sequences_drops_content_token_plus_eos_window():
    sequences = [[1]]
    packed = pack_token_sequences(sequences, sequence_length=4, pad_token_id=0, eos_token_id=9)

    assert packed == []


def test_packer_tracks_dropped_short_windows_across_state_restore():
    packer = PackedSequencePacker(sequence_length=4, pad_token_id=0, eos_token_id=9)

    assert list(packer.push([1, 2, 3, 4])) == [[1, 2, 3, 4]]
    assert packer.dropped_short_windows == 1

    restored = PackedSequencePacker(sequence_length=4, pad_token_id=0, eos_token_id=9)
    restored.load_state_dict(packer.state_dict())

    assert restored.dropped_short_windows == 1


def test_finish_with_metadata_drops_nearly_empty_current_window():
    packer = PackedSequencePacker(sequence_length=4, pad_token_id=0, eos_token_id=9)

    assert list(packer.push_with_metadata([42], {"source": "tiny"})) == []
    assert list(packer.finish_with_metadata()) == []
    assert packer.dropped_short_windows == 1
