from __future__ import annotations

from collections.abc import Iterable


MIN_PACKED_LM_NONPAD_TOKENS = 3


class PackedSequencePacker:
    def __init__(
        self,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        current: list[int] | None = None,
        current_metadata: list[dict[str, object]] | None = None,
        dropped_short_windows: int = 0,
    ):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.current = list(current or [])
        self.current_metadata = [dict(item) for item in (current_metadata or [])]
        self.dropped_short_windows = int(dropped_short_windows)

    def state_dict(self) -> dict[str, object]:
        return {
            "current": list(self.current),
            "current_metadata": [dict(item) for item in self.current_metadata],
            "dropped_short_windows": self.dropped_short_windows,
        }

    def load_state_dict(self, payload: dict[str, object] | None) -> None:
        self.current = list((payload or {}).get("current", []))
        self.current_metadata = [dict(item) for item in (payload or {}).get("current_metadata", [])]
        self.dropped_short_windows = int((payload or {}).get("dropped_short_windows", 0))

    def _padded_window(self, window: list[int]) -> list[int]:
        if len(window) < self.sequence_length:
            return window + [self.pad_token_id] * (self.sequence_length - len(window))
        return window

    def _finalize_window(self, window: list[int]) -> list[int] | None:
        finalized = self._padded_window(window)
        # Rows with only one content token plus EOS produce a single trivial loss target.
        if sum(token != self.pad_token_id for token in finalized) < MIN_PACKED_LM_NONPAD_TOKENS:
            self.dropped_short_windows += 1
            return None
        return finalized

    def _finalize_metadata(self, metadata: list[dict[str, object]]) -> dict[str, object]:
        contributors = [dict(item) for item in metadata]
        source_names = sorted(
            {
                str(item.get("source", ""))
                for item in contributors
                if str(item.get("source", ""))
            }
        )
        return {
            "source_names": source_names,
            "contributors": contributors,
            "packed_document_count": len(contributors),
        }

    def push(self, tokens: list[int]):
        chunk = list(tokens)
        if not chunk:
            return
        if chunk[-1] != self.eos_token_id:
            chunk.append(self.eos_token_id)
        if len(chunk) > self.sequence_length:
            for start in range(0, len(chunk), self.sequence_length):
                window = chunk[start : start + self.sequence_length]
                finalized = self._finalize_window(window)
                if finalized is not None:
                    yield finalized
            return
        if len(self.current) + len(chunk) > self.sequence_length:
            current = self._finalize_window(self.current)
            self.current = []
            if current is not None:
                yield current
        self.current.extend(chunk)

    def finish(self):
        if self.current:
            current = self._finalize_window(self.current)
            self.current = []
            self.current_metadata = []
            if current is not None:
                yield current

    def push_with_metadata(self, tokens: list[int], metadata: dict[str, object]):
        chunk = list(tokens)
        if not chunk:
            return
        contributor = dict(metadata)
        if chunk[-1] != self.eos_token_id:
            chunk.append(self.eos_token_id)
        if len(chunk) > self.sequence_length:
            for start in range(0, len(chunk), self.sequence_length):
                window = chunk[start : start + self.sequence_length]
                finalized = self._finalize_window(window)
                if finalized is not None:
                    yield finalized, self._finalize_metadata([contributor])
            return
        if len(self.current) + len(chunk) > self.sequence_length:
            current = self._finalize_window(self.current)
            current_metadata = self._finalize_metadata(self.current_metadata)
            self.current = []
            self.current_metadata = []
            if current is not None:
                yield current, current_metadata
        self.current.extend(chunk)
        self.current_metadata.append(contributor)

    def finish_with_metadata(self):
        if self.current:
            current = self._finalize_window(self.current)
            current_metadata = self._finalize_metadata(self.current_metadata)
            self.current = []
            self.current_metadata = []
            if current is not None:
                yield current, current_metadata


def iter_packed_token_sequences(
    token_sequences: Iterable[list[int]],
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
):
    packer = PackedSequencePacker(
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    for tokens in token_sequences:
        yield from packer.push(tokens)
    yield from packer.finish()


def pack_token_sequences(
    token_sequences: Iterable[list[int]],
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> list[list[int]]:
    return list(
        iter_packed_token_sequences(
            token_sequences,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    )
