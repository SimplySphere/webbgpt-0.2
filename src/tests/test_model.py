import logging

import pytest


torch = pytest.importorskip("torch")

from config import ModelConfig
from model.attention import GroupedQueryAttention
from model.cache import LayerKVCache
from model.modules import build_attention_mask
from model.transformer import CausalLMOutput, CausalTransformer


def _tiny_config(max_position_embeddings: int = 128) -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
    )


def test_forward_shapes():
    model = CausalTransformer(_tiny_config())
    input_ids = torch.randint(0, 128, (2, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 16, 128)
    assert outputs.loss is not None


def test_generation_cache_extends_sequence():
    model = CausalTransformer(_tiny_config())
    input_ids = torch.randint(0, 128, (1, 8))
    output = model.generate(input_ids=input_ids, max_new_tokens=4, temperature=0.0)
    assert output.shape[-1] >= input_ids.shape[-1]


def test_generate_prefills_full_prompt_then_decodes_newest_token():
    class RecordingModel(CausalTransformer):
        def __init__(self):
            super().__init__(_tiny_config())
            self.seen_input_ids = []

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **_kwargs):
            self.seen_input_ids.append(input_ids.detach().clone())
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[:, -1, 5] = 10.0
            cache = [object()] if use_cache else None
            return CausalLMOutput(logits=logits, past_key_values=cache)

    model = RecordingModel()
    input_ids = torch.tensor([[11, 12, 13, 14]])

    model.generate(input_ids=input_ids, max_new_tokens=3, temperature=0.0)

    assert model.seen_input_ids[0].tolist() == [[11, 12, 13, 14]]
    assert model.seen_input_ids[1].shape == (1, 1)
    assert model.seen_input_ids[2].shape == (1, 1)


def test_cached_forward_matches_full_context_logits_for_next_token():
    model = CausalTransformer(_tiny_config())
    model.eval()
    prompt_ids = torch.tensor([[4, 5, 6]])
    next_id = torch.tensor([[7]])
    full_ids = torch.cat([prompt_ids, next_id], dim=1)

    full_outputs = model(input_ids=full_ids)
    prefill = model(input_ids=prompt_ids, use_cache=True)
    cached_outputs = model(
        input_ids=next_id,
        attention_mask=torch.ones(1, full_ids.size(1), dtype=torch.long),
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )

    assert torch.allclose(full_outputs.logits[:, -1, :], cached_outputs.logits[:, -1, :], atol=1e-5)


def test_attention_concatenates_cached_keys_and_values():
    config = _tiny_config()
    attention = GroupedQueryAttention(config)
    hidden_states = torch.randn(1, 2, config.hidden_size)
    past_key = torch.randn(1, config.num_key_value_heads, 3, config.head_dim)
    past_value = torch.randn(1, config.num_key_value_heads, 3, config.head_dim)
    past = LayerKVCache(key=past_key.clone(), value=past_value.clone())

    _output, present = attention(
        hidden_states,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        past_key_value=past,
        use_cache=True,
    )

    assert present is not None
    assert present.key.size(-2) == 5
    assert present.value.size(-2) == 5
    assert torch.allclose(present.key[:, :, :3, :], past_key)
    assert torch.allclose(present.value[:, :, :3, :], past_value)


def test_generated_continuation_depends_on_prompt_context_cache():
    class ContextAwareModel(CausalTransformer):
        def __init__(self):
            super().__init__(_tiny_config())

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **_kwargs):
            if past_key_values is None:
                context_token = int(input_ids[0, 0].item())
            else:
                context_token = int(past_key_values[0])
            next_token = 21 if context_token == 9 else 22
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[:, -1, next_token] = 10.0
            return CausalLMOutput(logits=logits, past_key_values=[context_token] if use_cache else None)

    model = ContextAwareModel()

    first = model.generate(torch.tensor([[9, 1, 1]]), max_new_tokens=1, temperature=0.0)
    second = model.generate(torch.tensor([[10, 1, 1]]), max_new_tokens=1, temperature=0.0)

    assert int(first[0, -1].item()) == 21
    assert int(second[0, -1].item()) == 22


def test_generate_short_prompt_uses_full_context(caplog):
    class RecordingModel(CausalTransformer):
        def __init__(self):
            super().__init__(_tiny_config(max_position_embeddings=4))
            self.seen_input_ids = []
            self.seen_attention_lengths = []

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **_kwargs):
            self.seen_input_ids.append(input_ids.detach().clone())
            self.seen_attention_lengths.append(None if attention_mask is None else attention_mask.size(-1))
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[:, -1, 5] = 10.0
            key = torch.zeros(
                input_ids.size(0),
                self.config.num_key_value_heads,
                input_ids.size(1),
                self.config.head_dim,
            )
            cache = [LayerKVCache(key=key, value=key.clone())] if use_cache else None
            return CausalLMOutput(logits=logits, past_key_values=cache)

    model = RecordingModel()
    input_ids = torch.tensor([[11, 12, 13]])

    with caplog.at_level(logging.INFO, logger="model.transformer"):
        output = model.generate(input_ids=input_ids, max_new_tokens=1, temperature=0.0)

    assert model.seen_input_ids[0].tolist() == [[11, 12, 13]]
    assert model.seen_attention_lengths[0] == 3
    assert output.shape[-1] == 4
    assert "prompt_tokens=3" in caplog.text
    assert "effective_context_window=4" in caplog.text
    assert "truncation_occurred=False" in caplog.text


def test_generate_overlong_prompt_left_truncates_with_warning(caplog):
    class RecordingModel(CausalTransformer):
        def __init__(self):
            super().__init__(_tiny_config(max_position_embeddings=4))
            self.seen_input_ids = []

        def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **_kwargs):
            self.seen_input_ids.append(input_ids.detach().clone())
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[:, -1, 5] = 10.0
            key = torch.zeros(
                input_ids.size(0),
                self.config.num_key_value_heads,
                input_ids.size(1),
                self.config.head_dim,
            )
            cache = [LayerKVCache(key=key, value=key.clone())] if use_cache else None
            return CausalLMOutput(logits=logits, past_key_values=cache)

    model = RecordingModel()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with caplog.at_level(logging.INFO, logger="model.transformer"):
        output = model.generate(input_ids=input_ids, max_new_tokens=1, temperature=0.0)

    assert model.seen_input_ids[0].tolist() == [[3, 4, 5, 6]]
    assert output.tolist()[0][:6] == [1, 2, 3, 4, 5, 6]
    assert output.shape[-1] == 7
    assert "left-truncating prompt_tokens=6 effective_context_window=4" in caplog.text
    assert "truncation_occurred=True" in caplog.text


def test_generate_sliding_window_never_exceeds_max_position_embeddings(caplog):
    class CacheLengthRecordingModel(CausalTransformer):
        def __init__(self):
            super().__init__(_tiny_config(max_position_embeddings=4))
            self.attended_lengths = []
            self.past_lengths = []
            self.input_lengths = []
            self.attention_mask_lengths = []
            self.returned_cache_lengths = []
            self.position_ids = []

        def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            **_kwargs,
        ):
            past_length = 0
            if past_key_values:
                past_length = past_key_values[0].key.size(-2)
            total_length = past_length + input_ids.size(-1)
            self.past_lengths.append(past_length)
            self.input_lengths.append(input_ids.size(-1))
            self.attended_lengths.append(total_length)
            self.attention_mask_lengths.append(None if attention_mask is None else attention_mask.size(-1))
            self.position_ids.append(position_ids.detach().clone())
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.config.vocab_size)
            logits[:, -1, 5] = 10.0
            key = torch.zeros(
                input_ids.size(0),
                self.config.num_key_value_heads,
                total_length,
                self.config.head_dim,
            )
            self.returned_cache_lengths.append(total_length)
            cache = [LayerKVCache(key=key, value=key.clone())] if use_cache else None
            return CausalLMOutput(logits=logits, past_key_values=cache)

    model = CacheLengthRecordingModel()

    with caplog.at_level(logging.INFO, logger="model.transformer"):
        output = model.generate(torch.tensor([[11, 12, 13, 14]]), max_new_tokens=6, temperature=0.0)

    assert output.shape[-1] == 10
    assert max(model.attended_lengths) == 4
    assert max(length for length in model.attention_mask_lengths if length is not None) == 4
    assert model.input_lengths[0] == 4
    assert model.input_lengths[1:] == [1, 1, 1, 1, 1]
    assert model.past_lengths[1:] == [3, 3, 3, 3, 3]
    assert model.position_ids[0].tolist() == [[0, 1, 2, 3]]
    assert [int(position_ids[0, -1].item()) for position_ids in model.position_ids[1:]] == [4, 5, 6, 7, 8]
    assert max(model.returned_cache_lengths) == 4
    assert "truncation_occurred=True" in caplog.text


def test_build_attention_mask_shape():
    mask = torch.ones(2, 10, dtype=torch.long)
    additive = build_attention_mask(mask, query_length=4, key_length=10, device=mask.device, dtype=torch.float32)
    assert additive.shape == (2, 1, 4, 10)
