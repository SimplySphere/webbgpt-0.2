from __future__ import annotations

import logging
from dataclasses import dataclass

from config import ModelConfig
from generation import apply_no_repeat_ngram, apply_repetition_penalty
from model.attention import GroupedQueryAttention
from model.cache import KVCache, LayerKVCache
from model.modules import RMSNorm, SwiGLU


def _require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CausalLMOutput:
    logits: "torch.Tensor"
    loss: "torch.Tensor | None" = None
    past_key_values: KVCache | None = None
    hidden_states: list["torch.Tensor"] | None = None


class DecoderLayer(_require_torch()[1].Module):
    def __init__(self, config: ModelConfig):
        _, nn, _ = _require_torch()
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        attn_output, present = self.self_attn.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = residual + self.dropout(self.mlp.forward(hidden_states))
        return hidden_states, present


class CausalTransformer(_require_torch()[1].Module):
    def __init__(self, config: ModelConfig):
        torch, nn, _ = _require_torch()
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.embed_dropout = nn.Dropout(config.emb_dropout)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = config.gradient_checkpointing
        self._torch = torch
        self.apply(self._init_weights)

    def _init_weights(self, module):
        _, nn, _ = _require_torch()
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> CausalLMOutput:
        torch, _, F = _require_torch()
        batch_size, seq_len = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_length = 0
        else:
            past_length = past_key_values[0].key.size(-2) if past_key_values and past_key_values[0] else 0
        if position_ids is None:
            position_ids = (
                torch.arange(past_length, past_length + seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, past_length + seq_len, device=input_ids.device, dtype=torch.long
            )

        hidden_states = self.embed_dropout(self.embed_tokens(input_ids))
        all_hidden_states = [] if output_hidden_states else None
        next_cache: KVCache | None = [] if use_cache else None

        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            past = past_key_values[index]
            if self.training and self.gradient_checkpointing and not use_cache:
                from torch.utils.checkpoint import checkpoint

                def _layer_forward(hidden_states):
                    output, _ = layer.forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                    )
                    return output

                hidden_states = checkpoint(_layer_forward, hidden_states, use_reentrant=False)
                present = None
            else:
                hidden_states, present = layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past,
                    use_cache=use_cache,
                )
            if use_cache and present is not None and next_cache is not None:
                next_cache.append(present)

        hidden_states = self.norm.forward(hidden_states)
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )

    @property
    def device(self):
        return self.embed_tokens.weight.device

    def _sample_next_token(self, logits, temperature: float, top_k: int | None, top_p: float | None):
        torch, _, F = _require_torch()
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)
        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            mask = torch.zeros_like(sorted_mask).scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generation_context_window(self) -> int:
        context_window = int(getattr(self.config, "max_position_embeddings", 0) or 0)
        if context_window <= 0:
            raise ValueError("model.config.max_position_embeddings must be a positive integer for generation")
        return context_window

    @staticmethod
    def _cache_length(cache: KVCache | None) -> int:
        if not cache:
            return 0
        first_layer = cache[0]
        key = getattr(first_layer, "key", None)
        if key is None:
            return 0
        return int(key.size(-2))

    @staticmethod
    def _crop_cache(cache: KVCache | None, max_tokens: int) -> KVCache | None:
        if cache is None:
            return None
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative when cropping KV cache")
        cropped: KVCache = []
        changed = False
        for layer_cache in cache:
            key = getattr(layer_cache, "key", None)
            value = getattr(layer_cache, "value", None)
            if key is None or value is None:
                cropped.append(layer_cache)
                continue
            current_tokens = int(key.size(-2))
            if current_tokens <= max_tokens:
                cropped.append(layer_cache)
                continue
            if max_tokens == 0:
                key = key[:, :, :0, :]
                value = value[:, :, :0, :]
            else:
                key = key[:, :, -max_tokens:, :]
                value = value[:, :, -max_tokens:, :]
            cropped.append(LayerKVCache(key=key, value=value))
            changed = True
        return cropped if changed else cache

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_token_ids: list[int] | None = None,
        use_cache: bool = True,
    ):
        torch, _, _ = _require_torch()
        context_window = self._generation_context_window()
        prompt_tokens = int(input_ids.size(-1))
        if prompt_tokens <= 0:
            raise ValueError("input_ids must contain at least one prompt token for generation")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        prompt_truncated = prompt_tokens > context_window
        if prompt_truncated:
            logger.warning(
                "generation prompt exceeds context window; left-truncating prompt_tokens=%s "
                "effective_context_window=%s max_new_tokens=%s",
                prompt_tokens,
                context_window,
                max_new_tokens,
            )
            active_generated = input_ids[:, -context_window:]
            active_attention_mask = attention_mask[:, -context_window:]
        else:
            active_generated = input_ids
            active_attention_mask = attention_mask
        active_position_ids = (
            torch.arange(active_generated.size(-1), device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.size(0), -1)
        )

        logger.info(
            "generation_context prompt_tokens=%s max_new_tokens=%s effective_context_window=%s "
            "prompt_truncated=%s",
            prompt_tokens,
            max_new_tokens,
            context_window,
            prompt_truncated,
        )

        returned_generated = input_ids
        cache: KVCache | None = None
        effective_stop_ids = set(stop_token_ids or [self.config.eos_token_id])
        history_truncated = prompt_truncated
        generated_tokens = 0
        for _ in range(max_new_tokens):
            if cache is None:
                model_input = active_generated
                model_position_ids = active_position_ids
            else:
                model_input = active_generated[:, -1:]
                model_position_ids = active_position_ids[:, -1:]
                cache = self._crop_cache(cache, max(context_window - int(model_input.size(-1)), 0))
            cache_length = self._cache_length(cache)
            attention_tokens = cache_length + int(model_input.size(-1))
            effective_attention_mask = active_attention_mask[:, -attention_tokens:]
            output = self.forward(
                model_input,
                attention_mask=effective_attention_mask,
                position_ids=model_position_ids,
                past_key_values=cache,
                use_cache=use_cache,
            )
            cache = self._crop_cache(output.past_key_values, context_window)
            next_logits = output.logits[:, -1, :]
            penalty_context = active_generated[:, -context_window:]
            next_logits = apply_repetition_penalty(next_logits, penalty_context, repetition_penalty)
            next_logits = apply_no_repeat_ngram(next_logits, penalty_context, no_repeat_ngram_size)
            next_token = self._sample_next_token(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            returned_generated = torch.cat([returned_generated, next_token], dim=-1)
            next_position_id = active_position_ids[:, [-1]] + 1
            active_generated = torch.cat([active_generated, next_token], dim=-1)
            active_attention_mask = torch.cat(
                [active_attention_mask, torch.ones_like(next_token, dtype=active_attention_mask.dtype)], dim=-1
            )
            active_position_ids = torch.cat([active_position_ids, next_position_id], dim=-1)
            if active_generated.size(-1) > context_window:
                history_truncated = True
                active_generated = active_generated[:, -context_window:]
                active_attention_mask = active_attention_mask[:, -context_window:]
                active_position_ids = active_position_ids[:, -context_window:]
            generated_tokens += 1
            if torch.all(
                torch.tensor(
                    [int(token_id) in effective_stop_ids for token_id in next_token.view(-1).tolist()],
                    device=next_token.device,
                    dtype=torch.bool,
                )
            ):
                break
        logger.info(
            "generation_context_complete prompt_tokens=%s max_new_tokens=%s generated_tokens=%s "
            "effective_context_window=%s truncation_occurred=%s",
            prompt_tokens,
            max_new_tokens,
            generated_tokens,
            context_window,
            history_truncated,
        )
        return returned_generated
