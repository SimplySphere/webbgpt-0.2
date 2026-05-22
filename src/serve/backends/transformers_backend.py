from __future__ import annotations

from copy import deepcopy

from config import ServeConfig
from generation import strip_stop_strings
from torch_runtime import get_torch_device


class TransformersChatBackend:
    def __init__(self, config: ServeConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers serving fallback requires `transformers` and `torch` to be installed."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, trust_remote_code=config.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            trust_remote_code=config.trust_remote_code,
        )
        self.device = get_torch_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and not getattr(generation_config, "do_sample", False):
            # Avoid noisy warnings from sampling-only fields when we are doing greedy decode.
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = None
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = None
            if hasattr(generation_config, "top_k"):
                generation_config.top_k = None
        self.max_model_len = self._resolve_max_model_len(config.max_model_len)
        self.backend_name = "transformers"
        self.seed = config.seed

    def _resolve_max_model_len(self, configured_max_model_len: int) -> int:
        candidates = [configured_max_model_len]
        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max_len, int) and 0 < tokenizer_max_len < 1_000_000:
            candidates.append(tokenizer_max_len)
        model_config = getattr(self.model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
            value = getattr(model_config, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return min(candidates)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int | None = None,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        stop_strings: list[str] | None = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        do_sample = temperature > 0
        effective_stop_strings = list(stop_strings or [])
        stop_token_ids: list[int] = []
        for stop in effective_stop_strings:
            token_id = self.tokenizer.convert_tokens_to_ids(stop)
            if token_id is None or token_id == self.tokenizer.unk_token_id or token_id < 0:
                continue
            stop_token_ids.append(int(token_id))
        prompt_token_count = int(inputs["input_ids"].shape[1])
        max_model_len = getattr(self, "max_model_len", None)
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            max_model_len = self._resolve_max_model_len(prompt_token_count + max_tokens)
        prompt_budget = max(max_model_len - max_tokens, 1)
        if prompt_token_count > prompt_budget:
            for key in list(inputs.keys()):
                inputs[key] = inputs[key][:, -prompt_budget:]
        inputs = inputs.to(self.device)
        prompt_token_count = int(inputs["input_ids"].shape[1])
        effective_max_tokens = min(max_tokens, max(max_model_len - prompt_token_count, 1))
        eos_token_id = stop_token_ids or self.tokenizer.eos_token_id
        generation_config = deepcopy(getattr(self.model, "generation_config", None))
        if generation_config is not None:
            generation_config.do_sample = do_sample
            if do_sample:
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = max(temperature, 1e-5)
                if top_k is not None and hasattr(generation_config, "top_k"):
                    generation_config.top_k = max(int(top_k), 0)
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = top_p
            else:
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = None
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = None
                if hasattr(generation_config, "top_k"):
                    generation_config.top_k = None
        generate_kwargs = dict(
            max_new_tokens=effective_max_tokens,
            do_sample=do_sample,
            repetition_penalty=max(repetition_penalty, 1.0),
            no_repeat_ngram_size=max(no_repeat_ngram_size, 0),
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            tokenizer=self.tokenizer,
        )
        if generation_config is not None:
            generate_kwargs["generation_config"] = generation_config
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            if top_k is not None:
                generate_kwargs["top_k"] = max(int(top_k), 0)
            generate_kwargs["top_p"] = top_p
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return strip_stop_strings(text, effective_stop_strings)
