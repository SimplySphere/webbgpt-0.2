from __future__ import annotations

from config import ServeConfig
from generation import strip_stop_strings


class VLLMChatBackend:
    def __init__(self, config: ServeConfig):
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("vLLM is required for serving. Install with `pip install vllm`.") from exc
        self._sampling_cls = SamplingParams
        self.llm = LLM(
            model=config.checkpoint_path,
            tokenizer=config.tokenizer_path,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=config.trust_remote_code,
            max_model_len=config.max_model_len,
        )
        self.backend_name = "vllm"
        self.seed = config.seed

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
        del no_repeat_ngram_size
        effective_stop_strings = list(stop_strings or [])
        sampling = self._sampling_cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k is not None and top_k > 0 else -1,
            top_p=top_p,
            repetition_penalty=max(repetition_penalty, 1.0),
            stop=effective_stop_strings or None,
            seed=self.seed,
        )
        outputs = self.llm.generate(prompt, sampling)
        return strip_stop_strings(outputs[0].outputs[0].text, effective_stop_strings)
