from __future__ import annotations

from pathlib import Path

from config import ModelConfig, ServeConfig
from generation import resolve_stop_token_ids, strip_stop_strings
from model.transformer import CausalTransformer
from tokenizer import SentencePieceTokenizer
from torch_runtime import get_torch_device


class NativeCheckpointChatBackend:
    def __init__(self, config: ServeConfig):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "native checkpoint serving requires `torch` to be installed."
            ) from exc

        checkpoint_dir = Path(config.checkpoint_path)
        checkpoint_file = checkpoint_dir / "checkpoint.pt"
        if not checkpoint_file.exists():
            raise RuntimeError(f"native checkpoint backend requires {checkpoint_file}")

        self._torch = torch
        self.tokenizer = SentencePieceTokenizer(config.tokenizer_path)
        self.model_config = self._load_model_config(config, checkpoint_dir)
        self.model = CausalTransformer(self.model_config)
        payload = torch.load(checkpoint_file, map_location="cpu")
        self.model.load_state_dict(payload["model"], strict=True)
        self.device = get_torch_device()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_model_len = min(
            int(config.max_model_len),
            int(self.model_config.max_position_embeddings),
        )
        self.backend_name = "native"
        self.seed = config.seed

    def _load_model_config(self, config: ServeConfig, checkpoint_dir: Path) -> ModelConfig:
        import json

        candidates = []
        if config.model_config_path:
            candidates.append(Path(config.model_config_path))
        candidates.extend(
            [
                checkpoint_dir.parent / "configs" / "model.json",
                Path("sample-configs/model-local-mvp.json"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return ModelConfig.from_dict(json.loads(candidate.read_text()))
        searched = ", ".join(str(path) for path in candidates)
        raise RuntimeError(
            f"Could not find model config for native checkpoint. Searched: {searched}"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        stop_strings: list[str] | None = None,
    ) -> str:
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if not token_ids:
            token_ids = [self.tokenizer.bos_token_id]
        max_tokens = max(int(max_tokens), 1)
        prompt_budget = max(self.max_model_len - max_tokens, 1)
        token_ids = token_ids[-prompt_budget:]
        input_ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=self.device)
        attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)
        effective_stop_strings = list(stop_strings or [])
        stop_token_ids = resolve_stop_token_ids(self.tokenizer, effective_stop_strings)
        with self._torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=max(repetition_penalty, 1.0),
                no_repeat_ngram_size=max(no_repeat_ngram_size, 0),
                stop_token_ids=stop_token_ids,
            )
        generated = output[0, input_ids.shape[1] :].detach().cpu().tolist()
        text = self.tokenizer.decode(int(token_id) for token_id in generated)
        return strip_stop_strings(text, effective_stop_strings)
