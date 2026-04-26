from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints


T = TypeVar("T")


def _default_stop_strings() -> list[str]:
    return [
        "</s>",
        "<|assistant|>",
        "<|user|>",
        "<|system|>",
        "<|tool|>",
        "<|citation|>",
    ]


def _default_special_tokens() -> dict[str, str]:
    return {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "assistant_token": "<|assistant|>",
        "user_token": "<|user|>",
        "system_token": "<|system|>",
        "tool_token": "<|tool|>",
        "citation_token": "<|citation|>",
    }


def _default_webb_freshness_policy() -> dict[str, Any]:
    return {
        "athletics": {"active_season": "6h", "off_season": "24h"},
        "faculty": {"cadence": "24h"},
        "admissions_general": {"cadence": "24h"},
        "student_life": {"cadence": "24h"},
        "college_guidance": {"cadence": "24h"},
        "course_catalog": {"cadence": "168h"},
        "handbook_policy": {"cadence": "168h"},
        "academic_publications": {"cadence": "168h"},
        "mission_values": {"cadence": "168h"},
        "museum_programs": {"cadence": "168h"},
    }


def _default_live_handbook_url() -> str:
    return "https://webb.myschoolapp.com/ftpimages/823/download/download_10529422.pdf?_=1774412901890"


def _coerce_field(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if is_dataclass(field_type):
        return _from_dict(field_type, value)
    if origin is list and value is not None:
        inner = get_args(field_type)[0]
        return [_coerce_field(inner, item) for item in value]
    if origin is dict and value is not None:
        key_type, inner = get_args(field_type)
        return {
            _coerce_field(key_type, item_key): _coerce_field(inner, item_value)
            for item_key, item_value in value.items()
        }
    if origin is Literal:
        if value not in get_args(field_type):
            raise ValueError(f"Expected one of {get_args(field_type)} but received {value!r}")
        return value
    if origin is not None and type(None) in get_args(field_type):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        inner = args[0] if args else Any
        return None if value is None else _coerce_field(inner, value)
    if field_type is Path and value is not None:
        return Path(value)
    return value


def _from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    resolved_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if field_info.name not in payload:
            continue
        field_type = resolved_hints.get(field_info.name, field_info.type)
        kwargs[field_info.name] = _coerce_field(field_type, payload[field_info.name])
    return cls(**kwargs)


@dataclass(slots=True)
class TokenizerConfig:
    version: str = "1.0"
    model_prefix: str = "artifacts/tokenizer/webbgpt"
    vocab_size: int = 50_176
    model_type: Literal["bpe", "unigram"] = "bpe"
    character_coverage: float = 0.9995
    byte_fallback: bool = True
    normalization_rule_name: str = "nmt_nfkc"
    sample_input_sentence_size: int = 10_000_000
    max_sentence_length: int = 16_384
    train_extremely_large_corpus: bool = True
    special_tokens: dict[str, str] = field(default_factory=_default_special_tokens)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenizerConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenizerCorpusConfig:
    version: str = "1.0"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config_name: str = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    output_path: str = "data/raw/tokenizer_corpus.txt"
    streaming: bool = True
    max_documents: int = 2_000_000
    max_characters: int = 500_000_000
    min_document_chars: int = 128
    normalize_whitespace: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenizerCorpusConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    version: str = "1.0"
    name: str = "webbgpt-3b"
    vocab_size: int = 50_176
    hidden_size: int = 3_072
    intermediate_size: int = 8_192
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8_192
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    resid_dropout: float = 0.0
    emb_dropout: float = 0.0
    initializer_range: float = 0.02
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataSourceConfig:
    name: str
    path: str = ""
    paths: list[str] = field(default_factory=list)
    split: str = "train"
    format: Literal["jsonl", "parquet", "arrow", "hf", "text", "prepared"] = "jsonl"
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    streaming: bool | None = None
    weight: float = 1.0
    text_field: str = "text"
    messages_field: str = "messages"
    prompt_field: str = "prompt"
    response_field: str = "response"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    id_field: str | None = None
    group_field: str | None = None
    family: str | None = None
    metadata_fields: list[str] = field(default_factory=list)
    language: str | None = None
    quality_filter: bool = True
    quality_filter_mode: Literal["basic", "broad_lm", "domain_lm"] = "basic"
    deduplicate: bool = True
    pii_scrub: bool = True
    max_records: int | None = None
    skip_records: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataSourceConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataConfig:
    version: str = "1.0"
    tokenizer_path: str = "artifacts/tokenizer/webbgpt.model"
    sequence_length: int = 8_192
    prepared_shard_size: int = 2_048
    seed: int = 52
    pretrain_sources: list[DataSourceConfig] = field(default_factory=list)
    continued_pretrain_sources: list[DataSourceConfig] = field(default_factory=list)
    sft_sources: list[DataSourceConfig] = field(default_factory=list)
    sft_validation_sources: list[DataSourceConfig] = field(default_factory=list)
    preference_sources: list[DataSourceConfig] = field(default_factory=list)
    preference_validation_sources: list[DataSourceConfig] = field(default_factory=list)
    validation_sources: list[DataSourceConfig] = field(default_factory=list)
    pretraining_token_budget: int = 120_000_000_000
    continued_pretraining_token_budget: int = 15_000_000_000
    min_document_chars: int = 128
    max_document_chars: int = 200_000
    lm_weighted_source_token_budget: int = 32_768
    lm_max_source_token_share: float = 0.65
    lm_max_source_repeat_rate: float = 0.1
    pretrain_domain_realization_gate_mode: Literal["warn", "fail"] = "warn"
    continue_readiness_min_clean_token_fraction: float = 0.8
    continue_readiness_min_documents: int = 1_500
    continue_readiness_min_source_families: int = 3
    continue_readiness_max_single_source_share: float = 0.65
    continue_readiness_max_repeat_rate: float = 0.1
    allow_unsafe_code: bool = False
    domain_tags: list[str] = field(
        default_factory=lambda: ["philosophy", "education", "advising", "course_catalog"]
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CheckpointConfig:
    output_dir: str = "artifacts/checkpoints"
    save_every_steps: int = 1_000
    keep_last_n: int = 5
    async_write: bool = False
    initialize_from: str | None = None
    resume_from: str | None = None
    export_every_eval: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainConfig:
    version: str = "1.0"
    run_name: str = "webbgpt-pretrain"
    seed: int = 52
    global_batch_size: int = 512
    micro_batch_size: int = 2
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2_000
    max_steps: int = 400_000
    continued_learning_rate: float | None = None
    continued_min_learning_rate: float | None = None
    continued_warmup_steps: int | None = None
    continued_max_steps: int | None = None
    sft_learning_rate: float | None = None
    sft_min_learning_rate: float | None = None
    sft_warmup_steps: int | None = None
    sft_max_steps: int | None = None
    sft_max_epochs: int | None = 5
    sft_evals_per_epoch: int = 4
    sft_min_eval_interval_steps: int = 25
    sft_early_stopping_patience_evals: int = 2
    sft_best_min_delta: float = 0.02
    sft_sample_every_steps: int = 100
    dpo_learning_rate: float | None = None
    dpo_min_learning_rate: float | None = None
    dpo_warmup_steps: int | None = None
    dpo_max_steps: int | None = None
    dpo_max_epochs: int | None = 2
    dpo_evals_per_epoch: int = 4
    dpo_early_stopping_patience_evals: int = 2
    dpo_best_min_delta: float = 0.005
    dpo_enable_lm_health_eval: bool = False
    sft_validation_fraction: float = 0.1
    sft_validation_min_examples: int = 16
    require_explicit_sft_validation: bool = False
    dpo_validation_fraction: float = 0.1
    dpo_validation_min_examples: int = 16
    dpo_min_train_examples: int = 0
    dpo_min_validation_examples: int = 0
    require_explicit_dpo_validation: bool = False
    allow_weak_posttrain_validation: bool = False
    posttrain_top_k_checkpoints: int = 3
    token_budget: int | None = None
    pretrain_progress_mode: Literal["prepared_tokens", "steps", "token_budget"] = "prepared_tokens"
    pretrain_stop_mode: Literal[
        "one_prepared_pass",
        "token_budget_repeat_allowed",
        "max_steps_limited",
    ] = "one_prepared_pass"
    pretrain_flush_final_partial_accumulation: bool = True
    final_eval_full_validation: bool = False
    final_num_eval_batches: int | None = None
    raw_lm_short_probe_max_new_tokens: int = 48
    raw_lm_long_probe_max_new_tokens: int = 128
    raw_lm_stable_temperature: float = 0.4
    raw_lm_stable_top_p: float = 0.9
    raw_lm_stress_temperature: float = 0.7
    raw_lm_stress_top_p: float = 0.95
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    eval_every_steps: int = 500
    num_eval_batches: int = 50
    compile_model: bool = True
    use_bf16: bool = True
    gradient_accumulation_steps: int = 1
    fsdp_sharding_strategy: Literal["full_shard", "shard_grad_op"] = "full_shard"
    activation_checkpointing: bool = True
    report_to: list[Literal["stdout", "tensorboard", "wandb"]] = field(
        default_factory=lambda: ["stdout"]
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReleaseGateConfig:
    assistant_pass_rate_min: float = 0.9
    chat_sanity_pass_rate_min: float = 0.9
    catalog_exactness_min: float = 1.0
    catalog_citation_rate_min: float = 1.0
    catalog_missing_abstention_min: float = 1.0
    webb_course_present_exactness_min: float = 0.9
    webb_course_present_citation_rate_min: float = 0.9
    webb_course_missing_abstention_min: float = 1.0
    webb_handbook_present_exactness_min: float = 0.9
    webb_handbook_present_citation_rate_min: float = 0.9
    webb_handbook_missing_abstention_min: float = 1.0
    webb_faculty_exactness_min: float = 0.9
    webb_admissions_exactness_min: float = 0.9
    webb_student_life_exactness_min: float = 0.85
    webb_student_life_citation_rate_min: float = 0.9
    webb_mission_values_exactness_min: float = 0.85
    webb_mission_values_citation_rate_min: float = 0.9
    webb_college_guidance_exactness_min: float = 0.85
    webb_college_guidance_citation_rate_min: float = 0.9
    webb_museum_programs_exactness_min: float = 0.85
    webb_museum_programs_citation_rate_min: float = 0.9
    webb_athletics_present_exactness_min: float = 0.9
    webb_athletics_present_citation_rate_min: float = 0.9
    webb_athletics_missing_abstention_min: float = 1.0
    webb_route_false_negative_rate_max: float = 0.05
    webb_require_citable_handbook: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleaseGateConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroundingConfig:
    version: str = "1.0"
    dsn: str = "sqlite:///artifacts/grounding/webbgpt.db"
    snapshot_id: str = "latest"
    seed_url_pack: str = "data/webb/seed_urls.json"
    offline_seed_url_pack: str | None = None
    source_policy_path: str = "data/webb/source_policies.json"
    handbook_url: str | None = field(default_factory=_default_live_handbook_url)
    sync_on_start: bool = False
    allow_ocr_fallback: bool = False
    sync_families: list[str] = field(default_factory=list)
    route_fanout_limit: int = 2
    planner_beta_enabled: bool = False
    freshness_policy: dict[str, Any] = field(default_factory=_default_webb_freshness_policy)
    legacy_catalog_input_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundingConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalConfig:
    version: str = "1.0"
    run_name: str = "webbgpt-eval"
    seed: int = 52
    decode_preset: str = "eval"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 4
    stop_strings: list[str] = field(default_factory=_default_stop_strings)
    batch_size: int = 8
    benchmark_paths: list[str] = field(default_factory=list)
    compare_to_checkpoint: str | None = None
    require_citations: bool = True
    grounding: GroundingConfig | None = None
    catalog_dsn: str = "sqlite:///artifacts/catalog/webbgpt-eval.db"
    catalog_input_path: str = "data/catalog/webb_catalog.json"
    enforce_release_gates: bool = False
    release_gates: ReleaseGateConfig = field(default_factory=ReleaseGateConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ServeConfig:
    version: str = "1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "webbgpt-3b-instruct"
    seed: int = 52
    checkpoint_path: str = "artifacts/export/webbgpt-3b-instruct"
    tokenizer_path: str = "artifacts/tokenizer/webbgpt.model"
    max_model_len: int = 8_192
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    grounding: GroundingConfig | None = None
    catalog_dsn: str = "postgresql+psycopg://webbgpt:webbgpt@localhost:5432/webbgpt"
    catalog_input_path: str = "data/catalog/webb_catalog.json"
    enable_grounding: bool = True
    enable_citations: bool = True
    decode_preset: str = "serve"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 4
    stop_strings: list[str] = field(default_factory=_default_stop_strings)
    transcript_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ServeConfig":
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
