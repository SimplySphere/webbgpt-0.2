from config import DataConfig, DataSourceConfig, EvalConfig, GroundingConfig, ModelConfig, TokenizerConfig, TrainConfig


def test_model_config_head_dim():
    config = ModelConfig(hidden_size=512, num_attention_heads=8)
    assert config.head_dim == 64


def test_train_config_roundtrip():
    config = TrainConfig(
        continued_learning_rate=1e-4,
        continued_min_learning_rate=1e-5,
        continued_warmup_steps=25,
        sft_learning_rate=2e-4,
        sft_min_learning_rate=2e-5,
        sft_warmup_steps=12,
        sft_max_epochs=7,
        sft_validation_fraction=0.2,
        sft_validation_min_examples=3,
        require_explicit_sft_validation=True,
        sft_evals_per_epoch=5,
        sft_min_eval_interval_steps=40,
        sft_early_stopping_patience_evals=4,
        sft_best_min_delta=0.03,
        sft_sample_every_steps=250,
        dpo_learning_rate=5e-5,
        dpo_min_learning_rate=5e-6,
        dpo_warmup_steps=8,
        dpo_validation_fraction=0.25,
        dpo_validation_min_examples=4,
        dpo_min_train_examples=64,
        dpo_min_validation_examples=12,
        require_explicit_dpo_validation=True,
        dpo_evals_per_epoch=6,
        dpo_early_stopping_patience_evals=3,
        dpo_best_min_delta=0.01,
        dpo_enable_lm_health_eval=True,
        allow_weak_posttrain_validation=True,
        posttrain_top_k_checkpoints=5,
    )
    payload = config.to_dict()
    restored = TrainConfig.from_dict(payload)
    assert restored.run_name == config.run_name
    assert restored.checkpoint.output_dir == config.checkpoint.output_dir
    assert restored.continued_learning_rate == 1e-4
    assert restored.continued_min_learning_rate == 1e-5
    assert restored.continued_warmup_steps == 25
    assert restored.sft_learning_rate == 2e-4
    assert restored.sft_min_learning_rate == 2e-5
    assert restored.sft_warmup_steps == 12
    assert restored.sft_max_epochs == 7
    assert restored.sft_validation_fraction == 0.2
    assert restored.sft_validation_min_examples == 3
    assert restored.require_explicit_sft_validation is True
    assert restored.sft_evals_per_epoch == 5
    assert restored.sft_min_eval_interval_steps == 40
    assert restored.sft_early_stopping_patience_evals == 4
    assert restored.sft_best_min_delta == 0.03
    assert restored.sft_sample_every_steps == 250
    assert restored.dpo_learning_rate == 5e-5
    assert restored.dpo_min_learning_rate == 5e-6
    assert restored.dpo_warmup_steps == 8
    assert restored.dpo_validation_fraction == 0.25
    assert restored.dpo_validation_min_examples == 4
    assert restored.dpo_min_train_examples == 64
    assert restored.dpo_min_validation_examples == 12
    assert restored.require_explicit_dpo_validation is True
    assert restored.dpo_evals_per_epoch == 6
    assert restored.dpo_early_stopping_patience_evals == 3
    assert restored.dpo_best_min_delta == 0.01
    assert restored.dpo_enable_lm_health_eval is True
    assert restored.allow_weak_posttrain_validation is True
    assert restored.posttrain_top_k_checkpoints == 5


def test_tokenizer_defaults_include_special_tokens():
    config = TokenizerConfig()
    assert config.special_tokens["assistant_token"] == "<|assistant|>"


def test_data_config_domain_tags_present():
    config = DataConfig(
        lm_weighted_source_token_budget=4096,
        lm_max_source_token_share=0.55,
        lm_max_source_repeat_rate=0.2,
        continue_readiness_min_clean_token_fraction=0.6,
        continue_readiness_min_documents=100,
        continue_readiness_min_source_families=2,
        continue_readiness_max_single_source_share=0.7,
        continue_readiness_max_repeat_rate=0.15,
    )
    assert "course_catalog" in config.domain_tags
    restored = DataConfig.from_dict(config.to_dict())
    assert restored.lm_weighted_source_token_budget == 4096
    assert restored.continue_readiness_min_documents == 100
    assert restored.continue_readiness_max_repeat_rate == 0.15


def test_data_source_config_roundtrip_with_extended_fields():
    source = DataSourceConfig(
        name="fineweb",
        format="hf",
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config_name="sample-10BT",
        dataset_revision="main",
        paths=["data/shard-000.jsonl", "data/shard-001.jsonl"],
        messages_field="messages",
        prompt_field="prompt",
        response_field="response",
        chosen_field="chosen",
        rejected_field="rejected",
        streaming=True,
        skip_records=128,
        max_records=1024,
        id_field="example_id",
        group_field="conversation_id",
        family="public_prose",
        quality_filter_mode="broad_lm",
    )
    restored = DataSourceConfig.from_dict(source.to_dict())
    assert restored.dataset_name == source.dataset_name
    assert restored.paths == source.paths
    assert restored.streaming is True
    assert restored.max_records == 1024
    assert restored.response_field == "response"
    assert restored.id_field == "example_id"
    assert restored.group_field == "conversation_id"
    assert restored.family == "public_prose"
    assert restored.quality_filter_mode == "broad_lm"


def test_eval_config_roundtrip_with_release_gates():
    config = EvalConfig(
        enforce_release_gates=True,
        catalog_dsn="sqlite:///artifacts/catalog/eval.db",
        decode_preset="release",
        repetition_penalty=1.1,
        no_repeat_ngram_size=5,
        grounding=GroundingConfig(
            dsn="sqlite:///artifacts/grounding/webbgpt.db",
            seed_url_pack="data/webb/seed_urls_demo.json",
            offline_seed_url_pack="data/webb/seed_urls_private.json",
            handbook_url="data/webb/mock/handbook.txt",
            sync_on_start=True,
        ),
    )
    restored = EvalConfig.from_dict(config.to_dict())
    assert restored.enforce_release_gates is True
    assert restored.catalog_dsn == "sqlite:///artifacts/catalog/eval.db"
    assert restored.decode_preset == "release"
    assert restored.repetition_penalty == 1.1
    assert restored.no_repeat_ngram_size == 5
    assert restored.grounding is not None
    assert restored.grounding.seed_url_pack == "data/webb/seed_urls_demo.json"
    assert restored.grounding.offline_seed_url_pack == "data/webb/seed_urls_private.json"
    assert restored.grounding.sync_on_start is True
    assert restored.grounding.route_fanout_limit == 2
    assert restored.grounding.planner_beta_enabled is False
    assert "athletics" in restored.grounding.freshness_policy
    assert restored.release_gates.chat_sanity_pass_rate_min == 0.9
