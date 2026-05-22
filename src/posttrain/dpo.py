from __future__ import annotations

import math

from config import DataConfig, ModelConfig, TrainConfig


LEGACY_DPO_MESSAGE = (
    "WebbGPT DPO training is archived under junk/dpo-legacy/ and is not part of "
    "the active final-demo pipeline. The small local-MVP DPO run worsened real "
    "sample behavior, so use curated pretraining, optional continued pretraining, "
    "SFT with grounding, and the website demo path instead."
)


def _require_torch():
    import torch

    return torch


def _sequence_log_probs(model, input_ids, attention_mask):
    torch = _require_torch()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].to(token_log_probs.dtype)
    return (token_log_probs * mask).sum(dim=-1)


def _apply_dpo_overrides(train_config: TrainConfig) -> TrainConfig:
    stage_config = TrainConfig.from_dict(train_config.to_dict())
    if stage_config.dpo_learning_rate is not None:
        stage_config.learning_rate = stage_config.dpo_learning_rate
    if stage_config.dpo_min_learning_rate is not None:
        stage_config.min_learning_rate = stage_config.dpo_min_learning_rate
    if stage_config.dpo_warmup_steps is not None:
        stage_config.warmup_steps = stage_config.dpo_warmup_steps
    if stage_config.dpo_max_steps is not None:
        stage_config.max_steps = stage_config.dpo_max_steps
    return stage_config


def _compute_dpo_schedule(
    *,
    train_loader_steps: int,
    stage_config: TrainConfig,
) -> tuple[int, int, int, int]:
    steps_per_epoch = max(
        1,
        math.ceil(train_loader_steps / max(stage_config.gradient_accumulation_steps, 1)),
    )
    effective_max_steps = stage_config.max_steps
    if stage_config.dpo_max_epochs is not None and stage_config.dpo_max_epochs > 0:
        effective_max_steps = min(effective_max_steps, steps_per_epoch * stage_config.dpo_max_epochs)
    eval_interval = max(1, math.ceil(steps_per_epoch / max(stage_config.dpo_evals_per_epoch, 1)))
    early_eval_step = min(10, eval_interval)
    return steps_per_epoch, effective_max_steps, eval_interval, early_eval_step


def _dpo_scale_blockers(
    *,
    train_examples: int,
    validation_examples: int,
    stage_config: TrainConfig,
) -> list[str]:
    blockers: list[str] = []
    if stage_config.dpo_min_train_examples > 0 and train_examples < stage_config.dpo_min_train_examples:
        blockers.append("dpo_train_dataset_too_small")
    if (
        stage_config.dpo_min_validation_examples > 0
        and validation_examples < stage_config.dpo_min_validation_examples
    ):
        blockers.append("dpo_validation_dataset_too_small")
    return blockers


def evaluate_dpo_model(policy_model, reference_model, dataloader, max_batches: int, beta: float) -> dict[str, float]:
    """Legacy metric helper retained for old reports/tests; no training entrypoint remains active."""

    torch = _require_torch()
    device = next(policy_model.parameters()).device
    policy_training = bool(getattr(policy_model, "training", False))
    policy_model.eval()
    reference_model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_margin = torch.tensor(0.0, device=device)
    total_examples = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            policy_chosen = _sequence_log_probs(policy_model, chosen_input_ids, chosen_attention_mask)
            policy_rejected = _sequence_log_probs(policy_model, rejected_input_ids, rejected_attention_mask)
            ref_chosen = _sequence_log_probs(reference_model, chosen_input_ids, chosen_attention_mask)
            ref_rejected = _sequence_log_probs(reference_model, rejected_input_ids, rejected_attention_mask)
            logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
            losses = -torch.nn.functional.logsigmoid(logits)
            margins = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            batch_size = torch.tensor(float(chosen_input_ids.size(0)), device=device)
            total_loss += losses.sum()
            total_correct += (margins > 0).to(torch.float32).sum()
            total_margin += margins.sum()
            total_examples += batch_size
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_margin, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)
    total = max(float(total_examples.item()), 1.0)
    policy_model.train(policy_training)
    return {
        "val_dpo_loss": float(total_loss.item() / total),
        "preference_accuracy": float(total_correct.item() / total),
        "mean_margin": float(total_margin.item() / total),
        "examples_evaluated": int(total_examples.item()),
    }


def run_dpo_job(
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    reference_checkpoint: str,
    beta: float = 0.1,
) -> None:
    raise RuntimeError(LEGACY_DPO_MESSAGE)
