from __future__ import annotations

import math

from config import TrainConfig


def _require_torch():
    import torch

    return torch


def build_optimizer(model, train_config: TrainConfig):
    torch = _require_torch()
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    parameter_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        fused=torch.cuda.is_available(),
    )


def build_scheduler(
    optimizer,
    train_config: TrainConfig,
    *,
    max_steps_override: int | None = None,
):
    torch = _require_torch()
    scheduler_max_steps = int(max_steps_override or train_config.max_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < train_config.warmup_steps:
            return float(current_step) / float(max(1, train_config.warmup_steps))
        progress = (current_step - train_config.warmup_steps) / float(
            max(1, scheduler_max_steps - train_config.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        floor = train_config.min_learning_rate / max(train_config.learning_rate, 1e-12)
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.webbgpt_max_steps = scheduler_max_steps
    return scheduler
