from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager


def get_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_device_name() -> str:
    return str(get_torch_device())


def autocast_if_available(torch, *, device, use_bf16: bool) -> AbstractContextManager:
    if use_bf16 and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
