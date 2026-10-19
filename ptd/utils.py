# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for states that are either a tensor or a ``{"video", "audio"}`` dict (MiniMax-H3)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict, Union

import torch

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]
DICT_TYPES = (Mapping, torch.nn.ModuleDict)


def per_modality(fn: Callable, primary: Any, *args: Any, **kwargs: Any) -> Any:
    """``fn`` per key of a dict ``primary`` (dict arguments are indexed per key, others broadcast), or ``fn(primary, ...)``."""
    if not isinstance(primary, DICT_TYPES):
        return fn(primary, *args, **kwargs)

    def _idx(v: Any, k: Any) -> Any:
        return v[k] if isinstance(v, DICT_TYPES) else v

    return {k: fn(primary[k], *(_idx(a, k) for a in args), **{kk: _idx(v, k) for kk, v in kwargs.items()}) for k in primary}


def tmap(fn: Callable[[torch.Tensor], Any], structure: Any) -> Any:
    """``fn`` on every tensor of a nested dict / list / tuple; other leaves pass through."""
    if isinstance(structure, torch.Tensor):
        return fn(structure)
    if isinstance(structure, DICT_TYPES):
        return {k: tmap(fn, v) for k, v in structure.items()}
    if isinstance(structure, (list, tuple)):
        return type(structure)(tmap(fn, v) for v in structure)
    return structure


def to(structure: Any, device: Any = None, dtype: torch.dtype | None = None) -> Any:
    """Move every tensor to ``device`` and cast floating-point tensors to ``dtype``."""
    non_blocking = device is not None and torch.device(device).type != "cpu"

    def _cast(x: torch.Tensor) -> torch.Tensor:
        if x.is_floating_point():
            return x.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return x.to(device=device, non_blocking=non_blocking)

    return tmap(_cast, structure)


def first_tensor(structure: TensorOrDict) -> torch.Tensor:
    return next(iter(structure.values())) if isinstance(structure, DICT_TYPES) else structure


def batch_size(structure: TensorOrDict) -> int:
    return first_tensor(structure).shape[0]


def randn(shape: Union[list, Dict[str, list]], batch: int, device, dtype: torch.dtype) -> TensorOrDict:
    """Gaussian noise ``[batch, *shape]`` per modality (drawn in dict order)."""
    return per_modality(lambda s: torch.randn(batch, *s, device=device, dtype=dtype), shape)
