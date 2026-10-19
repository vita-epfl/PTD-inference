# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed setup, precision policy and FSDP2 sharding.

Large networks are built on rank 0 only (other ranks hold meta tensors), sharded with FSDP2, and filled from rank 0
by a broadcast, so the base model on disk is read once per job.  Parameters are stored in ``storage_dtype`` (float32,
as in training) and computed in bfloat16, except for the modules a network keeps in float32.
"""

from __future__ import annotations

import copy
import datetime
import os
import random
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def init_distributed() -> None:
    """Initialise the process group from the ``torchrun`` environment (NCCL for CUDA tensors, gloo for CPU)."""
    if dist.is_initialized():
        return
    timeout = datetime.timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", "3600")))
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", init_method="env://", timeout=timeout)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def set_seed(seed: int) -> int:
    seed %= 1 << 31
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def set_cuda_backend() -> None:
    """TF32 matmuls and convolutions, non-deterministic cuDNN autotuning (the setting the students were trained in)."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def cast_(module: nn.Module, dtype: torch.dtype, keep_fp32: Callable[[str], bool]) -> nn.Module:
    """Cast floating-point parameters and buffers to ``dtype``, except those whose qualified name ``keep_fp32``
    selects, which go to float32 directly (never through ``dtype``)."""
    targets = {}
    for name, t in list(module.named_parameters(remove_duplicate=False)) + list(module.named_buffers(remove_duplicate=False)):
        if t is not None and t.is_floating_point():
            targets[id(t)] = torch.float32 if keep_fp32(name) else dtype

    def convert(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=targets.get(id(t), dtype)) if t.is_floating_point() else t

    module._apply(convert)
    return module


def mp_policies(compute_dtype: torch.dtype = torch.bfloat16, reduce_dtype: torch.dtype = torch.float32):
    """Mixed-precision policies: bfloat16 compute, and float32 compute for a network's float32 modules.  Inputs are
    never cast by FSDP (the networks cast where the release does)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return (MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False),
            MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=reduce_dtype, cast_forward_inputs=False))


def build_on_rank0(factory: Callable[[], nn.Module]) -> nn.Module:
    """``factory()`` with real weights on rank 0 and on the meta device elsewhere."""
    if is_main():
        return factory()
    with torch.device("meta"):
        return factory()


def shard(net: nn.Module, storage_dtype: torch.dtype = torch.float32) -> nn.Module:
    """Cast (keeping the network's float32 modules), shard over all ranks, and broadcast rank 0's weights."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh

    cast_(net, storage_dtype, net.keep_fp32)
    state = copy.deepcopy(net.state_dict()) if is_main() else None
    mesh = init_device_mesh("cuda", (world_size(),))
    bf16_policy, fp32_policy = mp_policies()
    net.shard(mesh, bf16_policy, fp32_policy)
    net.to_empty(device=torch.cuda.current_device())
    net.reset_buffers()
    set_model_state_dict(net, model_state_dict=state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    del state
    torch.cuda.empty_cache()
    barrier()
    return net
