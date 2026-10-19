# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 as a PTD student (``h3``).

The network takes ``(x_t, t, condition, hop)`` with ``t = 0`` clean and returns the hop's mean velocity on the shared
clock.  ``build_student`` constructs it on rank 0 (meta elsewhere), shards it with FSDP2 and broadcasts the weights.
"""

from __future__ import annotations

import torch

from ptd.dist import build_on_rank0, shard
from ptd.recipes import Recipe


def default_model() -> str:
    return "MiniMaxAI/MiniMax-H3"


def build_student(recipe: Recipe, model: str, transformer_subfolder: str = "transformer"):
    from ptd.models.h3 import MiniMaxH3PTD, resolve_model_root

    root = resolve_model_root(model)
    net = build_on_rank0(lambda: MiniMaxH3PTD(root, hops=recipe.hops, ranks=[recipe.lora_rank] * recipe.hops,
                                              refiner_lora=True, transformer_subfolder=transformer_subfolder))
    shard(net, storage_dtype=torch.float32)
    net.eval()
    return net
