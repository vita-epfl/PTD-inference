# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hop-routed adapters of a PTD student.

A PTD student takes N steps ("hops").  Every module below reads the active hop from one shared ``HopRef`` that the
owning network sets at each forward call, so a single backbone hosts the N per-hop students:

* ``HopLoRALinear``: a frozen ``nn.Linear`` plus one LoRA pair ``(A_j, B_j)`` per hop (scale 1).
* ``PTDOutputProj``: the student's output projection, predicting the hop's mean velocity ``h``: a hop-routed LoRA of
  rank ``out_features`` (an exact dense per-hop delta) on the frozen projection.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class HopRef:
    """Routing state shared by all hop modules of one network."""

    def __init__(self, n_hops: int):
        self.n_hops = int(n_hops)
        self.h = 0  # active hop


class HopLoRALinear(nn.Module):
    """Frozen ``nn.Linear`` + one LoRA expert per hop: ``y = W x + b + B_h A_h x`` (A ~ N(0, 1/r), B = 0 before loading)."""

    def __init__(self, base: nn.Linear, ranks: Sequence[int], hop_ref: HopRef):
        super().__init__()
        self.base = base
        self.hop_ref = hop_ref
        self.ranks = [int(r) for r in ranks]
        if len(self.ranks) != hop_ref.n_hops:
            raise ValueError(f"need one LoRA rank per hop ({hop_ref.n_hops}), got {self.ranks}")
        kw = dict(dtype=base.weight.dtype, device=base.weight.device)
        self.lora_A = nn.ModuleList(nn.Linear(base.in_features, r, bias=False, **kw) for r in self.ranks)
        self.lora_B = nn.ModuleList(nn.Linear(r, base.out_features, bias=False, **kw) for r in self.ranks)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if self.base.weight.device.type == "meta":
            return
        for j, r in enumerate(self.ranks):
            nn.init.normal_(self.lora_A[j].weight, std=1.0 / r)
            nn.init.zeros_(self.lora_B[j].weight)

    # Callers that read the wrapped layer's attributes see the frozen base.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base.weight, self.base.bias)
        h = int(self.hop_ref.h)
        xa = x.to(self.lora_A[h].weight.dtype)
        return y + self.lora_B[h](self.lora_A[h](xa)).to(y.dtype)


class PTDOutputProj(nn.Module):
    """Output projection of a PTD student: the mean velocity ``h`` of the active hop, through a hop-routed LoRA of rank
    ``out_features`` on the frozen projection ``main``."""

    def __init__(self, main: nn.Linear, hop_ref: HopRef):
        super().__init__()
        self.hop_ref = hop_ref
        self.main = HopLoRALinear(main, [main.out_features] * hop_ref.n_hops, hop_ref)

    @property
    def weight(self) -> torch.Tensor:
        return self.main.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
