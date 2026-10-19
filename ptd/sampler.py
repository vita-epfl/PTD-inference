# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N-step sampling of a PTD student.

The student takes N hops on the recipe's grid ``t_0 > t_1 > ... > t_N = 0``; hop ``k`` is one forward pass of the hop-k
expert, ``x_{k+1} = x_k + (t_{k+1} - t_k) h(x_k, t_k)``, where ``h`` is the predicted mean velocity of the hop.
"""

from __future__ import annotations

from typing import Any

import torch

from ptd.schedule import explicit_grid
from ptd.utils import TensorOrDict, batch_size, first_tensor, per_modality, to


def hop_grid(recipe, device) -> torch.Tensor:
    return explicit_grid(recipe.grid, device=device)


def input_shape(recipe):
    from ptd.models.h3 import h3_input_shape

    return h3_input_shape(recipe.frames)


def euler_hop(net, state: TensorOrDict, step_index: int, t_list: torch.Tensor, condition: Any,
              precision: torch.dtype = torch.bfloat16) -> TensorOrDict:
    """One student hop ``x_{k+1} = x_k + dt_k h`` (in the state's dtype).  With fewer or more sampling steps than hops,
    each interval uses the hop expert that owns its start."""
    num_steps = len(t_list) - 1
    bsz = batch_size(state)
    t_a, t_e = t_list[step_index], t_list[step_index + 1]
    hop = step_index * int(net.N_HOPS) // num_steps
    h = net(to(state, dtype=precision), t_a.expand(bsz), condition, hop=hop)
    dt = (t_e - t_a).to(torch.float32)
    return per_modality(lambda s, hh: (s.to(torch.float32) + dt * hh.to(torch.float32)).to(s.dtype), state, h)


@torch.no_grad()
def sample(net, noise: TensorOrDict, condition: Any, t_list: torch.Tensor, fp32_carry: bool = True,
           precision: torch.dtype = torch.bfloat16) -> TensorOrDict:
    """N-step student sampling from pure noise; returns latents in the noise's dtype."""
    x = to(noise, dtype=torch.float32) if fp32_carry else noise
    x = net.latents(x, t_list[0])
    for k in range(len(t_list) - 1):
        x = euler_hop(net, x, k, t_list, condition, precision)
    return to(x, dtype=first_tensor(noise).dtype)
