# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N-step sampling of a PTD student.

The student takes N hops on the recipe's grid ``t_0 > t_1 > ... > t_N = 0``; hop ``k`` is one forward pass of the hop-k
expert, ``x_{k+1} = x_k + (t_{k+1} - t_k) h(x_k, t_k)``, where ``h`` is the predicted mean velocity of the hop.

Hop transitions (``Recipe.renoise``, ``hop_transition``): the ``_sde`` recipes carry from every hop but the last not the
Euler endpoint but the hop's clean endpoint ``x0 = x_k - (sigma / sigma')(t_k) h`` re-noised to the next knot with fresh
noise, ``x_{k+1} = (1 - sigma(t_{k+1})) x0 + sigma(t_{k+1}) eps'`` per modality on its clock, as in their training.  The
fresh noise comes from the ``generator`` passed to ``sample`` (``ptd.sample`` seeds it with the clip's seed, like the
initial noise); ``renoise = 0`` leaves the Euler recurrence bit for bit.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from ptd.schedule import explicit_grid
from ptd.utils import TensorOrDict, batch_size, first_tensor, per_modality, tmap, to


def hop_grid(recipe, device) -> torch.Tensor:
    return explicit_grid(recipe.grid, device=device)


def input_shape(recipe):
    from ptd.models.h3 import h3_input_shape

    return h3_input_shape(recipe.frames)


def _bvc(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-sample scalar ``[B]`` broadcast against ``ref``."""
    return v.view(-1, *([1] * (ref.dim() - 1))).to(ref.dtype)


def hop_transition(net, x_k: TensorOrDict, h: TensorOrDict, t_a: torch.Tensor, t_e: torch.Tensor, frac: float,
                   generator: Optional[torch.Generator] = None) -> TensorOrDict:
    """The float32 state carried from a hop that starts at ``x_k`` (time ``t_a``, ``[B]``) with mean velocity ``h`` to the
    next knot ``t_e``: the Euler endpoint ``x_k + dt h`` with a fraction ``frac`` of its noise variance redrawn
    (``Recipe.renoise``; module docstring).  ``frac = 0`` or a final knot (``t_e = 0``): the plain endpoint."""
    if not 0.0 <= float(frac) <= 1.0:
        raise ValueError(f"renoise must lie in [0, 1], got {frac}")
    x32, h32 = to(x_k, dtype=torch.float32), to(tmap(lambda v: v.detach(), h), dtype=torch.float32)
    dt = t_e - t_a
    x_e = per_modality(lambda x, hh: x + _bvc(dt, x) * hh, x32, h32)
    if float(frac) <= 0.0 or float(t_e.flatten()[0]) <= 0.0:
        return x_e
    if not (hasattr(net, "flow_to_x0") and hasattr(net, "sigma")):
        raise ValueError("re-noised hop transitions need the network's sigma / flow_to_x0 (the clock of every modality)")
    x0 = to(net.flow_to_x0(x32, h32, t_a), dtype=torch.float32)  # the hop's clean endpoint: x_k - (sigma / sigma')(t_a) h
    sig = net.sigma(t_e)
    keep, new = math.sqrt(max(1.0 - float(frac), 0.0)), math.sqrt(min(float(frac), 1.0))

    def mix(xe, x0m, sg):
        sg = _bvc(sg.to(torch.float32).clamp(min=1e-6), xe)
        fresh = torch.randn(xe.shape, generator=generator, device=xe.device, dtype=torch.float32)
        if float(frac) >= 1.0:
            return (1.0 - sg) * x0m + sg * fresh
        eps = (xe - (1.0 - sg) * x0m) / sg  # the noise the endpoint carries from the initial draw
        return (1.0 - sg) * x0m + sg * (keep * eps + new * fresh)

    return per_modality(mix, x_e, x0, sig)


def euler_hop(net, state: TensorOrDict, step_index: int, t_list: torch.Tensor, condition: Any,
              precision: torch.dtype = torch.bfloat16, renoise: float = 0.0,
              generator: Optional[torch.Generator] = None) -> TensorOrDict:
    """One student hop ``x_{k+1} = x_k + dt_k h`` (in the state's dtype).  With fewer or more sampling steps than hops,
    each interval uses the hop expert that owns its start.  ``renoise`` > 0: the re-noised transition (``hop_transition``)."""
    num_steps = len(t_list) - 1
    bsz = batch_size(state)
    t_a, t_e = t_list[step_index], t_list[step_index + 1]
    hop = step_index * int(net.N_HOPS) // num_steps
    h = net(to(state, dtype=precision), t_a.expand(bsz), condition, hop=hop)
    if float(renoise) <= 0.0:  # the paper's recipes
        dt = (t_e - t_a).to(torch.float32)
        return per_modality(lambda s, hh: (s.to(torch.float32) + dt * hh.to(torch.float32)).to(s.dtype), state, h)
    x_e = hop_transition(net, state, h, t_a.expand(bsz), t_e.expand(bsz), renoise, generator)
    return per_modality(lambda s, xe: xe.to(s.dtype), state, x_e)


@torch.no_grad()
def sample(net, noise: TensorOrDict, condition: Any, t_list: torch.Tensor, fp32_carry: bool = True,
           precision: torch.dtype = torch.bfloat16, renoise: float = 0.0,
           generator: Optional[torch.Generator] = None) -> TensorOrDict:
    """N-step student sampling from pure noise; returns latents in the noise's dtype.  ``renoise`` > 0 re-noises every
    hop transition but the last (``Recipe.renoise``), drawing the fresh noise from ``generator`` (``None``: the global RNG)."""
    x = to(noise, dtype=torch.float32) if fp32_carry else noise
    x = net.latents(x, t_list[0])
    for k in range(len(t_list) - 1):
        x = euler_hop(net, x, k, t_list, condition, precision, renoise=renoise, generator=generator)
    return to(x, dtype=first_tensor(noise).dtype)
