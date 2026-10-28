# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rectified-flow time on a shared clock.

A modality is noised as ``x_t = (1 - sigma(t)) x0 + sigma(t) eps`` with ``t = 0`` clean and ``t = T_MAX`` noise, and
``sigma = f_s`` an exponential time shift of the shared clock ``t``:

    f_s(t) = s t T / ((s - 1) t + T),     T = T_MAX,     f_s . f_r = f_{s r},     f_1 = identity.

Video uses ``s = 1``.  MiniMax-H3 audio uses ``s = 3 / 12``: the H3 release samples video on a shift-12 clock and audio
on a shift-3 clock, and putting video on the shared clock leaves audio with the relative warp ``3 / 12``.  Network
outputs are ``flow = dx/dt = sigma'(t) (eps - x0)`` on the shared clock.  Time arithmetic is float64.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

T_MAX = 0.999


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[B]`` -> ``[B, 1, ..., 1]`` with ``target``'s rank."""
    x = torch.atleast_1d(x)
    while x.dim() < target.dim():
        x = x[..., None]
    return x


def time_shift(t: torch.Tensor, shift: float, max_t: float = T_MAX) -> torch.Tensor:
    """``f_shift(t)`` on ``[0, max_t]`` (the identity, bit-exactly, at ``shift = 1``)."""
    if shift == 1.0:
        return t
    return t * shift * max_t / (t * (shift - 1) + max_t)


def safe_clamp(t: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None) -> torch.Tensor:
    """Clamp so that ``min <= t <= max`` holds in ``t``'s precision."""
    if t.dtype == torch.float64:
        return torch.clamp(t, min=min, max=max)
    lo = hi = None
    if min is not None:
        lo = torch.as_tensor(min, dtype=t.dtype, device=t.device)
        lo = torch.where(lo.to(torch.float64) < min, torch.nextafter(lo, torch.full_like(lo, torch.inf)), lo)
    if max is not None:
        hi = torch.as_tensor(max, dtype=t.dtype, device=t.device)
        hi = torch.where(hi.to(torch.float64) > max, torch.nextafter(hi, torch.full_like(hi, -torch.inf)), hi)
    return torch.clamp(t, min=lo, max=hi)


class Clock:
    """``sigma(t) = f_shift(t)`` of one modality on the shared clock."""

    def __init__(self, shift: float = 1.0, max_t: float = T_MAX):
        self.shift = float(shift)
        self.max_t = float(max_t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return time_shift(t, self.shift, self.max_t)

    def sigma_prime(self, t: torch.Tensor) -> torch.Tensor:
        if self.shift == 1.0:
            return torch.ones_like(t)
        return self.shift * self.max_t**2 / (t * (self.shift - 1) + self.max_t) ** 2

    def shift_t(self, t: torch.Tensor, shift: float) -> torch.Tensor:
        """Put the shared time ``t`` itself under a further shift (e.g. to lay out a sampling grid)."""
        return safe_clamp(time_shift(t, shift, self.max_t), max=self.max_t)

    def latents(self, noise: torch.Tensor, t_init: torch.Tensor) -> torch.Tensor:
        """The noisy state at ``t_init`` for pure noise: ``sigma(t_init) * noise``."""
        sigma = expand_like(self.sigma(t_init.to(torch.float64)), noise)
        return (noise.to(torch.float64) * sigma).to(noise.dtype)

    def flow_to_x0(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``x0 = x_t - (sigma / sigma') flow`` (float64 arithmetic, ``x_t``'s dtype)."""
        t64 = t.to(torch.float64)
        sp = self.sigma_prime(t64)
        sp = torch.where(sp >= 0, sp.clamp(min=1e-6), sp.clamp(max=-1e-6))
        scale = expand_like(self.sigma(t64) / sp, xt)
        return (xt.to(torch.float64) - v.to(torch.float64).mul(scale)).to(xt.dtype)


def explicit_grid(timesteps: Sequence[float], device=None, max_t: float = T_MAX) -> torch.Tensor:
    """A student grid ``t_0 > ... > t_N = 0`` given by its knots (float64)."""
    grid = torch.as_tensor(list(timesteps), device=device, dtype=torch.float64)
    if grid.dim() != 1 or grid.numel() < 2 or float(grid[-1]) != 0.0 or not bool((grid[:-1] > grid[1:]).all()):
        raise ValueError(f"a student grid must be strictly decreasing and end at 0, got {list(timesteps)}")
    if float(grid[0]) > max_t:
        raise ValueError(f"a student grid must lie in [0, {max_t}], got {list(timesteps)}")
    return safe_clamp(grid, min=0.0, max=max_t)
