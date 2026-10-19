# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the clock, the hop grids and the hop modules (``pytest tests`` from the repository root)."""

import torch
from torch import nn

from ptd.modules import HopLoRALinear, HopRef, PTDOutputProj
from ptd.recipes import RECIPES
from ptd.schedule import Clock, explicit_grid


def test_grids():
    """The released hop grids are knots of the shift-6 clock (first knot clamped to the 0.999 edge)."""
    for recipe in RECIPES.values():
        g = explicit_grid(recipe.grid)
        n = recipe.hops
        u = torch.arange(n, 0, -1, dtype=torch.float64)[1:] / n
        assert g.dtype == torch.float64 and g.shape == (n + 1,) and float(g[0]) == 0.999 and float(g[-1]) == 0.0
        assert torch.allclose(g[1:-1], 6 * u / (1 + 5 * u), atol=6e-4)


def test_audio_clock():
    """The audio clock is the 3/12 warp of the shared clock: shift-12 video times map to shift-3 audio times."""
    audio = Clock(3.0 / 12.0)
    u = torch.linspace(0.0, 0.999, 11, dtype=torch.float64)
    assert torch.allclose(audio.sigma(Clock().shift_t(u, 12.0)), Clock().shift_t(u, 3.0))
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    h = 1e-6
    fd = (audio.sigma(t + h) - audio.sigma(t - h)) / (2 * h)
    assert torch.allclose(audio.sigma_prime(t), fd, atol=1e-6)


def test_hop_modules():
    """Zero-initialised hop adapters reproduce the base layer; only the active hop's adapter acts."""
    torch.manual_seed(0)
    ref = HopRef(3)
    base = nn.Linear(16, 8)
    lora = HopLoRALinear(base, [4, 4, 4], ref)
    x = torch.randn(5, 16)
    assert torch.equal(lora(x), base(x))
    with torch.no_grad():
        lora.lora_B[1].weight.normal_()
    ref.h = 0
    assert torch.equal(lora(x), base(x))
    ref.h = 1
    assert not torch.allclose(lora(x), base(x))

    proj = PTDOutputProj(nn.Linear(16, 6), ref)
    assert torch.equal(proj(x), proj.main(x)) and proj.main.ranks == [6, 6, 6]
