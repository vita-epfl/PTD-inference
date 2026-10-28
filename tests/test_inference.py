# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the clock, the hop grids, the hop modules and the hop transitions (``pytest tests`` from the repository root)."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from ptd.modules import HopLoRALinear, HopRef, PTDOutputProj
from ptd.recipes import RECIPES, get_recipe
from ptd.sampler import hop_grid, hop_transition, sample
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


# ------------------------------------------------------------------------------------------------------ hop transitions
class _Toy:
    """A two-modality student on the MiniMax-H3 clocks with hop-specific affine bfloat16 heads."""

    N_HOPS = 4

    def __init__(self):
        self.clocks = {"video": Clock(1.0), "audio": Clock(0.25)}

    def latents(self, noise, t):
        return {m: self.clocks[m].latents(noise[m], t) for m in noise}

    def flow_to_x0(self, x, v, t):
        return {m: self.clocks[m].flow_to_x0(x[m], v[m], t) for m in x}

    def sigma(self, t):
        return {m: c.sigma(t) for m, c in self.clocks.items()}

    def __call__(self, x, t, condition, hop):
        return {"video": (0.5 * x["video"] - 0.1 * (hop + 1)).to(torch.bfloat16),
                "audio": (0.25 * x["audio"] + 0.05 * (hop + 1)).to(torch.bfloat16)}


def _state():
    g = torch.Generator().manual_seed(0)
    return {"video": torch.randn(2, 3, 2, 2, 2, generator=g), "audio": torch.randn(2, 4, 3, generator=g)}


def _bvc(v, ref):
    return v.view(-1, *([1] * (ref.dim() - 1))).to(ref.dtype)


def _euler(x, h, t_a, t_e):
    return {m: x[m] + _bvc(t_e - t_a, x[m]) * h[m].float() for m in x}


def test_sde_recipes_differ_from_the_paper_recipes_in_name_and_transition_only():
    for name in ("h3_4step", "h3_6step"):
        paper, sde = asdict(get_recipe(name)), asdict(get_recipe(name + "_sde"))
        assert paper.pop("renoise") == 0.0 and sde.pop("renoise") == 1.0
        assert paper.pop("name") == name and sde.pop("name") == name + "_sde"
        assert paper == sde


def test_hop_transition_is_the_euler_endpoint_without_renoise_and_on_the_last_hop():
    net, x = _Toy(), _state()
    grid = hop_grid(get_recipe("h3_4step_sde"), None)
    t_a, t_e = grid[0].expand(2), grid[1].expand(2)
    h = net(x, t_a, None, hop=0)
    euler = _euler(x, h, t_a, t_e)
    plain = hop_transition(net, x, h, t_a, t_e, 0.0)
    assert all(torch.equal(plain[m], euler[m]) and plain[m].dtype == torch.float32 for m in x)
    last_a, last_e = grid[-2].expand(2), grid[-1].expand(2)
    h_last = net(x, last_a, None, hop=3)
    last = hop_transition(net, x, h_last, last_a, last_e, 1.0, torch.Generator().manual_seed(0))
    assert all(torch.equal(last[m], _euler(x, h_last, last_a, last_e)[m]) for m in x)
    with pytest.raises(ValueError):
        hop_transition(net, x, h, t_a, t_e, 1.5)


def test_hop_transition_keeps_the_clean_endpoint_and_replaces_the_noise():
    net, x = _Toy(), _state()
    grid = hop_grid(get_recipe("h3_4step_sde"), None)
    t_a, t_e = grid[1].expand(2), grid[2].expand(2)
    h = net(x, t_a, None, hop=1)
    x0 = net.flow_to_x0(x, {m: v.float() for m, v in h.items()}, t_a)
    sig = net.sigma(t_e)

    def gen(seed):
        return torch.Generator().manual_seed(seed)

    full, again, other = (hop_transition(net, x, h, t_a, t_e, 1.0, gen(s)) for s in (1, 1, 2))
    g = gen(1)
    fresh = {m: torch.randn(x[m].shape, generator=g) for m in x}  # the draws of seed 1, video first
    half = hop_transition(net, x, h, t_a, t_e, 0.5, gen(1))
    euler = _euler(x, h, t_a, t_e)
    for m in x:
        sg = _bvc(sig[m].float(), x[m])
        assert torch.equal(full[m], again[m]) and not torch.equal(full[m], other[m])
        # fraction 1: the clean endpoint is kept, the noise is a fresh standard draw from the generator
        assert torch.allclose(full[m], (1 - sg) * x0[m] + sg * fresh[m], atol=1e-6, rtol=0)
        # fraction 1/2: the endpoint's own noise and the fresh draw are mixed with equal weight in variance
        eps = (euler[m] - (1 - sg) * x0[m]) / sg
        expected = (1 - sg) * x0[m] + sg * (0.5 ** 0.5 * eps + 0.5 ** 0.5 * fresh[m])
        assert torch.allclose(half[m], expected, atol=1e-6, rtol=0)


def test_sample_is_bit_for_bit_the_euler_sampler_at_renoise_zero_and_deterministic_under_a_generator():
    net, noise = _Toy(), _state()
    grid = hop_grid(get_recipe("h3_4step_sde"), None)
    euler = sample(net, noise, None, grid)
    assert all(torch.equal(sample(net, noise, None, grid, renoise=0.0)[m], euler[m]) for m in noise)

    def gen(seed):
        return torch.Generator().manual_seed(seed)

    a, b, c = (sample(net, noise, None, grid, renoise=1.0, generator=gen(s)) for s in (5, 5, 6))
    for m in noise:
        assert a[m].dtype == noise[m].dtype
        assert torch.equal(a[m], b[m]) and not torch.equal(a[m], c[m]) and not torch.equal(a[m], euler[m])
