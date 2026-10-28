# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The released MiniMax-H3 students.

``h3_4step_sde`` / ``h3_6step_sde`` (the checkpoints of ``epfl-vita/PTD-weights``) and ``h3_4step`` / ``h3_6step`` (the paper's
recipes): MiniMax-H3 (33B, 768x1344, 124 frames at 24 fps, stereo audio) with one rank-128 LoRA per hop on the attention
and feed-forward projections of all transformer and token-refiner blocks, and one per-hop LoRA on both output projections.
The hop grids are knots of the shift-6 scheduler ``t = 6 u / (1 + 5 u)`` (first knot clamped to the 0.999 edge of the
clock).  States are carried in float32 between hops.

Hop transitions (``renoise``, ``ptd.sampler.hop_transition``): in the ``_sde`` recipes (``renoise = 1``) the state carried
from a hop to the next is the hop's clean endpoint re-noised to the next knot with fresh noise instead of the Euler
endpoint, as in their training (the last hop is unchanged); the paper's recipes (``renoise = 0``) carry the Euler endpoint.
A checkpoint is sampled with the recipe it was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Recipe:
    name: str
    hops: int
    grid: Tuple[float, ...]  # hop grid t_0 > t_1 > ... > t_N = 0
    lora_rank: int = 128
    frames: int = 124
    fp32_carry: bool = True  # carry the state in float32 between hops (as in training)
    # fraction of the carried state's noise variance redrawn with fresh noise at every hop transition but the last
    # (ptd.sampler.hop_transition): 0 = the Euler endpoint, 1 = the hop's clean endpoint re-noised to the next knot
    renoise: float = 0.0


RECIPES: Dict[str, Recipe] = {
    # re-noised hop transitions: the recipes of the checkpoints ptd_h3_4step_sde / ptd_h3_6step_sde
    "h3_4step_sde": Recipe(name="h3_4step_sde", hops=4, grid=(0.999, 0.947, 0.857, 0.667, 0.0), renoise=1.0),
    "h3_6step_sde": Recipe(name="h3_6step_sde", hops=6, grid=(0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0), renoise=1.0),
    # the paper's recipes: Euler hop transitions, for the paper's checkpoints
    "h3_4step": Recipe(name="h3_4step", hops=4, grid=(0.999, 0.947, 0.857, 0.667, 0.0)),
    "h3_6step": Recipe(name="h3_6step", hops=6, grid=(0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0)),
}


def get_recipe(name: str) -> Recipe:
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; available: {sorted(RECIPES)}")
    return RECIPES[name]
