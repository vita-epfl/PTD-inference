# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The released MiniMax-H3 students.

``h3_4step`` / ``h3_6step``: MiniMax-H3 (33B, 768x1344, 124 frames at 24 fps, stereo audio) with one rank-128 LoRA per
hop on the attention and feed-forward projections of all transformer and token-refiner blocks, and one per-hop LoRA on
both output projections.  The hop grids are knots of the shift-6 scheduler ``t = 6 u / (1 + 5 u)`` (first knot clamped
to the 0.999 edge of the clock).  States are carried in float32 between hops.
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


RECIPES: Dict[str, Recipe] = {
    "h3_4step": Recipe(name="h3_4step", hops=4, grid=(0.999, 0.947, 0.857, 0.667, 0.0)),
    "h3_6step": Recipe(name="h3_6step", hops=6, grid=(0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0)),
}


def get_recipe(name: str) -> Recipe:
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; available: {sorted(RECIPES)}")
    return RECIPES[name]
