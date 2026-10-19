# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading PTD checkpoints (``torch.distributed.checkpoint`` directories, loadable at any world size).

A released checkpoint holds the student's trained parameters (the per-hop LoRA adapters) and buffers; the frozen base
weights come from the MiniMax-H3 release.  Training-only tensors stored in the checkpoint are not loaded.  ``load_model``
accepts a checkpoint directory itself (any directory with a ``.metadata`` file) or a directory holding it as ``model/``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict


def model_dir(path: str) -> str:
    return path if os.path.exists(os.path.join(path, ".metadata")) else os.path.join(path, "model")


def load_model(net: torch.nn.Module, path: str) -> None:
    """Load the adapters (the network's trainable parameters) and buffers of ``net`` from a checkpoint."""
    options = StateDictOptions(ignore_frozen_params=True, strict=False)
    state = get_model_state_dict(net, options=options)
    dcp.load(state, checkpoint_id=model_dir(path))
    set_model_state_dict(net, state, options=options)
