# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Native MODEL/SAMPLER/SIGMAS nodes for the released PTD H3 recipes."""

from pathlib import Path

import folder_paths

from .checkpoint import load_adapter_file
from .patcher import get_ptd_controller, patch_ptd_model
from .sampling import make_sampler, ptd_sigmas

folder_paths.add_model_folder_path("ptd", str(Path(folder_paths.models_dir) / "ptd"))


class PTDH3Loader:
    CATEGORY = "PTD"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = "Apply PTD hop adapters to an unpruned native MiniMax-H3 model. Use PTD Sampler and PTD Sigmas."

    @classmethod
    def INPUT_TYPES(cls):
        files = [f for f in folder_paths.get_filename_list("ptd") if f.endswith(".safetensors")]
        return {"required": {"model": ("MODEL",), "checkpoint": (files,)}}

    def load(self, model, checkpoint):
        path = folder_paths.get_full_path_or_raise("ptd", checkpoint)
        tensors, metadata = load_adapter_file(path)
        return (patch_ptd_model(model, tensors, metadata),)


class PTDH3Sampler:
    CATEGORY = "PTD"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    DESCRIPTION = ("PTD's hop updates with the matching per-step adapter and audio clock; _sde adapters re-noise the hop "
                   "transitions with fresh noise from the workflow's seed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def get_sampler(self, model):
        controller = get_ptd_controller(model)
        return (make_sampler(controller, controller.recipe),)


class PTDH3Sigmas:
    CATEGORY = "PTD"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    DESCRIPTION = "The exact four- or six-step grid recorded in the PTD checkpoint."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def get_sigmas(self, model):
        return (ptd_sigmas(get_ptd_controller(model).recipe),)


NODE_CLASS_MAPPINGS = {
    "PTDH3Loader": PTDH3Loader,
    "PTDH3Sampler": PTDH3Sampler,
    "PTDH3Sigmas": PTDH3Sigmas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PTDH3Loader": "Load PTD MiniMax-H3 Adapters",
    "PTDH3Sampler": "PTD Sampler",
    "PTDH3Sigmas": "PTD Sigmas",
}
