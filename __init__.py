# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""ComfyUI custom-node entry point; ordinary library imports do not require ComfyUI."""


def __getattr__(name):
    if name in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
        from .ptd_comfy import nodes
        return getattr(nodes, name)
    raise AttributeError(name)
