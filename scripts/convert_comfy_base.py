# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Convert an original Diffusers H3 transformer directory to unpruned ComfyUI weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptd_comfy.base import convert_base_state_dict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="MiniMax-H3/transformer (or transformer_ref)")
    ap.add_argument("output", type=Path, help="New .safetensors file; existing files are never replaced")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output already exists")
    import safetensors.torch

    config = json.loads((args.source / "config.json").read_text())
    index = json.loads((args.source / "diffusion_pytorch_model.safetensors.index.json").read_text())
    state = {}
    for filename in sorted(set(index["weight_map"].values())):
        path = (args.source / filename).resolve()
        if not path.is_relative_to(args.source.resolve()):
            raise ValueError("Shard path escapes the model directory")
        shard = safetensors.torch.load_file(str(path), device="cpu")
        if state.keys() & shard.keys():
            raise ValueError("Duplicate tensor across model shards")
        state.update(shard)
    if set(state) != set(index["weight_map"]):
        raise ValueError("Model shards do not match the tensor index")
    native = convert_base_state_dict(state, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        ap.error("partial output already exists; inspect it before retrying")
    safetensors.torch.save_file(native, str(temporary), metadata={
        "format": "pt", "ptd_base_format": "minimax-h3-unpruned-v1",
        "source_config": json.dumps(config, sort_keys=True),
    })
    # Hard-link publication is atomic and refuses a concurrent existing destination.
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({"tensor_count": len(native), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
