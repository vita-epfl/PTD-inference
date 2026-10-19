# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a trusted PTD DCP checkpoint into native ComfyUI PTD safetensors.

This copies adapter factors without precision changes or dense-weight merging.
It requires CPU RAM for the adapter tensors, but no GPU or base model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptd_comfy.checkpoint import RECIPE_HOPS, convert_dcp_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="DCP directory containing .metadata, or its parent")
    parser.add_argument("--output", required=True, help="destination .safetensors file")
    parser.add_argument("--recipe", choices=sorted(RECIPE_HOPS), required=True)
    parser.add_argument("--trust-dcp", action="store_true", help="acknowledge that this trusted checkpoint's DCP metadata uses pickle")
    parser.add_argument("--exclude-key", action="append", default=[], help="exact reviewed training-only key to exclude; repeat as needed (recorded in metadata)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing destination after successful conversion")
    args = parser.parse_args(argv)
    report = convert_dcp_checkpoint(args.checkpoint, args.output, args.recipe, trust_dcp=args.trust_dcp,
                                    exclude_keys=tuple(args.exclude_key), overwrite=args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
