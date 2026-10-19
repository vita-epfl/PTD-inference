# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-encode MiniMax-H3 text prompts into the condition cache (Qwen3-VL hidden states, as the release computes them).

Every unique prompt becomes ``<cache_dir>/<sha1[:2]>/<sha1>.pt`` with ``sha1 = sha1(prompt.strip())``, holding
``{"prompt_embeds": bf16 [T, 5120], "text_token_tags": int64 [T], "sha1", "n_tokens", "prompt"}``.  Sampling reads
this cache, so the 62 GiB text encoder never sits next to the 33B model.  Prompts already in the cache are skipped.
Run one process per GPU, e.g. on 8 GPUs:

    for i in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$i python scripts/encode_prompts.py --prompts prompts.json \\
        --cache-dir cond_cache --rank $i --world 8 & done; wait
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from ptd.data import load_prompts, prompt_cache_key, prompt_cache_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", required=True, help="prompt files / directories (see ptd/data.py)")
    ap.add_argument("--cache-dir", required=True, help="condition cache directory")
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-H3", help="Hugging Face id or local snapshot of MiniMax-H3")
    ap.add_argument("--rank", type=int, default=0, help="this process's share of the prompts (with --world)")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--list-only", action="store_true", help="only report how many prompts still need encoding")
    args = ap.parse_args()

    uniq, seen = [], set()
    for prompt in load_prompts(args.prompts):
        key = prompt_cache_key(prompt)
        if key not in seen:
            seen.add(key)
            uniq.append(prompt)
    mine = uniq[args.rank::args.world]
    todo = [p for p in mine if not os.path.exists(prompt_cache_path(args.cache_dir, p))]
    print(f"[rank {args.rank}/{args.world}] unique {len(uniq)} mine {len(mine)} todo {len(todo)}", flush=True)
    if args.list_only or not todo:
        return

    import torch

    from ptd.models.h3 import H3TextEncoder, resolve_model_root

    enc = H3TextEncoder(resolve_model_root(args.model), device=torch.device("cuda", 0))
    t0 = time.time()
    for n, prompt in enumerate(todo, 1):
        cond = enc.encode(prompt)
        rec = {
            "prompt_embeds": cond["prompt_embeds"][0].to(torch.bfloat16).cpu().contiguous(),
            "text_token_tags": cond["text_token_tags"].cpu(),
            "sha1": prompt_cache_key(prompt),
            "n_tokens": int(cond["text_token_tags"].numel()),
            "prompt": prompt.strip(),
        }
        out = prompt_cache_path(args.cache_dir, prompt)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save(rec, out + f".tmp{args.rank}")
        os.replace(out + f".tmp{args.rank}", out)
        if n % 200 == 0 or n == len(todo):
            print(f"[rank {args.rank}] {n}/{len(todo)} encoded ({n / (time.time() - t0):.2f} prompts/s)", flush=True)


if __name__ == "__main__":
    main()
