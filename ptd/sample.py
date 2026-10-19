# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample a PTD student of MiniMax-H3 (launch with ``torchrun``; see ``scripts/sample.sh``).

The student is sharded over all GPUs; GPU ``r`` renders every prompt with seed ``--seed + r`` (one noise draw per
seed, shared by the prompts) and writes ``<output>/<prompt index>_seed<seed>.mp4`` with audio.  Conditions come from
the condition cache (text prompts: ``scripts/encode_prompts.py``; image-conditioned requests:
``scripts/encode_image_conditions.py``; reference-image requests also need ``--transformer-subfolder transformer_ref``).
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from ptd import checkpoint
from ptd.data import load_prompts
from ptd.dist import barrier, init_distributed, log, rank, set_cuda_backend, set_seed
from ptd.media import save_mp4
from ptd.models import build_student, default_model
from ptd.models.h3 import H3_AUDIO_SAMPLE_RATE, ConditionCache, H3VAE, resolve_model_root
from ptd.models.h3_packing import MINIMAX_H3_FPS
from ptd.recipes import RECIPES, get_recipe
from ptd.sampler import hop_grid, input_shape, sample
from ptd.utils import randn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--ckpt", required=True, help="checkpoint directory (e.g. weights/ptd_h3_6step)")
    ap.add_argument("--prompts", nargs="+", required=True, help="prompt files (see ptd/data.py)")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--cond-cache", required=True, help="condition cache directory")
    ap.add_argument("--model", default=None, help="Hugging Face id or local snapshot of MiniMax-H3")
    ap.add_argument("--transformer-subfolder", default="transformer", help="'transformer_ref' for reference images")
    ap.add_argument("--seed", type=int, default=0, help="seed of GPU 0 (GPU r uses seed + r)")
    ap.add_argument("--save-latents", action="store_true", help="also write the final latents (<name>.latents.pt)")
    args = ap.parse_args(argv)

    recipe = get_recipe(args.recipe)
    model = args.model or default_model()
    init_distributed()
    set_cuda_backend()
    device = torch.device("cuda", torch.cuda.current_device())
    prompts = load_prompts(args.prompts)

    set_seed(args.seed)
    net = build_student(recipe, model, transformer_subfolder=args.transformer_subfolder)
    checkpoint.load_model(net, args.ckpt)
    log(f"[sample] {recipe.name}: loaded {args.ckpt}; {len(prompts)} prompts")
    cache = ConditionCache(args.cond_cache)
    vae = H3VAE(resolve_model_root(model), device=device)
    grid = hop_grid(recipe, device)

    seed = set_seed(args.seed + rank())
    noise = randn(input_shape(recipe), 1, device, torch.bfloat16)
    os.makedirs(args.output, exist_ok=True)
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        cond = cache.condition(prompt, device=device)
        latents = sample(net, noise, cond, grid, fp32_carry=recipe.fp32_carry)
        name = os.path.join(args.output, f"{i:04d}_seed{seed}")
        if args.save_latents:
            torch.save({k: v.float().cpu() for k, v in latents.items()}, name + ".latents.pt")
        media = vae.decode(latents)
        save_mp4(name + ".mp4", media["video"][0], MINIMAX_H3_FPS, audio=media["audio"][0], sample_rate=H3_AUDIO_SAMPLE_RATE)
        print(f"[sample] rank {rank()}: {name}.mp4 ({time.time() - t0:.1f} s)", flush=True)
    barrier()
    log("[sample] done")


if __name__ == "__main__":
    main()
