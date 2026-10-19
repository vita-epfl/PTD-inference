# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-encode MiniMax-H3 image-conditioned requests into the condition cache (the key is ``sha1(prompt.strip())`` as
for text prompts, so sampling only needs the prompt text).

Input: a directory of loose files or WebDataset ``*.tar`` shards; samples are grouped by key (file name up to the first
dot):

  keyframes   <key>.txt + <key>.first.jpg and/or <key>.last.jpg   -> image-to-video (i2va), last-frame (l2va) or
              first-and-last-frame (fl2va); an optional <key>.json {"anchors": "first" | "last" | "first+last"}
  references  <key>.txt + <key>.ref1.jpg, <key>.ref2.jpg, ...      -> reference-to-video (ref2va, up to 9 images;
              sample with the release's "transformer_ref" partition: --transformer-subfolder transformer_ref)

Each record holds what the release builds from these pixels, minus the per-request noise: the Qwen3-VL embedding of
the presentation ("<Picture i>: " + vision rows per image, then the prompt) and the un-noised video-VAE latents
(posterior sample, generator seed 42 per image, stored in fp16).  Canvas policy of the release: keyframes go on the
target canvas (first image stretched, second cover-cropped; the first image's natural canvas must match --canvas unless
--force-canvas); reference images keep their own aspect ratio at a 2048-pixel short edge.  The VAE runs first, then the
text encoder (both do not fit one GPU together).  --write-prompts lists the encoded prompts as a JSON file for sampling.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from ptd.data import prompt_cache_key, prompt_cache_path  # noqa: E402

ANCHOR_SETS = {"first+last": ("first", "last"), "first,last": ("first", "last"), "first": ("first",), "last": ("last",)}
KEYFRAME_TASKS = {("first",): "i2va", ("last",): "l2va", ("first", "last"): "fl2va"}
IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
REFERENCE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
MAX_REFERENCES = 9


class SkipSample(Exception):
    pass


def list_shards(dirs: List[str]) -> List[Tuple[str, str]]:
    shards: List[Tuple[str, str]] = []
    for d in dirs:
        tars = sorted(glob.glob(os.path.join(d, "*.tar")))
        for sub in ("train", "prompts_train"):
            tars += sorted(glob.glob(os.path.join(d, sub, "*.tar")))
        shards += [("tar", t) for t in tars]
        if glob.glob(os.path.join(d, "*.txt")):
            shards.append(("dir", d))
    return shards


def read_shard(kind: str, path: str) -> List[Tuple[str, Dict[str, bytes]]]:
    """All samples of one shard as ``(key, {ext: bytes})`` in file order (samples with a ``.txt`` only)."""
    groups: Dict[str, Dict[str, bytes]] = {}
    order: List[str] = []

    def add(name: str, data: bytes) -> None:
        base = os.path.basename(name)
        if "." not in base:
            return
        key, ext = base.split(".", 1)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][ext.lower()] = data

    if kind == "tar":
        with tarfile.open(path) as tf:
            for m in tf:
                if m.isfile():
                    add(m.name, tf.extractfile(m).read())
    else:
        for f in sorted(glob.glob(os.path.join(path, "*.*"))):
            if os.path.isfile(f):
                with open(f, "rb") as fh:
                    add(f, fh.read())
    return [(k, groups[k]) for k in order if "txt" in groups[k]]


# ----------------------------------------------------------------------------- keyframes
def keyframe_anchors(members: Dict[str, bytes]) -> Tuple[str, ...]:
    meta: Dict[str, Any] = json.loads(members["json"].decode("utf-8")) if "json" in members else {}
    have = {a for a in ("first", "last") if any(f"{a}.{e}" in members for e in IMAGE_EXTS)}
    spec = meta.get("anchors") if isinstance(meta, dict) else None
    if spec:
        if spec not in ANCHOR_SETS:
            raise SkipSample(f"unknown anchors {spec!r}")
        anchors = ANCHOR_SETS[spec]
    elif have:
        anchors = tuple(a for a in ("first", "last") if a in have)
    else:
        raise SkipSample("no keyframe image")
    missing = [a for a in anchors if a not in have]
    if missing:
        raise SkipSample(f"anchors {anchors} but image(s) missing: {missing}")
    return anchors


def image_bytes(members: Dict[str, bytes], name: str) -> bytes:
    for e in IMAGE_EXTS:
        if f"{name}.{e}" in members:
            return members[f"{name}.{e}"]
    raise SkipSample(f"no {name} image")


def prepare_keyframes(members, anchors, canvas_hw, force_canvas):
    """Release canvas policy -> (PIL keyframes on the canvas in packed order, natural canvas of the first one)."""
    from PIL import Image, ImageOps

    from ptd.models.h3_packing import prepare_keyframe_image, resolve_canvas_size

    height, width = canvas_hw
    images, natural = [], None
    for i, a in enumerate(anchors):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes(members, a)))).convert("RGB")
        if i == 0:
            natural = resolve_canvas_size(*img.size)
            if tuple(natural) != (height, width) and not force_canvas:
                raise SkipSample(f"natural canvas {tuple(natural)} != target {(height, width)} (image {img.size}); use --force-canvas")
        images.append(prepare_keyframe_image(img, height, width, stretch=(i == 0)))
    return images, natural


# ----------------------------------------------------------------------------- references
def reference_images(members: Dict[str, bytes]):
    """Reference images in packed order at their own canvas (short edge 2048, sides multiples of 32, LANCZOS)."""
    from PIL import Image, ImageOps

    found: Dict[int, bytes] = {}
    for ext, data in members.items():
        m = re.fullmatch(r"ref(\d+)\.(%s)" % "|".join(IMAGE_EXTS), ext)
        if m:
            found[int(m.group(1))] = data
    if sorted(found) != list(range(1, len(found) + 1)) or len(found) > MAX_REFERENCES:
        raise SkipSample(f"reference images must be ref1..refN with N <= {MAX_REFERENCES}, got {sorted(found)}")
    images, sizes = [], []
    for i in range(1, len(found) + 1):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(found[i]))).convert("RGB")
        w, h = img.size
        if w > 4 * h or h > 4 * w:
            raise SkipSample(f"a reference image must be within 1:4 and 4:1, got {w}x{h}")
        scale = REFERENCE_SHORT_EDGE / min(w, h)
        hh = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        ww = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        if img.size != (ww, hh):
            img = img.resize((ww, hh), Image.Resampling.LANCZOS)
        images.append(img)
        sizes.append((hh, ww))
    return images, sizes


def to_video(images):
    """PIL images -> ``[n, 3, 1, H, W]`` in ``[-1, 1]`` (one image per batch item)."""
    import numpy as np
    import torch

    arr = np.stack([np.asarray(im) for im in images])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)[:, :, None]


def load_video_vae(model_root: str, device: str):
    from ptd.models.h3 import H3VAE

    return H3VAE(model_root, device=device, load_audio=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dirs", nargs="+", required=True, help="directories of loose files or *.tar shards")
    ap.add_argument("--cache-dir", required=True, help="condition cache directory")
    ap.add_argument("--model", default="MiniMaxAI/MiniMax-H3", help="Hugging Face id or local snapshot of MiniMax-H3")
    ap.add_argument("--canvas", default="768x1344", help="HxW target canvas of keyframes (= the video size)")
    ap.add_argument("--force-canvas", action="store_true", help="stretch keyframes whose natural canvas differs")
    ap.add_argument("--vae-seed", type=int, default=42, help="VAE posterior-sample seed per image (release: 42)")
    ap.add_argument("--rank", type=int, default=0, help="this process's share of the shards (with --world)")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-encode existing records")
    ap.add_argument("--write-prompts", default="", help="write the encoded prompts (JSON list) to this file")
    args = ap.parse_args()
    canvas_hw = tuple(int(x) for x in args.canvas.lower().split("x"))

    # ---- enumerate this rank's shards ----
    shards = [s for i, s in enumerate(list_shards(args.input_dirs)) if i % args.world == args.rank]
    todo: List[Dict[str, Any]] = []
    prompts: List[str] = []
    seen = set()
    for shard in shards:
        for key, members in read_shard(*shard):
            prompt = members["txt"].decode("utf-8")
            h = prompt_cache_key(prompt)
            if h in seen:
                continue
            seen.add(h)
            try:
                if any(re.fullmatch(r"ref\d+\.\w+", e) for e in members):
                    kind, anchors = "ref2va", ()
                    reference_images(members)
                else:
                    anchors = keyframe_anchors(members)
                    kind = KEYFRAME_TASKS[anchors]
                    prepare_keyframes(members, anchors, canvas_hw, args.force_canvas)
            except SkipSample as e:
                print(f"[rank {args.rank}] skip {shard[1]}:{key}: {e}", flush=True)
                continue
            prompts.append(prompt)
            if os.path.exists(prompt_cache_path(args.cache_dir, prompt)) and not args.force:
                continue
            todo.append({"shard": shard, "key": key, "prompt": prompt, "sha1": h, "task": kind, "anchors": anchors})
    print(f"[rank {args.rank}/{args.world}] {len(prompts)} samples, {len(todo)} to encode", flush=True)
    if args.write_prompts:
        with open(args.write_prompts, "w") as f:
            json.dump(prompts, f, indent=1)
    if not todo:
        return

    import torch

    from ptd.models.h3 import H3TextEncoder, resolve_model_root

    model_root = resolve_model_root(args.model)
    by_shard: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for rec in todo:
        os.makedirs(os.path.join(args.cache_dir, rec["sha1"][:2]), exist_ok=True)
        by_shard.setdefault(rec["shard"], {})[rec["key"]] = rec

    def images_of(rec, members):
        if rec["task"] == "ref2va":
            return reference_images(members)[0]
        return prepare_keyframes(members, rec["anchors"], canvas_hw, args.force_canvas)[0]

    # ---- pass 1: video VAE -> un-noised latents ----
    vae = load_video_vae(model_root, "cuda" if torch.cuda.is_available() else "cpu")
    latents: Dict[str, Any] = {}
    with torch.no_grad():
        for shard, recs in by_shard.items():
            for key, members in read_shard(*shard):
                rec = recs.get(key)
                if rec is None:
                    continue
                images = images_of(rec, members)
                if rec["task"] == "ref2va":  # one image at a time: their canvases differ
                    latents[rec["sha1"]] = [
                        vae.encode_keyframes(to_video([im]), seed=args.vae_seed)[0].to(torch.float16).contiguous()
                        for im in images
                    ]
                else:
                    lat = vae.encode_keyframes(to_video(images), seed=args.vae_seed)
                    latents[rec["sha1"]] = lat.to(torch.float16).contiguous()  # [n_kf, 24, 1, H/16, W/16]
    del vae
    torch.cuda.empty_cache()

    # ---- pass 2: Qwen3-VL presentation with the images -> records ----
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    enc = H3TextEncoder(model_root, device=dev)
    with torch.no_grad():
        for shard, recs in by_shard.items():
            for key, members in read_shard(*shard):
                rec = recs.get(key)
                if rec is None:
                    continue
                cond = enc.encode(rec["prompt"], images=images_of(rec, members))
                record = {
                    "prompt_embeds": cond["prompt_embeds"][0].to(torch.bfloat16).cpu().contiguous(),
                    "text_token_tags": cond["text_token_tags"].cpu(),
                    "task": rec["task"],
                    "sha1": rec["sha1"],
                    "n_tokens": int(cond["text_token_tags"].numel()),
                    "prompt": rec["prompt"].strip(),
                    "vae_seed": args.vae_seed,
                }
                if rec["task"] == "ref2va":
                    record["ref_latents"] = latents[rec["sha1"]]
                else:
                    record["keyframe_anchors"] = tuple(rec["anchors"])
                    record["cond_latents"] = latents[rec["sha1"]]
                    record["canvas"] = tuple(canvas_hw)
                out = os.path.join(args.cache_dir, rec["sha1"][:2], f"{rec['sha1']}.pt")
                torch.save(record, out + f".tmp{args.rank}")
                os.replace(out + f".tmp{args.rank}", out)
                print(f"[rank {args.rank}] {rec['task']} {key}: {record['n_tokens']} rows", flush=True)


if __name__ == "__main__":
    main()
