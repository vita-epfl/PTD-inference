# SPDX-FileCopyrightText: Copyright (c) 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Based on MiniMax-H3 inference code, with minor PTD-specific adaptations by EPFL VITA Lab.
# See NOTICE for upstream attribution.

"""MiniMax-H3 (33B joint video + audio) as a PTD student.

H3 denoises one packed 1-D sequence holding the text conditioning, optional image-conditioning rows and the target
audio and video rows, with full self-attention over all of it (``h3_packing`` holds the layout).  The transformer
reads ``t_h3 = 1 - sigma`` per row (``1`` clean) and predicts the data-ward velocity ``x0 - eps``; ``forward`` returns
``flow = dx/dt`` on the shared clock of ``ptd.schedule`` (video ``sigma = t``, audio the relative ``3 / 12`` warp), so
the released shift-12 video / shift-3 audio sampler is the shift-12 grid of the shared clock.

Latents: video ``[B, 24, F, H, W]`` (16x spatial, ``17 n + 5`` frames -> ``5 n + 2`` latent frames), audio
``[B, 32, 2, L]`` (stereo, 40 latents per second).  The request is packed into one sequence, so a batch shares one
layout and one timestep; sampling runs one video per GPU.

Student: one LoRA expert per hop (rank 128) on every transformer block's ``attn.to_q/k/v``, ``attn.to_out.0``,
``ff.net.0.proj`` and ``ff.net.2`` and on the same layers of the two token-refiner blocks; the two output projections
become ``PTDOutputProj`` (per-hop LoRA for the mean velocity).

Precision: the release keeps the patch projections, the timestep MLP and the output heads in float32 and the block
stack in bfloat16; ``shard`` gives those modules their own float32 FSDP groups (parameters stored in float32, blocks
computed in bfloat16).

Conditioning comes from a cache of pre-encoded requests (``ConditionCache``; ``scripts/encode_prompts.py`` and
``scripts/encode_image_conditions.py`` write it), so the 62 GiB Qwen3-VL text encoder never sits next to the model.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn
from diffusers.models.transformers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

from ptd.models.h3_packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_TIMESTEP,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from ptd.data import prompt_cache_key, prompt_cache_path
from ptd.modules import HopLoRALinear, HopRef, PTDOutputProj
from ptd.schedule import Clock, expand_like
from ptd.utils import to

# The release samples video on a shift-12 clock and audio on a shift-3 clock; video is the shared clock here.
H3_VIDEO_SHIFT = 12.0
H3_AUDIO_RELATIVE_SHIFT = 3.0 / 12.0
H3_AUDIO_SAMPLE_RATE = 32000

# LoRA targets (name suffixes): attention q/k/v/o and the SwiGLU up/down projections.
H3_LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")

# Modules the release keeps in float32 (name components of the transformer's parameters and buffers).
H3_FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope")

# Task of a keyframe record, by its anchors in packed order.
KEYFRAME_TASKS = {("first",): "i2va", ("last",): "l2va", ("first", "last"): "fl2va"}

H3_SUBFOLDERS = ("modular_model_index.json", "transformer/*", "transformer_ref/*", "text_encoder/*", "tokenizer/*",
                 "processor/*", "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*")


def resolve_model_root(model: str) -> str:
    """A local directory with the release's subfolders, for a local path or a Hugging Face id."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=list(H3_SUBFOLDERS))


def h3_input_shape(frames: int) -> Dict[str, List[int]]:
    """Latent shapes of a ``frames``-frame 768x1344 clip (``frames = 17 n + 5``)."""
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"MiniMax-H3 takes 17n+5 frames (e.g. 124, 243), got {frames}")
    return {"video": [24, (frames - 5) // 17 * 5 + 2, 48, 84], "audio": [32, MINIMAX_H3_AUDIO_CHANNELS, audio_latent_num_frames(frames)]}


# ----------------------------------------------------------------------------------------------------- condition cache
def record_task(rec: Mapping[str, Any]) -> str:
    """``t2va`` (text only), ``ref2va`` (image references) or the keyframe task (``i2va`` / ``l2va`` / ``fl2va``)."""
    if rec.get("ref_latents") is not None:
        return "ref2va"
    anchors = rec.get("keyframe_anchors")
    if not anchors or rec.get("cond_latents") is None:
        return "t2va"
    return KEYFRAME_TASKS.get(tuple(anchors), "kf")


def keyframe_rows_from_record(rec: Mapping[str, Any], patch_size, latent_channels: int, generator=None, device=None):
    """Cached un-noised keyframe latents -> the release's noise-augmented conditioning rows.

    ``rows = t_kf * patchify(cond_latents) + (1 - t_kf) * noise`` with ``t_kf = 0.999`` (float32, one noise draw per
    call, on the CPU by default).  Returns ``([R, C * prod(patch)], anchors)``.
    """
    cond_latents = rec["cond_latents"]
    anchors = tuple(rec["keyframe_anchors"])
    if anchors not in KEYFRAME_TASKS:
        raise ValueError(f"keyframe_anchors must be ('first',), ('last',) or ('first', 'last'), got {anchors!r}")
    if cond_latents.ndim != 5 or cond_latents.shape[0] != len(anchors) or cond_latents.shape[1] != latent_channels:
        raise ValueError(f"cond_latents must be [n_kf={len(anchors)}, C={latent_channels}, 1, H, W], got {tuple(cond_latents.shape)}")
    if cond_latents.shape[2] != 1:
        raise ValueError(f"A MiniMax-H3 keyframe latent holds one frame, got {cond_latents.shape[2]}.")
    latents = cond_latents.to(device=device or torch.device("cpu"), dtype=torch.float32)
    rows = patchify_video_latents(latents, patch_size)
    num_latent_frames, latent_height, latent_width = latents.shape[2:]
    noise = keyframe_condition_noise(((num_latent_frames, latent_height, latent_width),) * len(anchors), patch_size,
                                     latent_channels, generator=generator, device=rows.device)
    keyframe_timestep = rows.new_tensor(MINIMAX_H3_KEYFRAME_TIMESTEP)
    return keyframe_timestep * rows + (1.0 - keyframe_timestep) * noise, anchors


def reference_rows_from_record(rec: Mapping[str, Any], patch_size, latent_channels: int, generator=None, device=None):
    """Cached un-noised reference-image latents -> noise-augmented conditioning rows (same recipe as the keyframes).

    ``rec["ref_latents"]`` is a list of ``[C, 1, h_i, w_i]`` latents in packed order.  Returns
    ``([R, C * prod(patch)], geometries)``.
    """
    lats = rec["ref_latents"]
    if isinstance(lats, torch.Tensor):
        lats = list(lats)
    if not lats:
        raise ValueError("A ref2va record must carry at least one reference latent.")
    rows, geoms = [], []
    for lat in lats:
        lat = lat.to(device=device or torch.device("cpu"), dtype=torch.float32)
        if lat.ndim != 4 or lat.shape[0] != latent_channels or lat.shape[1] != 1:
            raise ValueError(f"ref_latents entries must be [C={latent_channels}, 1, h, w], got {tuple(lat.shape)}")
        rows.append(patchify_video_latents(lat[None], patch_size))
        geoms.append((1, int(lat.shape[2]), int(lat.shape[3])))
    rows = torch.cat(rows)
    noise = keyframe_condition_noise(tuple(geoms), patch_size, latent_channels, generator=generator, device=rows.device)
    t_ref = rows.new_tensor(MINIMAX_H3_KEYFRAME_TIMESTEP)
    return t_ref * rows + (1.0 - t_ref) * noise, tuple(geoms)


class ConditionCache:
    """Pre-encoded MiniMax-H3 requests, ``<cache_dir>/<sha1[:2]>/<sha1>.pt`` with ``sha1 = sha1(prompt.strip())``.

    A record holds the Qwen3-VL presentation of the request (``prompt_embeds [T, 5120]`` + ``text_token_tags [T]``;
    for image-conditioned requests including the vision rows) and, for keyframe / reference requests, the un-noised
    VAE latents of the images.
    """

    _MEMO_MAX = 8

    def __init__(self, cache_dir: str, patch_size=(1, 2, 2), latent_channels: int = 24):
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"condition cache directory not found: {cache_dir}")
        self.cache_dir = cache_dir
        self.patch_size = tuple(patch_size)
        self.latent_channels = int(latent_channels)
        self._memo: Dict[str, Dict[str, Any]] = {}

    def load(self, prompt: str) -> Dict[str, Any]:
        key = prompt_cache_key(prompt)
        rec = self._memo.get(key)
        if rec is None:
            path = prompt_cache_path(self.cache_dir, prompt)
            if not os.path.exists(path):
                raise FileNotFoundError(f"no cached condition {key} at {path} for prompt {prompt[:80]!r}...; "
                                        "run scripts/encode_prompts.py (or scripts/encode_image_conditions.py) first.")
            rec = torch.load(path, map_location="cpu", weights_only=False)
            if len(self._memo) >= self._MEMO_MAX:
                self._memo.clear()
            self._memo[key] = rec
        return rec

    def condition(self, prompt: str, device=None, dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
        """The network condition of one request.  Image-conditioned requests get fresh noise for their conditioning
        rows from the global CPU generator, as the release does per request."""
        rec = self.load(prompt)
        cond: Dict[str, Any] = {
            "prompt_embeds": rec["prompt_embeds"][None].to(dtype=dtype),
            "text_token_tags": rec["text_token_tags"].to(torch.long),
        }
        task = record_task(rec)
        if task == "ref2va":
            rows, geoms = reference_rows_from_record(rec, self.patch_size, self.latent_channels)
            cond.update(cond_video_rows=rows[None], keyframe_anchors=(), reference_geometries=geoms)
        elif task != "t2va":
            rows, anchors = keyframe_rows_from_record(rec, self.patch_size, self.latent_channels)
            cond.update(cond_video_rows=rows[None], keyframe_anchors=anchors)
        return to(cond, device=device, dtype=dtype)


# -------------------------------------------------------------------------------------------------------- the network
def _wrap_linears(root: nn.Module, targets: Sequence[str], wrap) -> int:
    """Replace every ``nn.Linear`` under *root* whose qualified name ends with one of *targets*."""
    n = 0
    for name, mod in list(root.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name == t or name.endswith("." + t) for t in targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        if attr.isdigit():
            parent[int(attr)] = wrap(mod)
        else:
            setattr(parent, attr, wrap(mod))
        n += 1
    return n


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, A, L]`` -> channel-major rows ``[B, A * L, C]``."""
    return latents.permute(0, 2, 3, 1).flatten(1, 2)


def _unpack_audio_latents(rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
    """Channel-major rows ``[B, A * L, C]`` -> ``[B, C, A, L]``."""
    return rows.unflatten(1, (MINIMAX_H3_AUDIO_CHANNELS, num_audio_latents)).permute(0, 3, 1, 2)


def _in_meta_context() -> bool:
    return torch.empty(0).device.type == "meta"


class MiniMaxH3PTD(nn.Module):
    """MiniMax-H3 transformer + PTD adapters (``hops`` hop experts)."""

    ADAPTERS = ("lora_",)

    def __init__(
        self,
        model_root: str,
        hops: int,
        ranks: Optional[Sequence[int]] = None,
        refiner_lora: bool = True,
        transformer_subfolder: str = "transformer",
    ):
        super().__init__()
        path = os.path.join(model_root, transformer_subfolder)
        if _in_meta_context():  # non-zero ranks: structure only, weights arrive from rank 0 after sharding
            self.transformer = MiniMaxH3Transformer3DModel.from_config(MiniMaxH3Transformer3DModel.load_config(path)).float()
        else:
            self.transformer = MiniMaxH3Transformer3DModel.from_pretrained(path, torch_dtype=torch.float32)
        tr = self.transformer
        self.clocks = {"video": Clock(1.0), "audio": Clock(H3_AUDIO_RELATIVE_SHIFT)}
        self.patch_size = tuple(tr.config.patch_size)
        self.video_latent_channels = tr.config.in_channels
        self._layout_cache: dict = {}

        self.N_HOPS = int(hops)
        self._hop_ref = HopRef(self.N_HOPS)
        self.ranks = [128] * self.N_HOPS if ranks is None else [int(r) for r in ranks]
        if len(self.ranks) != self.N_HOPS:
            raise ValueError(f"ranks must list one rank per hop ({self.N_HOPS}), got {self.ranks}")

        def wrap(lin: nn.Linear) -> HopLoRALinear:
            return HopLoRALinear(lin, self.ranks, self._hop_ref)

        n_wrapped = _wrap_linears(tr.transformer_blocks, H3_LORA_TARGETS, wrap)
        if n_wrapped != 6 * len(tr.transformer_blocks):
            raise RuntimeError(f"expected {6 * len(tr.transformer_blocks)} block targets, wrapped {n_wrapped}")
        if refiner_lora:
            n_wrapped += _wrap_linears(tr.token_refiner, H3_LORA_TARGETS, wrap)
        for name in ("proj_out", "audio_proj_out"):
            setattr(tr, name, PTDOutputProj(getattr(tr, name), self._hop_ref))
        self.freeze()

    # ------------------------------------------------------------------------------------------------- parameters
    def freeze(self) -> None:
        """Flag the adapters as the network's only trainable parameters: ``ptd.checkpoint.load_model`` loads exactly
        those (and the buffers) from a PTD checkpoint; everything else is the frozen release."""
        for n, p in self.transformer.named_parameters():
            p.requires_grad_(any(k in n for k in self.ADAPTERS))

    @staticmethod
    def keep_fp32(name: str) -> bool:
        """Whether a tensor of this module (qualified name) stays float32 when the network is cast down."""
        return name.startswith("transformer.") and any(m in name.split(".")[1:] for m in H3_FP32_MODULES)

    def shard(self, mesh, mp_policy, fp32_mp_policy) -> None:
        """FSDP2: one group per transformer block, one float32 group per float32 module, one root group."""
        from torch.distributed.fsdp import fully_shard

        tr = self.transformer
        for block in tr.transformer_blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp_policy)
        for module in (tr.proj_in, tr.audio_proj_in, tr.time_embedder, tr.proj_out, tr.audio_proj_out):
            fully_shard(module, mesh=mesh, mp_policy=fp32_mp_policy)
        fully_shard(tr, mesh=mesh, mp_policy=mp_policy)
        self.freeze()  # sharding replaces the parameters

    def reset_buffers(self) -> None:
        """Recompute the non-persistent rotary ``inv_freq`` after meta-device materialisation."""
        rope_theta = float(self.transformer.config.rope_theta)
        for module in self.transformer.modules():
            if isinstance(module, MiniMaxH3RotaryPosEmbed):
                freq_dim = module.rope_freq_dim
                inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32) / (2 * freq_dim)))
                device = module.inv_freq.device if module.inv_freq is not None else None
                module.register_buffer("inv_freq", inv_freq.to(device) if device is not None else inv_freq, persistent=False)

    def flow_convention_scale(self, t: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Per modality, ``sigma'(t)`` of its clock (``None`` for video, whose clock is the shared one)."""
        return {m: None if c.shift == 1.0 else c.sigma_prime(t) for m, c in self.clocks.items()}

    # ---------------------------------------------------------------------------------------------------- forward
    def _layout(self, text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
                keyframe_anchors, device, reference_geometries=()):
        """The packed layout (cached by structure: it never depends on latent values)."""
        key = (int(text_token_tags.numel()), tuple(text_token_tags.tolist()), num_latent_frames, latent_height,
               latent_width, num_audio_latents, keyframe_anchors, tuple(reference_geometries), str(device))
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode(False), torch.no_grad():
            if reference_geometries:
                if keyframe_anchors:
                    raise ValueError("A request carries either keyframe anchors or image references, not both.")
                layout = build_ref2va_packed_sequence(text_token_tags.cpu(), tuple(tuple(int(v) for v in g) for g in reference_geometries),
                                                      num_latent_frames, latent_height, latent_width, num_audio_latents, self.patch_size)
            else:
                layout = build_packed_sequence(text_token_tags.cpu(), num_latent_frames, latent_height, latent_width,
                                               num_audio_latents, self.patch_size, keyframe_anchors)
            layout = copy.copy(layout)
            for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
                setattr(layout, name, getattr(layout, name).to(device))
        if len(self._layout_cache) > 8:
            self._layout_cache.clear()
        self._layout_cache[key] = layout
        return layout

    @staticmethod
    def _scalar_timestep(sigma: torch.Tensor) -> float:
        """The request's H3 timestep ``1 - sigma`` (the batch must share one time)."""
        flat = sigma.flatten()
        if flat.numel() > 1 and not bool((flat == flat[0]).all()):
            raise ValueError(f"MiniMax-H3 packs one timestep per row of one sequence; t must be uniform over the batch, got {flat[:8].tolist()}")
        return 1.0 - float(flat[0])

    def forward(self, x_t: Dict[str, torch.Tensor], t: torch.Tensor, condition: Mapping[str, Any],
                hop: int) -> Dict[str, torch.Tensor]:
        """The mean velocity ``h`` of hop ``hop`` per modality (``flow = dx/dt`` on the shared clock) at the shared time
        ``t`` (``[B]``, ``0`` clean).  Outputs have the dtype of ``x_t``."""
        self._hop_ref.h = int(hop)

        video_latents, audio_latents = x_t["video"], x_t["audio"]
        batch = video_latents.shape[0]
        device = video_latents.device
        _, _, num_latent_frames, latent_height, latent_width = video_latents.shape
        num_audio_latents = audio_latents.shape[-1]

        text_token_tags = condition["text_token_tags"].to(device)
        cond_video_rows = condition.get("cond_video_rows")
        keyframe_anchors = tuple(condition.get("keyframe_anchors", ()))
        reference_geometries = tuple(tuple(int(v) for v in g) for g in condition.get("reference_geometries", ()))
        layout = self._layout(text_token_tags, num_latent_frames, latent_height, latent_width, num_audio_latents,
                              keyframe_anchors, device, reference_geometries=reference_geometries)

        # Rows: conditioning first, then the generated ones.
        video_rows = patchify_video_latents(video_latents, self.patch_size).unflatten(0, (batch, -1))
        if layout.num_condition_video_rows:
            if cond_video_rows is None:
                raise ValueError(f"the layout reserves {layout.num_condition_video_rows} conditioning rows but the condition has no 'cond_video_rows'")
            video_rows = torch.cat([cond_video_rows.to(video_rows), video_rows], dim=1)
        audio_rows = _pack_audio_latents(audio_latents)

        # Row timesteps: H3 time is 1 - sigma per modality; conditioning rows stay at the keyframe time (0.999).
        video_timestep = self._scalar_timestep(self.clocks["video"].sigma(t))
        audio_timestep = self._scalar_timestep(self.clocks["audio"].sigma(t))
        unique_timesteps, timestep_indices = build_row_timesteps(layout, video_timestep, audio_timestep,
                                                                 max(video_timestep, MINIMAX_H3_KEYFRAME_TIMESTEP))

        video_packed, audio_packed = self.transformer(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=condition["prompt_embeds"],
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
        )

        video_packed = video_packed[:, layout.num_condition_video_rows:]
        channels = video_packed.shape[-1] // math.prod(self.patch_size)
        video_out = unpatchify_video_tokens(video_packed.flatten(0, 1), num_latent_frames, latent_height, latent_width,
                                            channels, self.patch_size)
        audio_out = _unpack_audio_latents(audio_packed, num_audio_latents)

        # eps - x0 per head (the negated data-ward velocity) -> dx/dt on the shared clock, in the latents' dtype.
        out = {"video": -video_out, "audio": -audio_out}
        for m, scale in self.flow_convention_scale(t).items():
            if scale is not None:
                out[m] = out[m] * expand_like(scale.to(out[m].dtype), out[m])
        return {"video": out["video"].to(video_latents.dtype), "audio": out["audio"].to(audio_latents.dtype)}

    # ------------------------------------------------------------------------------------------------- sampling helpers
    def latents(self, noise: Dict[str, torch.Tensor], t_init: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {m: self.clocks[m].latents(noise[m], t_init) for m in noise}

    def flow_to_x0(self, x: Dict[str, torch.Tensor], v: Dict[str, torch.Tensor], t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per modality, the clean endpoint ``x - (sigma / sigma')(t) v`` of the flow ``v`` at ``x`` (time ``t``)."""
        return {m: self.clocks[m].flow_to_x0(x[m], v[m], t) for m in x}

    def sigma(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per modality, the noise level ``sigma(t)`` of its clock."""
        return {m: c.sigma(t) for m, c in self.clocks.items()}


# --------------------------------------------------------------------------------------------------- VAE and encoder
class H3VAE:
    """The release's video VAE (ImageNet-normalised RGB in [0, 1], spatial tiling on) and audio VAE (32 kHz stereo,
    decoded as two mono items), both float32.  Pixels are exchanged in [-1, 1]."""

    def __init__(self, model_root: str, device=None, load_audio: bool = True):
        from diffusers.models.autoencoders import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.video = AutoencoderKLMiniMaxH3.from_pretrained(os.path.join(model_root, "vae"))
        self.video.eval().requires_grad_(False)
        self.video.enable_tiling(tile_sample_min_height=None, tile_sample_min_width=None)
        self.video.to(device=device, dtype=torch.float32)
        self.audio = None
        if load_audio:
            self.audio = AutoencoderKLMiniMaxH3Audio.from_pretrained(os.path.join(model_root, "audio_vae"))
            self.audio.eval().requires_grad_(False)
            self.audio.to(device=device, dtype=torch.float32)

    @property
    def device(self):
        return self.video.device

    def _norm(self, vae, ndim: int):
        shape = (1, -1) + (1,) * (ndim - 2)
        mean = torch.tensor(vae.config.latents_mean, device=self.device).view(*shape)
        std = torch.tensor(vae.config.latents_std, device=self.device).view(*shape)
        return mean, std

    @torch.no_grad()
    def decode(self, latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``{"video": [B, 3, T, H, W] in [-1, 1], "audio": [B, 2, samples]}``."""
        dev = self.device
        mean, std = self._norm(self.video, 5)
        lat = latents["video"].to(device=dev, dtype=torch.float32) * std + mean
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            video = self.video.decode(lat, return_dict=False)[0]
        pmean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=dev).view(1, -1, 1, 1, 1)
        pstd = torch.tensor(MINIMAX_H3_PIXEL_STD, device=dev).view(1, -1, 1, 1, 1)
        out = {"video": (video.float() * pstd + pmean).clamp(0, 1).mul(2.0).sub(1.0)}
        if self.audio is not None and "audio" in latents:
            amean, astd = self._norm(self.audio, 4)
            alat = latents["audio"].to(device=dev, dtype=torch.float32) * astd + amean
            b = alat.shape[0]
            wave = self.audio.decode(alat.permute(0, 2, 1, 3).flatten(0, 1), return_dict=False)[0]
            out["audio"] = wave.float().reshape(b, MINIMAX_H3_AUDIO_CHANNELS, -1)
        return out

    @torch.no_grad()
    def encode_keyframes(self, video: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Keyframes ``[n, 3, 1, H, W]`` in [-1, 1] -> normalised latents ``[n, 24, 1, H/16, W/16]`` (posterior samples;
        with ``seed`` a fresh generator per image, as the release's pipeline does), via the release's fp16 round trip."""
        dev = self.device
        mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        pixels = (video.to(device=dev, dtype=torch.float32).mul(0.5).add(0.5) - mean) / std
        if pixels.shape[2] != 1:
            raise ValueError(f"A MiniMax-H3 keyframe must contain one frame, got {pixels.shape[2]}.")
        out = []
        for keyframe in pixels.split(1):
            distr = self.video.encode(keyframe, return_dict=False)[0]
            generator = None if seed is None else torch.Generator().manual_seed(seed)
            lat = distr.sample(generator=generator).to(torch.float16).float().cpu()
            lmean = torch.tensor(self.video.config.latents_mean, dtype=lat.dtype).view(1, -1, 1, 1, 1)
            lstd = torch.tensor(self.video.config.latents_std, dtype=lat.dtype).view(1, -1, 1, 1, 1)
            out.append((lat - lmean) / lstd)
        return torch.cat(out)


class H3TextEncoder:
    """The Qwen3-VL conditioner, read at its 50th decoder layer, with the release's request presentation: for each
    image ``"<Picture i>: "`` + its vision rows (tagged as video), then the prompt verbatim (tagged as text)."""

    def __init__(self, model_root: str, device=None, dtype: torch.dtype = torch.bfloat16):
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(model_root, "tokenizer"))
        self.processor = Qwen3VLProcessor.from_pretrained(os.path.join(model_root, "processor"))
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(os.path.join(model_root, "text_encoder"), torch_dtype=dtype)
        self.model.eval().requires_grad_(False)
        if self.model.config.text_config.num_hidden_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError("the text encoder has fewer decoder layers than the layer MiniMax-H3 reads")
        self.model.to(device)

    def _vision_ids(self, pad_token: str, num_tokens: int):
        ids = ([self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
               + [self.tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
               + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        return ids, [MINIMAX_H3_VIDEO_TAG] * len(ids)

    def _text_ids(self, value: str):
        ids = self.tokenizer(value, add_special_tokens=False)["input_ids"]
        return ids, [MINIMAX_H3_TEXT_TAG] * len(ids)

    def _presentation(self, prompt: str, images: list):
        token_ids: List[int] = []
        token_tags: List[int] = []
        vision: dict = {}
        if images:
            processed = self.processor.image_processor(images=images, return_tensors="pt")
            vision = {"pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"]}
            merge = self.processor.image_processor.merge_size ** 2
            for i in range(len(images)):
                n_img = int(vision["image_grid_thw"][i].prod()) // merge
                for ids, tags in (self._text_ids(f"<Picture {i + 1}>: "), self._vision_ids("<|image_pad|>", n_img)):
                    token_ids += ids
                    token_tags += tags
        ids, tags = self._text_ids(prompt)
        return token_ids + ids, token_tags + tags, vision

    @torch.no_grad()
    def encode(self, prompt: str, images: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """``{"prompt_embeds": [1, T, 5120] bf16, "text_token_tags": [T]}`` of one request (images: PIL, packed order)."""
        device = next(self.model.parameters()).device
        token_ids, token_tags, vision = self._presentation(prompt.strip(), images or [])
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        mm_token_type_ids = torch.tensor(self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
        vision = {k: v.to(device, self.model.dtype) if k.startswith("pixel_values") else v.to(device) for k, v in vision.items()}
        out = self.model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), mm_token_type_ids=mm_token_type_ids,
                               use_cache=False, output_hidden_states=True, **vision)
        return {"prompt_embeds": out.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(torch.bfloat16),
                "text_token_tags": torch.tensor(token_tags, dtype=torch.long, device=device)}
