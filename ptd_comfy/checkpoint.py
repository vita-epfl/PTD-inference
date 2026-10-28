# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab (PTD adaptations and supporting code). All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated PTD adapter files and the mapping to ComfyUI's native H3 modules.

The interchange file keeps the original A/B factors, including separate Q/K/V
factors and the full-rank output-head factors. Its scale is exactly one. This is
not a conventional ComfyUI LoRA file: a PTD sampler selects one expert per hop.

Only conversion reads PyTorch DCP metadata (pickle). Runtime loading uses
safetensors and never unpickles a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

FORMAT = "comfyui-native-h3"
SCHEMA = "1"
BASE_MODEL = "MiniMaxAI/MiniMax-H3"
BASE_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
RECIPE_HOPS = {"h3_4step": 4, "h3_6step": 6, "h3_4step_sde": 4, "h3_6step_sde": 6}
# The source H3 implementation normally recreates this non-persistent buffer.
# Preserve it if an exporter included it; the native loader must apply it.
BUFFER_TARGETS = {
    "transformer.rope.inv_freq": "rope.inv_freq",
    "transformer.proj_out.orig_weight": "final_layer.video_out.weight",
    "transformer.proj_out.orig_bias": "final_layer.video_out.bias",
    "transformer.audio_proj_out.orig_weight": "final_layer.audio_out.weight",
    "transformer.audio_proj_out.orig_bias": "final_layer.audio_out.bias",
}
_LAYER_PARTS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "ff.net.0.proj", "ff.net.2",
)
_KEY = re.compile(r"^(?P<module>transformer\..+)\.lora_(?P<factor>[AB])\.(?P<hop>0|[1-9][0-9]*)\.weight$")
_BLOCK = re.compile(r"^transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.(.+)$")


@dataclass(frozen=True)
class AdapterTarget:
    """Native module name; qkv_index selects one third of a fused projection."""

    module: str
    qkv_index: int | None = None
    dense_head: bool = False

    @property
    def swap_halves(self) -> bool:
        """Native SwiGLU expects [gate, up]; Diffusers stores [up, gate]."""
        return self.module.endswith(".mlp.fc1")


@dataclass(frozen=True)
class AdapterPair:
    target: AdapterTarget
    hop: int
    a_key: str
    b_key: str


def map_adapter_key(key: str) -> tuple[AdapterTarget, int, str]:
    """Map one canonical Diffusers PTD factor key without inspecting tensors.

Q/K/V factors remain independent: their deltas belong in consecutive thirds of
    ``qkv_proj``'s output. A target with ``swap_halves`` needs its B factor's output
    rows (or the computed delta) permuted from [up, gate] to [gate, up]. No
    transpose, alpha/rank scaling, or factor fusion occurs.
Unknown prefixes, projections and factor names fail explicitly.
    """
    match = _KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"Not a supported PTD adapter key: {key}")
    module = match["module"]
    heads = {"transformer.proj_out.main": "final_layer.video_out",
             "transformer.audio_proj_out.main": "final_layer.audio_out"}
    if module in heads:
        target = AdapterTarget(heads[module], dense_head=True)
    else:
        block = _BLOCK.fullmatch(module)
        if block is None:
            raise ValueError(f"Unknown PTD adapter module: {module}")
        prefix = "blocks" if block[1] == "transformer_blocks" else "token_refiner.blocks"
        base = f"{prefix}.{block[2]}"
        part = block[3]
        qkv = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
        if part in qkv:
            target = AdapterTarget(f"{base}.attn.qkv_proj", qkv[part])
        else:
            names = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
            if part not in names:
                raise ValueError(f"Unknown PTD adapter projection: {module}")
            target = AdapterTarget(f"{base}.{names[part]}")
    return target, int(match["hop"]), match["factor"]


def _expected_modules(blocks: int, refiners: int) -> set[str]:
    modules = {f"transformer.transformer_blocks.{i}.{part}"
               for i in range(blocks) for part in _LAYER_PARTS}
    modules.update(f"transformer.token_refiner.refiner_blocks.{i}.{part}"
                   for i in range(refiners) for part in _LAYER_PARTS)
    modules.update(("transformer.proj_out.main", "transformer.audio_proj_out.main"))
    return modules


def training_only_keys(recipe: str) -> frozenset[str]:
    """Exact curvature-scaffold keys present in the released training checkpoint.

These heads are absent from ``ptd.modules.PTDOutputProj`` and never participate
in inference. Their exclusion is always listed in the conversion report and
safetensors metadata; other unrecognized entries are errors.
    """
    if recipe not in RECIPE_HOPS:
        raise ValueError(f"Unknown PTD recipe: {recipe}")
    return frozenset(f"transformer.{head}.fan_chead.heads.{hop}.{part}"
                     for head in ("proj_out", "audio_proj_out")
                     for hop in range(RECIPE_HOPS[recipe]) for part in ("weight", "bias"))


def _validate_shapes(
    shapes: Mapping[str, tuple[int, ...]], recipe: str, *, expected_blocks: int = 50,
    expected_refiner_blocks: int = 2, expected_rank: int = 128,
) -> tuple[AdapterPair, ...]:
    if recipe not in RECIPE_HOPS:
        raise ValueError(f"Unknown PTD recipe {recipe!r}; choose {tuple(RECIPE_HOPS)}")
    if expected_blocks < 0 or expected_refiner_blocks < 0 or expected_rank < 1:
        raise ValueError("Invalid expected model dimensions")
    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for key, shape in shapes.items():
        if key in BUFFER_TARGETS:
            ndim = 2 if key.endswith("orig_weight") else 1
            if len(shape) != ndim or min(shape) < 1:
                raise ValueError(f"Invalid inference buffer shape: {key}: {shape}")
            continue
        _, hop, factor = map_adapter_key(key)
        if hop >= RECIPE_HOPS[recipe]:
            raise ValueError(f"Adapter hop {hop} exceeds {recipe}: {key}")
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f"Adapter must be a nonempty matrix: {key}: {shape}")
        module = _KEY.fullmatch(key)["module"]
        grouped.setdefault((module, hop), {})[factor] = key
    expected = {(m, h) for m in _expected_modules(expected_blocks, expected_refiner_blocks)
                for h in range(RECIPE_HOPS[recipe])}
    actual = set(grouped)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"Incomplete or unexpected PTD topology: {len(missing)} missing pairs {missing[:4]}; "
                         f"{len(extra)} unexpected pairs {extra[:4]}")
    pairs = []
    module_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for (module, hop), factors in sorted(grouped.items()):
        if set(factors) != {"A", "B"}:
            raise ValueError(f"Incomplete A/B pair: {module}, hop {hop}")
        a_key, b_key = factors["A"], factors["B"]
        a, b = shapes[a_key], shapes[b_key]
        target, _, _ = map_adapter_key(a_key)
        if target.swap_halves and b[0] % 2:
            raise ValueError(f"SwiGLU output width must be even: {module}")
        if a[0] != b[1]:
            raise ValueError(f"Incompatible A/B ranks: {a_key}: {a}, {b_key}: {b}")
        rank = b[0] if target.dense_head else expected_rank
        if a[0] != rank:
            raise ValueError(f"Expected {'output-sized' if target.dense_head else expected_rank} rank for {module}; got {a[0]}")
        previous = module_shapes.setdefault(module, (a, b))
        if previous != (a, b):
            raise ValueError(f"Adapter dimensions differ between hops: {module}")
        pairs.append(AdapterPair(target, hop, a_key, b_key))
    # All three independent projections must fit the same native fused QKV layer.
    for pair in pairs:
        if pair.target.qkv_index == 0:
            qmodule = _KEY.fullmatch(pair.a_key)["module"]
            for name in ("to_k", "to_v"):
                other = qmodule.removesuffix("to_q") + name
                if module_shapes[other] != module_shapes[qmodule]:
                    raise ValueError(f"Q/K/V adapter shapes differ: {qmodule}, {other}")
    head_buffers = {k for k in BUFFER_TARGETS if ".orig_" in k}
    present = head_buffers & set(shapes)
    if present and present != head_buffers:
        raise ValueError(f"Incomplete original output-head buffers: missing {sorted(head_buffers - present)}")
    for head in ("proj_out", "audio_proj_out"):
        weight, bias = f"transformer.{head}.orig_weight", f"transformer.{head}.orig_bias"
        if weight in shapes:
            a, b = module_shapes[f"transformer.{head}.main"]
            if shapes[weight] != (b[0], a[1]) or shapes[bias] != (b[0],):
                raise ValueError(f"Original output-head buffer dimensions disagree with adapters: {head}")
    return tuple(pairs)


def validate_adapter_tensors(
    tensors: Mapping[str, Any], recipe: str, *, expected_blocks: int = 50,
    expected_refiner_blocks: int = 2, expected_rank: int = 128,
) -> tuple[AdapterPair, ...]:
    """Validate complete 4/6-hop adapters; return deterministic A/B pair records."""
    import torch

    shapes = {}
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or not value.is_floating_point():
            raise ValueError(f"Expected a dense floating-point tensor: {key}")
        if value.device.type == "meta":
            raise ValueError(f"A materialized tensor is required: {key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite values in {key}")
        shapes[key] = tuple(value.shape)
    pairs = _validate_shapes(shapes, recipe, expected_blocks=expected_blocks,
                             expected_refiner_blocks=expected_refiner_blocks, expected_rank=expected_rank)
    for pair in pairs:
        if tensors[pair.a_key].dtype != tensors[pair.b_key].dtype:
            raise ValueError(f"A/B dtypes differ: {pair.a_key}, {pair.b_key}")
    return pairs


def adapter_metadata(recipe: str, *, excluded_keys: tuple[str, ...] = ()) -> dict[str, str]:
    """Required metadata for the public MiniMax-H3 PTD adapter format."""
    if recipe not in RECIPE_HOPS:
        raise ValueError(f"Unknown PTD recipe: {recipe}")
    return {"ptd_format": FORMAT, "ptd_schema": SCHEMA, "ptd_recipe": recipe,
            "ptd_hops": str(RECIPE_HOPS[recipe]), "ptd_lora_scale": "1.0",
            "ptd_base_model": BASE_MODEL, "ptd_base_revision": BASE_REVISION,
            "ptd_mapping": "diffusers-h3-to-native-h3-v1",
            "ptd_source_format": "torch.distributed.checkpoint",
            "ptd_excluded_keys": json.dumps(sorted(excluded_keys), separators=(",", ":"))}


def load_adapter_file(path: str | Path, recipe: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a validated adapter safetensors file on CPU; never read pickle."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        stored_recipe = metadata.get("ptd_recipe")
        if stored_recipe not in RECIPE_HOPS:
            raise ValueError("Missing or unsupported ptd_recipe metadata")
        if recipe is not None and recipe != stored_recipe:
            raise ValueError(f"Requested {recipe}, but adapter file is {stored_recipe}")
        expected = adapter_metadata(stored_recipe)
        for key in ("ptd_format", "ptd_schema", "ptd_hops", "ptd_lora_scale", "ptd_mapping",
                    "ptd_base_model", "ptd_base_revision"):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"Unsupported adapter metadata {key}={metadata.get(key)!r}; expected {expected[key]!r}")
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
        _validate_shapes(shapes, stored_recipe)
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    validate_adapter_tensors(tensors, stored_recipe)
    return tensors, metadata


def convert_dcp_checkpoint(
    source: str | Path, destination: str | Path, recipe: str, *, trust_dcp: bool = False,
    exclude_keys: tuple[str, ...] = (), overwrite: bool = False,
) -> dict[str, Any]:
    """Select and convert PTD tensors from a trusted local DCP model directory.

Unknown entries stop conversion. Exact, explicitly supplied ``exclude_keys``
are recorded in the output metadata and report; they cannot exclude adapters or
required buffers. No optimizer or arbitrary Python objects are deserialized by
the load plan. DCP's metadata itself uses pickle, hence ``trust_dcp`` is required.
    """
    if not trust_dcp:
        raise ValueError("DCP metadata uses pickle. Convert only a trusted checkpoint and set trust_dcp=True.")
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from safetensors.torch import save_file

    if recipe not in RECIPE_HOPS:
        raise ValueError(f"Unknown PTD recipe: {recipe}")
    source, destination = Path(source), Path(destination)
    if not (source / ".metadata").is_file():
        source = source / "model"
    metadata_path = source / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No DCP .metadata in {source}")
    if destination.suffix != ".safetensors":
        raise ValueError("Destination must have a .safetensors suffix")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    reader = dcp.FileSystemReader(str(source))
    metadata = reader.read_metadata()
    entries = metadata.state_dict_metadata
    excluded = set(exclude_keys)
    if len(excluded) != len(exclude_keys):
        raise ValueError("Duplicate exclude_keys entries")
    missing_exclusions = excluded - set(entries)
    if missing_exclusions:
        raise ValueError(f"Excluded keys are absent from the checkpoint: {sorted(missing_exclusions)}")
    selected = {}
    unknown = []
    known_training = training_only_keys(recipe) & set(entries)
    if known_training and known_training != training_only_keys(recipe):
        raise ValueError("Incomplete training curvature scaffold; review the checkpoint before converting")
    excluded.update(known_training)
    for key, entry in entries.items():
        try:
            map_adapter_key(key)
            supported = True
        except ValueError:
            supported = key in BUFFER_TARGETS
        if supported:
            if key in excluded:
                raise ValueError(f"Cannot exclude an inference tensor: {key}")
            if not isinstance(entry, TensorStorageMetadata):
                raise ValueError(f"Expected DCP tensor metadata: {key}")
            if not entry.properties.dtype.is_floating_point:
                raise ValueError(f"Expected floating-point checkpoint tensor: {key}")
            selected[key] = entry
        elif key not in excluded:
            unknown.append(key)
    if unknown:
        raise ValueError(f"Unknown DCP entries ({len(unknown)}); none were discarded: {sorted(unknown)[:12]}. "
                         "Review training-only entries and explicitly list each with --exclude-key.")
    _validate_shapes({key: tuple(entry.size) for key, entry in selected.items()}, recipe)
    # Reject shard references outside the supplied checkpoint before tensor reads.
    for item in (metadata.storage_data or {}).values():
        relative = getattr(item, "relative_path", None)
        if relative is not None:
            shard = (source / relative).resolve()
            if not shard.is_relative_to(source.resolve()):
                raise ValueError("DCP metadata references a shard outside the checkpoint directory")
    tensors = {key: torch.empty(tuple(entry.size), dtype=entry.properties.dtype, device="cpu")
               for key, entry in selected.items()}
    dcp.load(tensors, storage_reader=reader, planner=DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
    pairs = validate_adapter_tensors(tensors, recipe)
    output_metadata = adapter_metadata(recipe, excluded_keys=tuple(excluded))
    output_metadata["ptd_source_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    output_metadata["ptd_buffer_keys"] = json.dumps(sorted(set(tensors) & set(BUFFER_TARGETS)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=output_metadata)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # A hard link publishes atomically and fails if another writer won.
            os.link(temporary, destination)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"recipe": recipe, "hops": RECIPE_HOPS[recipe], "tensor_count": len(tensors),
            "pair_count": len(pairs), "excluded_keys": sorted(excluded),
            "buffer_keys": sorted(set(tensors) & set(BUFFER_TARGETS)), "output": str(destination),
            "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values())}
