# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Small CPU fixtures cover the complete released adapter topology, without weights."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from ptd_comfy.checkpoint import (
    AdapterTarget, BUFFER_TARGETS, RECIPE_HOPS, adapter_metadata, convert_dcp_checkpoint,
    load_adapter_file, map_adapter_key, training_only_keys, validate_adapter_tensors,
)


def adapter_fixture(recipe, *, include_training=False):
    """Actual key topology with small input/output widths and the released rank."""
    modules = []
    for family, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
        for i in range(count):
            prefix = f"transformer.{family}.{i}"
            modules += [(prefix + ".attn.to_" + part, 2, 2, 128) for part in "qkv"]
            modules += [(prefix + ".attn.to_out.0", 2, 2, 128),
                        (prefix + ".ff.net.0.proj", 6, 2, 128), (prefix + ".ff.net.2", 2, 3, 128)]
    modules += [("transformer.proj_out.main", 2, 2, 2), ("transformer.audio_proj_out.main", 3, 2, 3)]
    generator = torch.Generator().manual_seed(37)
    result = {}
    for module, out_features, in_features, rank in modules:
        for hop in range(RECIPE_HOPS[recipe]):
            result[f"{module}.lora_A.{hop}.weight"] = torch.randn(rank, in_features, generator=generator)
            result[f"{module}.lora_B.{hop}.weight"] = torch.randn(out_features, rank, generator=generator)
    for name, width in (("proj_out", 2), ("audio_proj_out", 3)):
        result[f"transformer.{name}.orig_weight"] = torch.randn(width, 2, generator=generator)
        result[f"transformer.{name}.orig_bias"] = torch.randn(width, generator=generator)
    if include_training:
        for name in training_only_keys(recipe):
            result[name] = torch.ones(1)
    return result


@pytest.mark.parametrize("source,target,index,dense", [
    ("transformer.transformer_blocks.49.attn.to_q", "blocks.49.attn.qkv_proj", 0, False),
    ("transformer.transformer_blocks.49.attn.to_k", "blocks.49.attn.qkv_proj", 1, False),
    ("transformer.transformer_blocks.49.attn.to_v", "blocks.49.attn.qkv_proj", 2, False),
    ("transformer.token_refiner.refiner_blocks.1.attn.to_v", "token_refiner.blocks.1.attn.qkv_proj", 2, False),
    ("transformer.transformer_blocks.2.attn.to_out.0", "blocks.2.attn.out_proj", None, False),
    ("transformer.transformer_blocks.2.ff.net.0.proj", "blocks.2.mlp.fc1", None, False),
    ("transformer.transformer_blocks.2.ff.net.2", "blocks.2.mlp.fc2", None, False),
    ("transformer.proj_out.main", "final_layer.video_out", None, True),
    ("transformer.audio_proj_out.main", "final_layer.audio_out", None, True),
])
def test_native_factor_mapping(source, target, index, dense):
    mapped = map_adapter_key(source + ".lora_B.3.weight")
    assert mapped == (AdapterTarget(target, index, dense), 3, "B")
    assert mapped[0].swap_halves == source.endswith("ff.net.0.proj")


@pytest.mark.parametrize("key", ["model.transformer.proj_out.main.lora_A.0.weight",
    "transformer.transformer_blocks.0.attn.unknown.lora_A.0.weight",
    "transformer.transformer_blocks.0.attn.to_q.lora_C.0.weight",
    "transformer.proj_out.main.lora_A.-1.weight", "transformer.proj_out.main.lora_A.01.weight"])
def test_unknown_adapter_mapping_is_rejected(key):
    with pytest.raises(ValueError):
        map_adapter_key(key)


@pytest.mark.parametrize("recipe,pairs", [("h3_4step", 1256), ("h3_6step", 1884)])
def test_dcp_roundtrip_keeps_every_inference_tensor_exactly(tmp_path, recipe, pairs):
    state = adapter_fixture(recipe, include_training=True)
    source, output = tmp_path / "source", tmp_path / "adapters.safetensors"
    dcp.save(state, checkpoint_id=source, no_dist=True)
    report = convert_dcp_checkpoint(source, output, recipe, trust_dcp=True)
    actual, metadata = load_adapter_file(output, recipe)
    assert report["pair_count"] == pairs
    assert report["excluded_keys"] == sorted(training_only_keys(recipe))
    assert json.loads(metadata["ptd_excluded_keys"]) == report["excluded_keys"]
    assert set(actual) == set(state) - training_only_keys(recipe)
    assert all(torch.equal(actual[key], state[key]) and actual[key].dtype == state[key].dtype for key in actual)
    assert set(report["buffer_keys"]) == set(BUFFER_TARGETS) - {"transformer.rope.inv_freq"}
    assert metadata["ptd_lora_scale"] == "1.0"
    with pytest.raises(ValueError, match="trusted"):
        convert_dcp_checkpoint(source, tmp_path / "untrusted.safetensors", recipe)
    with pytest.raises(FileExistsError):
        convert_dcp_checkpoint(source, output, recipe, trust_dcp=True)
    with pytest.raises(ValueError, match="Requested"):
        load_adapter_file(output, "h3_6step" if recipe == "h3_4step" else "h3_4step")


@pytest.mark.parametrize("change", ["missing_b", "missing_hop", "extra_hop", "unknown", "rank", "dtype", "nan", "buffer"])
def test_invalid_adapter_tensors_are_rejected(change):
    state = adapter_fixture("h3_4step")
    a = "transformer.transformer_blocks.0.attn.to_q.lora_A.0.weight"
    b = a.replace("lora_A", "lora_B")
    if change == "missing_b":
        state.pop(b)
    elif change == "missing_hop":
        state = {key: value for key, value in state.items() if ".3.weight" not in key}
    elif change == "extra_hop":
        state[a.replace(".0.weight", ".4.weight")] = state[a]
    elif change == "unknown":
        state["transformer.unrecognized.weight"] = torch.ones(1)
    elif change == "rank":
        state[a] = state[a][:-1]
    elif change == "dtype":
        state[a] = state[a].half()
    elif change == "nan":
        state[a][0, 0] = float("nan")
    else:
        state.pop("transformer.proj_out.orig_bias")
    with pytest.raises(ValueError):
        validate_adapter_tensors(state, "h3_4step")


@pytest.mark.parametrize("field,value", [("ptd_schema", "999"), ("ptd_lora_scale", "0.5"),
    ("ptd_recipe", "other"), ("ptd_hops", "6"), ("ptd_base_revision", "wrong")])
def test_unrecognized_safetensors_metadata_is_rejected(tmp_path, field, value):
    metadata = adapter_metadata("h3_4step")
    metadata[field] = value
    path = tmp_path / "invalid.safetensors"
    save_file(adapter_fixture("h3_4step"), str(path), metadata=metadata)
    with pytest.raises(ValueError):
        load_adapter_file(path)


def test_unknown_training_entry_requires_explicit_recorded_exclusion(tmp_path):
    state = adapter_fixture("h3_4step")
    state["reviewed_training_scalar"] = torch.ones(1)
    source, output = tmp_path / "source", tmp_path / "adapters.safetensors"
    dcp.save(state, checkpoint_id=source, no_dist=True)
    with pytest.raises(ValueError, match="none were discarded"):
        convert_dcp_checkpoint(source, output, "h3_4step", trust_dcp=True)
    assert not output.exists()
    report = convert_dcp_checkpoint(source, output, "h3_4step", trust_dcp=True,
                                    exclude_keys=("reviewed_training_scalar",))
    assert report["excluded_keys"] == ["reviewed_training_scalar"]


def test_cannot_explicitly_exclude_inference_factors(tmp_path):
    state = adapter_fixture("h3_4step")
    dcp.save(state, checkpoint_id=tmp_path / "source", no_dist=True)
    with pytest.raises(ValueError, match="Cannot exclude"):
        convert_dcp_checkpoint(tmp_path / "source", tmp_path / "out.safetensors", "h3_4step",
                                trust_dcp=True, exclude_keys=(next(iter(state)),))
