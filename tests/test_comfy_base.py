# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab
# SPDX-License-Identifier: Apache-2.0
"""Base conversion checks; optional runtimes enable actual tiny-model parity."""

import pytest
import torch
from torch import nn

from ptd_comfy.base import base_key, convert_base_state_dict


TINY_CONFIG = dict(num_attention_heads=2, attention_head_dim=8, hidden_size=8,
                   num_layers=1, num_refiner_layers=2, ffn_dim=16, in_channels=2,
                   audio_in_channels=3, text_dim=5, freq_dim=8, time_embed_hidden_dim=8,
                   time_embed_dim=4, rope_freq_dim=1, rope_theta=10000.0)


@pytest.fixture
def diffusers_model():
    module = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    torch.manual_seed(13)
    return module.MiniMaxH3Transformer3DModel(**TINY_CONFIG).eval().requires_grad_(False)


@pytest.fixture
def native_runtime():
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.cpu = True
    return pytest.importorskip("comfy.ldm.minimax.model")


@pytest.fixture
def converted_models(diffusers_model, native_runtime):
    native = native_runtime.MiniMaxH3Model(
        hidden_size=8, num_layers=1, token_refiner_num_layers=2,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
        latents_dim=2, audio_latents_dim=3, text_dim=5, timestep_input_dim=8,
        time_embed_hidden_size=8, time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.float32, device="cpu", operations=nn,
    ).eval().requires_grad_(False)
    state = convert_base_state_dict(diffusers_model.state_dict(), dict(diffusers_model.config))
    assert set(state) == set(native.state_dict())
    assert {key: tuple(value.shape) for key, value in state.items()} == {
        key: tuple(value.shape) for key, value in native.state_dict().items()}
    native.load_state_dict(state, strict=True)
    return diffusers_model, native


@pytest.mark.parametrize("source,native", [
    ("proj_in.weight", "video_patch_proj.weight"),
    ("audio_proj_in.bias", "audio_patch_proj.bias"),
    ("context_embedder.weight", "condition_proj.weight"),
    ("time_embedder.linear_1.weight", "time_embedder.proj_in.weight"),
    ("time_embedder.linear_2.bias", "time_embedder.proj_out.bias"),
    ("transformer_blocks.1.attn.norm_q.weight", "blocks.1.attn.q_norm.weight"),
    ("transformer_blocks.1.ff.net.0.proj.weight", "blocks.1.mlp.fc1.weight"),
    ("token_refiner.refiner_blocks.1.ff.net.2.weight", "token_refiner.blocks.1.mlp.fc2.weight"),
    ("norm_out.linear.weight", "final_layer.adaln_proj.linear.weight"),
    ("norm_out.norm.weight", "final_layer.norm.weight"),
    ("proj_out.bias", "final_layer.video_out.bias"),
    ("audio_proj_out.weight", "final_layer.audio_out.weight"),
])
def test_base_key_mapping(source, native):
    assert base_key(source) == native


@pytest.mark.parametrize("key", ["proj_in.unrecognized", "blocks.0.attn.to_q.weight",
    "token_refiner.blocks.0.norm1.weight", "unknown.weight", "transformer_blocks.0.unrecognized.weight"])
def test_base_key_rejects_unknown_or_already_native_keys(key):
    with pytest.raises(ValueError):
        base_key(key)


def test_actual_diffusers_state_matches_native_keys_and_shapes(converted_models):
    source, native = converted_models
    assert len(native.blocks) == len(source.transformer_blocks) == 1
    assert len(native.token_refiner.blocks) == len(source.token_refiner.refiner_blocks) == 2


def test_qkv_order_and_non_qkv_precision_are_preserved(diffusers_model):
    state = diffusers_model.state_dict()
    for family in ("transformer_blocks.0", "token_refiner.refiner_blocks.0", "token_refiner.refiner_blocks.1"):
        for index, part in enumerate("qkv"):
            key = f"{family}.attn.to_{part}.weight"
            state[key] = torch.full_like(state[key], float(index + 1), dtype=torch.bfloat16)
    state["proj_out.weight"] = state["proj_out.weight"].double()
    native = convert_base_state_dict(state, dict(diffusers_model.config))
    for family in ("blocks.0", "token_refiner.blocks.0", "token_refiner.blocks.1"):
        fused = native[f"{family}.attn.qkv_proj.weight"]
        assert fused.dtype == torch.bfloat16
        for index, part in enumerate(fused.chunk(3)):
            assert torch.equal(part, torch.full_like(part, float(index + 1)))
    assert native["final_layer.video_out.weight"].dtype == torch.float64
    assert torch.equal(native["final_layer.video_out.weight"], state["proj_out.weight"])
    assert native["rope.inv_freq"].dtype == torch.float32


def test_swiglu_projection_rows_change_from_up_gate_to_gate_up(diffusers_model):
    state = diffusers_model.state_dict()
    native = convert_base_state_dict(state, dict(diffusers_model.config))
    for source, target in (("transformer_blocks.0", "blocks.0"),
                            ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0")):
        up, gate = state[source + ".ff.net.0.proj.weight"].chunk(2)
        native_gate, native_up = native[target + ".mlp.fc1.weight"].chunk(2)
        assert torch.equal(native_up, up)
        assert torch.equal(native_gate, gate)


@pytest.mark.parametrize("change", ["missing", "unknown", "qkv_dtype", "shape", "integer", "rope"])
def test_invalid_base_checkpoint_is_rejected(diffusers_model, change):
    state = dict(diffusers_model.state_dict())
    config = dict(diffusers_model.config)
    if change == "missing":
        state.pop("proj_out.bias")
    elif change == "unknown":
        state["blocks.0.attn.to_q.weight"] = state["transformer_blocks.0.attn.to_q.weight"]
    elif change == "qkv_dtype":
        state["transformer_blocks.0.attn.to_q.weight"] = state["transformer_blocks.0.attn.to_q.weight"].half()
    elif change == "shape":
        state["proj_out.weight"] = state["proj_out.weight"][:-1]
    elif change == "integer":
        state["proj_out.weight"] = state["proj_out.weight"].long()
    else:
        config["rope_theta"] = float("nan")
    with pytest.raises(ValueError):
        convert_base_state_dict(state, config)


@pytest.mark.parametrize("sigma", [0.02, 0.4, 0.999])
def test_actual_native_forward_matches_diffusers_float32(converted_models, native_runtime, sigma):
    """Compare both streams through real refiner, blocks, norms, and output heads."""
    from ptd.models.h3_packing import (
        build_packed_sequence, build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
    )

    reference, native = converted_models
    generator = torch.Generator().manual_seed(71)
    video = torch.randn(1, 2, 1, 4, 6, generator=generator)
    audio = torch.randn(1, 3, 2, 2, generator=generator)
    context = torch.randn(1, 3, 5, generator=generator)
    tags = torch.ones(3, dtype=torch.long)
    layout = build_packed_sequence(tags, 1, 4, 6, 2, (1, 2, 2))
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32)
    t_video = float(1 - sigma_tensor)
    t_audio = float(1 - native_runtime.time_shift_sigma(sigma_tensor, 12.0, 3.0))
    times, indices = build_row_timesteps(layout, t_video, t_audio, 0.999)
    with torch.inference_mode():
        packed_video, packed_audio = reference(
            hidden_states=patchify_video_latents(video, (1, 2, 2)).unsqueeze(0),
            audio_hidden_states=audio.permute(0, 2, 3, 1).flatten(1, 2),
            encoder_hidden_states=context, timestep=times, timestep_indices=indices,
            token_tags=layout.token_tags, position_ids=layout.position_ids,
            video_indices=layout.video_indices, audio_indices=layout.audio_indices,
            text_indices=layout.text_indices, return_dict=False,
        )
        expected_video = -unpatchify_video_tokens(packed_video.flatten(0, 1), 1, 4, 6, 2, (1, 2, 2))
        expected_audio = -packed_audio.unflatten(1, (2, 2)).permute(0, 3, 1, 2)
        actual_video, actual_audio = native(
            [video, audio], sigma_tensor.reshape(1) * 1000, context, transformer_options={},
        )
    torch.testing.assert_close(actual_video, expected_video, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_audio, expected_audio, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hop", [0, 1, 2, 3])
def test_real_tiny_ptd_experts_match_diffusers(converted_models, dtype, hop):
    """Real adapted refiner/DiT/head forwards catch packing and precision mistakes."""
    from ptd.modules import HopLoRALinear, HopRef, PTDOutputProj
    from ptd.models.h3_packing import (
        build_packed_sequence, build_row_timesteps, patchify_video_latents, unpatchify_video_tokens,
    )
    from ptd.recipes import get_recipe
    from ptd.schedule import Clock
    from ptd_comfy.checkpoint import validate_adapter_tensors
    from ptd_comfy.patcher import PTDController
    from ptd_comfy.sampling import native_audio_shift

    reference, native = converted_models
    routing = HopRef(4)
    suffixes = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    for name, layer in list(reference.named_modules()):
        if isinstance(layer, nn.Linear) and name.endswith(suffixes):
            parent, _, attr = name.rpartition(".")
            setattr(reference.get_submodule(parent), attr, HopLoRALinear(layer, [128] * 4, routing))
    reference.proj_out = PTDOutputProj(reference.proj_out, routing)
    reference.audio_proj_out = PTDOutputProj(reference.audio_proj_out, routing)
    generator = torch.Generator().manual_seed(111)
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .025)
    tensors = {"transformer." + name: value.clone() for name, value in reference.state_dict().items() if "lora_" in name}
    pairs = validate_adapter_tensors(tensors, "h3_4step", expected_blocks=1, expected_refiner_blocks=2)
    controller = PTDController(native, tensors, pairs, "h3_4step", 4)
    # Match original FSDP mixed precision: fp32 projection/time/head islands.
    reference.to(dtype=dtype)
    native.to(dtype=dtype)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope"):
        reference.get_submodule(name).float()
    for name in ("video_patch_proj", "audio_patch_proj", "time_embedder", "final_layer.video_out",
                 "final_layer.audio_out", "rope"):
        native.get_submodule(name).float()
    reference.requires_grad_(False)
    video = torch.randn(1, 2, 1, 4, 6, generator=generator).to(dtype)
    audio = torch.randn(1, 3, 2, 2, generator=generator).to(dtype)
    context = torch.randn(1, 3, 5, generator=generator).to(dtype)
    layout = build_packed_sequence(torch.ones(3, dtype=torch.long), 1, 4, 6, 2, (1, 2, 2))
    sigma = get_recipe("h3_4step").grid[hop]
    time = torch.tensor(sigma, dtype=torch.float64)
    clock = Clock(.25)
    times, indices = build_row_timesteps(layout, 1.0 - sigma, float(1 - clock.sigma(time)), .999)
    routing.h = hop
    with torch.inference_mode():
        packed_video, packed_audio = reference(
            hidden_states=patchify_video_latents(video, (1, 2, 2)).unsqueeze(0),
            audio_hidden_states=audio.permute(0, 2, 3, 1).flatten(1, 2),
            encoder_hidden_states=context, timestep=times, timestep_indices=indices,
            token_tags=layout.token_tags, position_ids=layout.position_ids,
            video_indices=layout.video_indices, audio_indices=layout.audio_indices,
            text_indices=layout.text_indices, return_dict=False,
        )
        expected_video = -unpatchify_video_tokens(packed_video.flatten(0, 1), 1, 4, 6, 2, (1, 2, 2))
        expected_audio = -packed_audio.unflatten(1, (2, 2)).permute(0, 3, 1, 2)
        expected_audio = expected_audio * clock.sigma_prime(time).to(expected_audio.dtype)
        with controller.activate(hop, sigma), controller.forward_hooks(controller.current_selection()):
            actual_video, actual_audio = native(
                [video, audio], time.reshape(1) * 1000, context,
                transformer_options={"minimax_h3_sigma_shift_video": 1.0,
                                     "minimax_h3_sigma_shift_audio": native_audio_shift(sigma)},
            )
    tolerance = 2e-6 if dtype == torch.float32 else .02
    torch.testing.assert_close(actual_video, expected_video.to(dtype), rtol=2e-5, atol=tolerance)
    torch.testing.assert_close(actual_audio, expected_audio.to(dtype), rtol=2e-5, atol=tolerance)


def test_bfloat16_fifty_layer_stack_matches_diffusers_exactly(native_runtime):
    """Real Comfy operations must preserve each source BF16 rounding boundary.

    Fifty layers and the released head/rotary widths catch errors that one
    tiny block with an absolute tolerance can hide. Attention is fixed to
    PyTorch SDPA on both sides so this checks the adapter's arithmetic contract
    independently of optional attention backends. This CPU regression does
    not claim identical rounding across CUDA GEMM/attention implementations.
    """
    source_module = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    comfy_ops = pytest.importorskip("comfy.ops")
    attention_module = pytest.importorskip("comfy.ldm.modules.attention")
    from ptd_comfy.patcher import PTDController

    torch.manual_seed(123)
    config = dict(TINY_CONFIG, hidden_size=32, num_layers=50,
                  attention_head_dim=128, ffn_dim=64, time_embed_dim=8,
                  rope_freq_dim=16)
    reference = source_module.MiniMaxH3Transformer3DModel(**config).eval().requires_grad_(False)
    # Nonunit learned norm weights exercise multiplication before the cast.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "norm" in name and name.endswith("weight") and parameter.ndim == 1:
                parameter.uniform_(.5, 1.5)
    native = native_runtime.MiniMaxH3Model(
        hidden_size=32, num_layers=50, token_refiner_num_layers=2,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=64,
        latents_dim=2, audio_latents_dim=3, text_dim=5, timestep_input_dim=8,
        time_embed_hidden_size=8, time_embed_dim=8, rope_inv_freq_len=16,
        dtype=torch.float32, device="cpu", operations=comfy_ops.disable_weight_init,
    ).eval().requires_grad_(False)
    native.load_state_dict(convert_base_state_dict(reference.state_dict(), dict(reference.config)), strict=True)
    reference.bfloat16()
    native.bfloat16()
    reference.time_embedder.float()
    native.time_embedder.float()
    reference.rope.float()
    native.rope.float()
    controller = PTDController(native, {}, (), "h3_4step", 4)
    x = torch.randn(1, 48, 32).bfloat16()
    positions = torch.randn(48, 3) * 100
    from ptd_comfy.sampling import audio_sigma
    sigma = .857
    times = torch.tensor([1.0 - sigma, 1.0 - audio_sigma(sigma), .999])
    indices = torch.tensor([0] * 16 + [4] * 16 + [8] * 16)
    segments = [(0, 16, 0), (16, 32, 4), (32, 48, 8)]

    def pytorch_attention(original, *args, **kwargs):
        return attention_module.attention_pytorch(*args, **kwargs)

    options = {"optimized_attention_override": pytorch_attention}
    with torch.inference_mode():
        expected = x.clone()
        source_time = reference.time_embedder(reference.time_proj(times))
        rotary = reference.rope(positions)
        expected_layers = []
        for block in reference.transformer_blocks:
            expected = block(expected, source_time, indices, rotary)
            expected_layers.append(expected.clone())
        with controller.activate(2, sigma), controller.forward_hooks(controller.current_selection()):
            native_time = native.time_embedder(times).bfloat16()
            table = native_runtime.rope_rotation_table(native.rope_freqs(positions, "cpu"), torch.bfloat16)
            actual = x[0].clone()
            for index, block in enumerate(native.blocks):
                actual = block(actual, native_time, segments, table, transformer_options=options)
                torch.testing.assert_close(actual, expected_layers[index][0], rtol=0, atol=0,
                                           msg=f"BF16 rounding diverged at block {index}")
