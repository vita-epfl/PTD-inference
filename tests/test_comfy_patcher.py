# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import copy
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ptd_comfy.checkpoint import AdapterPair, AdapterTarget
from ptd_comfy.patcher import PTDController, get_ptd_controller, patch_ptd_model
from ptd_comfy.sampling import audio_sigma, audio_sigma_prime, native_audio_shift


class NativeMLP(nn.Module):
    """Exercise the native fc2 fast-path contract without importing ComfyUI."""
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        if getattr(self.fc2, "comfy_force_forward", False):
            return self.fc2(x)
        return F.linear(x, self.fc2.weight)


class NativeToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_shift_video, self.sigma_shift_audio = 12.0, 3.0
        self.condition_proj = nn.Linear(3, 4, bias=False)
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = nn.Module()
        self.blocks[0].attn.qkv_proj = nn.Linear(4, 12, bias=False)
        self.blocks[0].mlp = NativeMLP()
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.final_layer = nn.Module()
        self.final_layer.video_out = nn.Linear(4, 2)
        self.final_layer.audio_out = nn.Linear(4, 2)
        self.refinements = 0

    def preprocess_text_embeds(self, x):
        return self.refine(x)

    def refine(self, x):
        self.refinements += 1
        return self.token_refiner.blocks[0](self.condition_proj(x))

    def forward(self, x, t=None, context=None, transformer_options=None, **kwargs):
        h = self.refine(context) if context is not None and context.shape[-1] == 3 else x[0]
        q, k, v = self.blocks[0].attn.qkv_proj(h).chunk(3, dim=-1)
        h = self.blocks[0].mlp(q + k + v)
        return [-self.final_layer.video_out(h), -self.final_layer.audio_out(h)]


class PatcherToy:
    def __init__(self, native):
        self.model = nn.Module()
        self.model.diffusion_model = native
        self.model.model_sampling = SimpleNamespace(shift=12.0, audio_shift=3.0)
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}

    def get_model_object(self, name):
        obj = self.model
        for part in name.split("."):
            obj = getattr(obj, part)
        return self.object_patches.get(name, obj)

    def clone(self):
        other = copy.copy(self)
        other.model_options = dict(self.model_options)
        other.object_patches = dict(self.object_patches)
        return other

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


def bank(model, *, names=None, offset=0.0):
    names = names or [("blocks.0.attn.qkv_proj", i) for i in range(3)] + [
        ("blocks.0.mlp.fc2", None), ("token_refiner.blocks.0", None),
        ("final_layer.video_out", None), ("final_layer.audio_out", None)]
    tensors, pairs = {}, []
    for name, qkv in names:
        layer = model.get_submodule(name)
        for hop in range(4):
            key = f"{name}.{qkv}.{hop}"
            tensors[key + ".A"] = torch.full((2, layer.in_features), 0.1 + offset)
            width = layer.out_features // (3 if qkv is not None else 1)
            tensors[key + ".B"] = torch.full((width, 2), (hop + 1) * 0.2)
            pairs.append(AdapterPair(AdapterTarget(name, qkv), hop, key + ".A", key + ".B"))
    return tensors, pairs


def controller(native, **kwargs):
    tensors, pairs = bank(native, **kwargs)
    return PTDController(native, tensors, pairs, "h3_4step", 4)


def test_qkv_and_fc2_low_rank_additions_leave_base_and_flags_intact():
    torch.manual_seed(1)
    native = NativeToy()
    c = controller(native)
    x = torch.randn(5, 4)
    layer = native.blocks[0].attn.qkv_proj
    before = layer(x)
    mlp_before = native.blocks[0].mlp(x)
    identities = {name: id(p) for name, p in native.named_parameters()}
    values = {name: p.clone() for name, p in native.named_parameters()}
    with c.activate(1, 0.947), c.forward_hooks(c.current_selection()):
        qkv = layer(x)
        delta = F.linear(F.linear(x, torch.full((2, 4), 0.1)), torch.full((4, 2), 0.4))
        torch.testing.assert_close(qkv, before + torch.cat([delta] * 3, dim=-1))
        torch.testing.assert_close(native.blocks[0].mlp(x), mlp_before + delta)
        assert native.blocks[0].mlp.fc2.comfy_force_forward
    assert not hasattr(native.blocks[0].mlp.fc2, "comfy_force_forward")
    torch.testing.assert_close(layer(x), before)
    torch.testing.assert_close(native.blocks[0].mlp(x), mlp_before)
    for name, p in native.named_parameters():
        assert id(p) == identities[name]
        torch.testing.assert_close(p, values[name], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in native.modules())


def test_clone_raw_refiner_audio_clock_and_previous_wrapper(monkeypatch):
    torch.manual_seed(2)
    native = NativeToy()
    original = PatcherToy(native)
    tensors, pairs = bank(native)
    monkeypatch.setattr("ptd_comfy.checkpoint.validate_adapter_tensors", lambda *a, **k: pairs)
    previous_calls = []

    def previous(fn, args):
        previous_calls.append(args)
        return fn(args["input"], args["timestep"], **args["c"])

    original.model_options["model_function_wrapper"] = previous
    patched = patch_ptd_model(original, tensors, {"ptd_recipe": "h3_4step"})
    c = get_ptd_controller(patched)
    assert patched.model is original.model
    assert original.model_options["model_function_wrapper"] is previous
    assert "ptd_native_controller" not in original.model_options
    assert not original.object_patches
    sampling = patched.object_patches["model_sampling"]
    assert sampling is not original.model.model_sampling
    assert sampling.audio_shift == sampling.shift == 12.0
    assert original.model.model_sampling.audio_shift == 3.0
    raw = torch.randn(1, 3)
    assert patched.object_patches["diffusion_model.preprocess_text_embeds"](raw) is raw
    with pytest.raises(ValueError, match="raw"):
        patched.object_patches["diffusion_model.preprocess_text_embeds"](torch.randn(1, 4))
    args = {"input": [torch.randn(1, 4), torch.randn(1, 4)], "timestep": torch.tensor([0.947]),
            "c": {"c_crossattn": raw, "transformer_options": {"existing": 123},
                  "minimax_payload": {"audio_scale": 4.0}}, "cond_or_uncond": [0]}
    calls = []

    def apply(x, sigma, **conditions):
        calls.append(conditions)
        return native(x, sigma, context=conditions["c_crossattn"],
                      transformer_options=conditions["transformer_options"])

    start = native.refinements
    wrapper = patched.model_options["model_function_wrapper"]
    with c.activate(1, 0.947):
        actual = wrapper(apply, args)
    assert native.refinements == start + 1
    assert previous_calls and calls
    assert calls[0]["transformer_options"]["minimax_h3_sigma_shift_audio"] == native_audio_shift(0.947)
    assert calls[0]["transformer_options"]["minimax_h3_sigma_shift_video"] == 1.0
    assert calls[0]["minimax_payload"]["audio_scale"] == 1.0
    assert args["c"]["transformer_options"] == {"existing": 123}
    assert args["c"]["minimax_payload"] == {"audio_scale": 4.0}
    assert len(actual) == 2
    with pytest.raises(RuntimeError, match="no PTD hop"):
        wrapper(apply, args)
    with c.activate(0, 0.999), pytest.raises(ValueError, match="sigma"):
        wrapper(apply, args)


@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16])
def test_audio_derivative_scales_adapted_head_before_latent_cast(head_dtype):
    class NativeCastToy(NativeToy):
        def forward(self, x):
            return [-self.final_layer.video_out(x[0]).to(torch.bfloat16),
                    -self.final_layer.audio_out(x[1]).to(torch.bfloat16)]

    torch.manual_seed(23)
    native = NativeCastToy().to(head_dtype)
    c = controller(native, names=[("final_layer.audio_out", None)])
    x = [torch.randn(64, 4).to(head_dtype)] * 2
    video_before = -native.final_layer.video_out(x[0]).to(torch.bfloat16)
    raw_audio = native.final_layer.audio_out(x[1]) + F.linear(
        F.linear(x[1], torch.full((2, 4), 0.1, dtype=head_dtype)),
        torch.full((2, 2), 0.6, dtype=head_dtype))
    factor = torch.tensor(audio_sigma_prime(0.857), dtype=head_dtype)
    expected = (-raw_audio * factor).to(torch.bfloat16)
    if head_dtype == torch.float32:
        late_scale = -raw_audio.to(torch.bfloat16) * factor.to(torch.bfloat16)
        assert not torch.equal(expected, late_scale)
    with c.activate(2, 0.857), c.forward_hooks(c.current_selection()):
        actual = native(x)
    torch.testing.assert_close(actual[0], video_before, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)


def test_fp32_stored_experts_use_active_linear_compute_dtype():
    torch.manual_seed(17)
    native = NativeToy().to(torch.bfloat16)
    layer = native.blocks[0].mlp.fc2
    tensors, pairs = bank(native, names=[("blocks.0.mlp.fc2", None)])
    for key in tensors:
        tensors[key] = torch.randn_like(tensors[key])
    c = PTDController(native, tensors, pairs, "h3_4step", 4)
    x = torch.randn(64, 4).to(torch.bfloat16)
    a = tensors["blocks.0.mlp.fc2.None.0.A"]
    b = tensors["blocks.0.mlp.fc2.None.0.B"]
    base = layer(x)
    expected = base + F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))
    fp32_result = base + F.linear(F.linear(x.float(), a), b).to(x.dtype)
    assert not torch.equal(expected, fp32_result)
    with c.activate(0, 0.999), c.forward_hooks(c.current_selection()):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(value.dtype == torch.float32 for value in tensors.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_adapter_matches_diffusers_up_gate_activation(dtype):
    torch.manual_seed(31)
    native = NativeToy().to(dtype)
    native.blocks[0].mlp.fc1 = nn.Linear(4, 8, dtype=dtype)
    fc1 = native.blocks[0].mlp.fc1
    diffusers_weight = torch.randn_like(fc1.weight)
    diffusers_bias = torch.randn_like(fc1.bias)
    with torch.no_grad():
        fc1.weight.copy_(torch.cat(diffusers_weight.chunk(2)[::-1]))
        fc1.bias.copy_(torch.cat(diffusers_bias.chunk(2)[::-1]))
    tensors, pairs = bank(native, names=[("blocks.0.mlp.fc1", None)])
    for key in tensors:
        tensors[key] = torch.randn_like(tensors[key])
    originals = {key: value.clone() for key, value in tensors.items()}
    c = PTDController(native, tensors, pairs, "h3_4step", 4)
    x = torch.randn(16, 4).to(dtype)
    a = tensors["blocks.0.mlp.fc1.None.1.A"].to(dtype)
    b = tensors["blocks.0.mlp.fc1.None.1.B"].to(dtype)
    # Diffusers: first half is up, second half is the activated gate.
    adapted = F.linear(x, diffusers_weight, diffusers_bias) + F.linear(F.linear(x, a), b)
    up, gate = adapted.chunk(2, dim=-1)
    expected = native.blocks[0].mlp.fc2(up * F.silu(gate))
    with c.activate(1, 0.947), c.forward_hooks(c.current_selection()):
        # ComfyUI: first half is the activated gate, second half is up.
        gate, up = fc1(x).chunk(2, dim=-1)
        actual = native.blocks[0].mlp.fc2(F.silu(gate) * up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key, original in originals.items():
        torch.testing.assert_close(tensors[key], original, rtol=0, atol=0)


def test_adaln_silu_runs_before_bfloat16_cast_and_restores_hooks():
    native = NativeToy()
    t = torch.linspace(-5.0, 5.0, 64).unsqueeze(0)

    class FixedEmbedding(nn.Module):
        def forward(self, times):
            return t

    native.time_embedder = FixedEmbedding()
    native.final_layer.adaln_proj = nn.Module()
    native.final_layer.adaln_proj.linear = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    native.final_layer.adaln_proj.linear.weight.data.copy_(torch.eye(64))
    c = controller(native)
    expected = F.silu(t).to(torch.bfloat16)
    native_order = F.silu(t.to(torch.bfloat16))
    assert not torch.equal(expected, native_order)
    with c.activate(0, 0.999), c.forward_hooks(c.current_selection()):
        rounded_time = native.time_embedder(torch.tensor([.001])).to(torch.bfloat16)
        actual = native.final_layer.adaln_proj.linear(F.silu(rounded_time))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not native.time_embedder._forward_hooks
    assert not native.final_layer.adaln_proj.linear._forward_pre_hooks


def test_native_time_labels_recover_float64_clock_before_float32_cast():
    native = NativeToy()
    native.time_embedder = nn.Identity()
    c = controller(native)
    sigma = .947
    rounded_sigma = torch.tensor(sigma * 1000, dtype=torch.float32) / 1000
    shift = native_audio_shift(sigma)
    rounded_audio = rounded_sigma * shift / (1.0 + rounded_sigma * (shift - 1.0))
    times = torch.stack((1.0 - rounded_sigma, 1.0 - rounded_audio, torch.tensor(.999)))
    expected = torch.tensor([1.0 - sigma, 1.0 - audio_sigma(sigma), .999], dtype=torch.float32)
    assert not torch.equal(times, expected)
    with c.activate(1, sigma), c.forward_hooks(c.current_selection()):
        torch.testing.assert_close(native.time_embedder(times), expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="time labels"):
            native.time_embedder(torch.tensor([.4]))
    assert not native.time_embedder._forward_pre_hooks
    torch.testing.assert_close(native.time_embedder(times), times, rtol=0, atol=0)


def test_interruption_and_nested_activation_restore_everything():
    native = NativeToy()
    c = controller(native)
    native.blocks[0].mlp.fc2.comfy_force_forward = False
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("cancelled")

    c.interrupt_check = interrupt
    with c.activate(0, 0.999):
        with pytest.raises(KeyboardInterrupt), c.activate(1, 0.947), c.forward_hooks(c.current_selection()):
            native([torch.ones(1, 4)] * 2)
        assert c.current_selection().hop == 0
    assert native.blocks[0].mlp.fc2.comfy_force_forward is False
    assert all(not m._forward_hooks for m in native.modules())
    with pytest.raises(RuntimeError):
        c.current_selection()


def test_scoped_math_forwards_restore_and_ignore_other_contexts():
    native = NativeToy()
    block = native.blocks[0]
    block.norm1, block.norm2 = nn.Identity(), nn.Identity()
    block.adaln_proj = nn.Identity()
    block.forward = lambda x: x + 2
    block.attn.forward = lambda x: x + 3
    original_block, original_attention = block.forward, block.attn.forward
    c = controller(native)
    with pytest.raises(KeyboardInterrupt), c.activate(0, 0.999), c.forward_hooks(c.current_selection()):
        assert block.forward is not original_block
        assert block.attn.forward is not original_attention
        with ThreadPoolExecutor(max_workers=1) as pool:
            # An unpatched MODEL in another context retains its own forward.
            assert pool.submit(block, torch.tensor(1)).result().item() == 3
            assert pool.submit(block.attn, torch.tensor(1)).result().item() == 4
        raise KeyboardInterrupt("cancelled")
    assert block.forward is original_block
    assert block.attn.forward is original_attention
    assert all(not module._forward_hooks for module in native.modules())


def test_concurrent_original_call_ignores_scoped_hooks_and_clones_keep_their_hops():
    native = NativeToy()
    c1, c2 = controller(native), controller(native, offset=0.2)
    layer = native.blocks[0].attn.qkv_proj
    x = torch.ones(1, 4)
    baseline = layer(x)
    installed, release = threading.Event(), threading.Event()

    def first():
        with c1.activate(0, 0.999), c1.forward_hooks(c1.current_selection()):
            installed.set()
            assert release.wait(3)
            return layer(x)

    def second():
        with c2.activate(3, 0.667), c2.forward_hooks(c2.current_selection()):
            return layer(x)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first)
        assert installed.wait(3)
        # Original MODEL runs in a different context while hooks are installed.
        torch.testing.assert_close(layer(x), baseline)
        two = pool.submit(second)
        release.set()
        a, b = one.result(), two.result()
    torch.testing.assert_close(a, baseline + 0.16)
    torch.testing.assert_close(b, baseline + 1.92)
    assert all(not m._forward_hooks for m in native.modules())


def test_wrong_recipe_time_and_dcp_base_reference_are_rejected(monkeypatch):
    native = NativeToy()
    c = controller(native)
    with pytest.raises(ValueError, match="trained recipe"):
        with c.activate(0, 0.5):
            pass
    tensors, pairs = bank(native)
    monkeypatch.setattr("ptd_comfy.checkpoint.validate_adapter_tensors", lambda *a, **k: pairs)
    tensors["transformer.proj_out.orig_weight"] = native.final_layer.video_out.weight.detach().clone() + 1
    with pytest.raises(ValueError, match="base does not match"):
        patch_ptd_model(PatcherToy(native), tensors, {"ptd_recipe": "h3_4step"})
