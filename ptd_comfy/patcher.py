# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab.
# SPDX-License-Identifier: Apache-2.0

"""Apply PTD experts to a cloned native ComfyUI MiniMax-H3 MODEL.

The base MODEL retains its original parameters and offloading implementation.
Adapters run as low-rank additions to linear outputs, with only the active
layer's expert transferred from CPU. Hooks exist for one model call and are
removed even when sampling fails or is interrupted. No ComfyUI source is
vendored here; integration uses ModelPatcher's public object/wrapper APIs.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import dataclass
import math
import threading
from typing import Any, Mapping
import weakref

import torch
import torch.nn.functional as F


CONTROLLER_KEY = "ptd_native_controller"
_FORWARD: ContextVar[Any] = ContextVar("ptd_native_forward", default=None)
_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _model_lock(model):
    with _LOCKS_GUARD:
        lock = _LOCKS.get(model)
        if lock is None:
            lock = _LOCKS[model] = threading.RLock()
        return lock


def _check_interrupt():
    # Keep imports lazy so the adapter math can be tested without ComfyUI.
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _modulated(normed, shift, scale, segments):
    """Materialize each bf16 arithmetic operation as in the trained model."""
    result = torch.empty_like(normed)
    for start, stop, row in segments:
        result[start:stop] = normed[start:stop] * (1.0 + scale[row].to(normed.dtype)) + shift[row].to(normed.dtype)
    return result


def _residual(state, update, gate, segments):
    result = torch.empty_like(state)
    for start, stop, row in segments:
        # addcmul fuses away the product's bf16 rounding. Across 50 blocks,
        # that changes the network against which the PTD experts were trained.
        product = update[start:stop] * gate[row].to(state.dtype)
        result[start:stop] = state[start:stop] + product
    return result


def _block_math(block, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
    """AdaLN transformer equations using the existing native submodules."""
    options = {} if transformer_options is None else transformer_options
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.adaln_proj(t_emb)
    normed = _modulated(block.norm1(x), shift_a, scale_a, mod_segments)
    attend = block.attn if attention is None else attention
    update = attend(normed, rope_freqs=rope_freqs, transformer_options=options)
    x = _residual(x, update, gate_a, mod_segments)
    normed = _modulated(block.norm2(x), shift_m, scale_m, mod_segments)
    return _residual(x, block.mlp(normed), gate_m, mod_segments)


def _split_half_rotation(x, table):
    # H3 supplies one rotation matrix per token/pair, shared by all heads.
    cos_half = table[0, :, 0, :, 0, 0]
    sin_half = table[0, :, 0, :, 1, 0]
    cos = cos_half.repeat(1, 2).unsqueeze(1).to(x.dtype)
    sin = sin_half.repeat(1, 2).unsqueeze(1).to(x.dtype)
    width = cos.shape[-1]
    leading = x[..., :width]
    first, second = leading.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    # Keep the two products and their sum separate. CUDA's fused RMS-RoPE
    # otherwise evaluates the entire rotation in fp32 before a single cast.
    result = leading * cos + rotated * sin
    return torch.cat((result, x[..., width:]), dim=-1)


def _attention_math(attention, x, rope_freqs=None, transformer_options=None):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    options = {} if transformer_options is None else transformer_options
    streams = attention.qkv_proj(x).chunk(3, dim=-1)
    q, k, v = [value.reshape(x.shape[0], attention.heads, attention.head_dim) for value in streams]
    q, k = attention.q_norm(q), attention.k_norm(k)
    if rope_freqs is not None:
        q, k = _split_half_rotation(q, rope_freqs), _split_half_rotation(k, rope_freqs)
    packed = [AttentionTensorContainer(value.transpose(0, 1).unsqueeze(0)) for value in (q, k, v)]
    result = optimized_attention(*packed, attention.heads, skip_reshape=True,
                                 preferred_attention=attention.comfy_attention, transformer_options=options)
    return attention.out_proj(result.squeeze(0))


def _scope_forward(module, function, frame, originals):
    original = module.forward
    originals.append((module, "forward" in vars(module), original))

    def forward(*args, **kwargs):
        if _FORWARD.get() is not frame:
            return original(*args, **kwargs)
        return function(module, *args, **kwargs)

    module.forward = forward


@dataclass(frozen=True)
class _Selection:
    hop: int
    sigma: float


@dataclass(frozen=True)
class _Expert:
    a: torch.Tensor
    b: torch.Tensor
    qkv_index: int | None


class PTDController:
    """Immutable adapter bank with context-local hop selection.

    ``activate`` is entered by the PTD sampler for each evaluation. Clones may
    share this bank: selections are context-local, and simultaneous patched
    calls sharing a backbone are serialized. GPU copies are never cached in
    the bank, so offloading or an interrupted run cannot retain adapter VRAM.
    """

    def __init__(self, diffusion_model, tensors: Mapping[str, torch.Tensor], pairs,
                 recipe: str, hops: int):
        self.recipe = recipe
        self.hops = int(hops)
        self.diffusion_model = diffusion_model
        self._lock = _model_lock(diffusion_model)
        self._selection = ContextVar(f"ptd_selection_{id(self)}", default=None)
        self.interrupt_check = _check_interrupt
        bank = defaultdict(lambda: defaultdict(list))
        for pair in pairs:
            target = pair.target
            module = diffusion_model.get_submodule(target.module)
            a, b = tensors[pair.a_key], tensors[pair.b_key]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"Invalid PTD adapter matrices for {target.module}")
            if a.shape[1] != getattr(module, "in_features", None):
                raise ValueError(f"PTD input width does not match {target.module}")
            expected_output = b.shape[0] * (3 if target.qkv_index is not None else 1)
            if expected_output != getattr(module, "out_features", None):
                raise ValueError(f"PTD output width does not match {target.module}")
            if target.qkv_index not in (None, 0, 1, 2):
                raise ValueError(f"Invalid fused QKV slice for {target.module}")
            if a.dtype != b.dtype or not a.is_floating_point():
                raise ValueError(f"PTD adapter dtype mismatch for {target.module}")
            if target.swap_halves:
                if b.shape[0] % 2:
                    raise ValueError(f"PTD SwiGLU output width must be even for {target.module}")
                # Diffusers stores [up, gate]; native H3 consumes [gate, up].
                # The converted base has the same row permutation. Transform
                # only this private B copy, retaining the original checkpoint.
                up, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, up), dim=0)
            # Do not register these tensors on the native model or its modules.
            # Retain checkpoint precision in storage; the active layer chooses
            # its execution dtype (bf16 blocks/refiner, fp32 output heads).
            bank[target.module][pair.hop].append(
                _Expert(a.detach().to("cpu"), b.detach().to("cpu"), target.qkv_index)
            )
        self._bank = {name: {hop: tuple(experts) for hop, experts in by_hop.items()}
                      for name, by_hop in bank.items()}
        for name, by_hop in self._bank.items():
            if set(by_hop) != set(range(self.hops)):
                raise ValueError(f"PTD adapter is missing hops for {name}")

    @contextmanager
    def activate(self, hop: int, sigma: float):
        from ptd.recipes import get_recipe

        if isinstance(hop, bool) or not isinstance(hop, int) or not 0 <= hop < self.hops:
            raise ValueError(f"PTD hop must be in [0, {self.hops})")
        sigma = float(sigma)
        if not math.isfinite(sigma) or not 0.0 < sigma <= 0.9990001:
            raise ValueError("PTD model evaluations require 0 < sigma <= 0.999")
        expected = get_recipe(self.recipe).grid[hop]
        if abs(sigma - expected) > 2e-6:
            raise ValueError("The active PTD hop must use its trained recipe time")
        sigma = expected
        token = self._selection.set(_Selection(hop, sigma))
        try:
            yield self
        finally:
            self._selection.reset(token)

    def current_selection(self) -> _Selection:
        selection = self._selection.get()
        if selection is None:
            raise RuntimeError("Use PTDSampler with this MODEL; no PTD hop is active")
        return selection

    def _linear_hook(self, frame, experts):
        def hook(module, args, kwargs, output):
            if _FORWARD.get() is not frame:
                return output
            self.interrupt_check()
            x = args[0] if args else kwargs.get("input")
            if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
                raise TypeError("PTD adapters require tensor-valued linear inputs and outputs")
            result = output.clone() if any(e.qkv_index is not None for e in experts) else output
            for expert in experts:
                # Original PTD uses FSDP's param_dtype for the adapted layer.
                # Checkpoint storage may be fp32 even when that layer executes
                # in bf16; match the native linear's effective compute dtype.
                a = expert.a.to(device=x.device, dtype=output.dtype)
                b = expert.b.to(device=x.device, dtype=output.dtype)
                delta = F.linear(F.linear(x.to(a.dtype), a), b).to(output.dtype)
                if expert.qkv_index is None:
                    result = result + delta
                else:
                    width = delta.shape[-1]
                    start = expert.qkv_index * width
                    result[..., start:start + width] += delta
            return result
        return hook

    @contextmanager
    def forward_hooks(self, selection: _Selection):
        """Scope additions and arithmetic ordering to this call only."""
        from .sampling import audio_sigma, audio_sigma_prime

        with self._lock:
            frame = object()
            token = _FORWARD.set(frame)
            handles = []
            forced = []
            forwards = []
            try:
                for block in self.diffusion_model.blocks:
                    if all(hasattr(block, name) for name in ("norm1", "norm2", "adaln_proj", "attn", "mlp")):
                        _scope_forward(block, _block_math, frame, forwards)
                        _scope_forward(block.attn, _attention_math, frame, forwards)
                time_embedding = {}
                if hasattr(self.diffusion_model, "time_embedder"):
                    def exact_times(module, args):
                        if _FORWARD.get() is not frame:
                            return args
                        times = args[0]
                        if not isinstance(times, torch.Tensor) or times.ndim != 1:
                            raise ValueError("PTD expects a vector of native H3 time labels")
                        video_time = 1.0 - selection.sigma
                        candidates = torch.tensor(
                            [video_time, 1.0 - audio_sigma(selection.sigma), max(video_time, .999)],
                            device=times.device, dtype=torch.float32).unique()
                        distance, indices = (times[:, None] - candidates[None]).abs().min(dim=1)
                        if not bool((distance <= 2e-6).all()):
                            raise ValueError("Native H3 time labels do not match the active PTD clocks")
                        # Native H3 derives these labels after rounding sigma
                        # to fp32. PTD subtracts its float64 clock first. Keep
                        # the existing table order/modality indices unchanged.
                        return (candidates[indices],) + args[1:]

                    def remember_time(module, args, output):
                        if _FORWARD.get() is frame:
                            time_embedding["fp32"] = output

                    def precise_adaln_input(module, args):
                        if _FORWARD.get() is not frame:
                            return args
                        if "fp32" not in time_embedding:
                            raise RuntimeError("PTD AdaLN ran before the time embedder")
                        # Diffusers applies SiLU to the fp32 time embedding
                        # before casting to the AdaLN projection dtype. Native
                        # H3 casts the embedding first; preserve the training
                        # order without replacing either module's forward.
                        value = F.silu(time_embedding["fp32"]).to(args[0].dtype)
                        return (value,) + args[1:]

                    handles.append(self.diffusion_model.time_embedder.register_forward_pre_hook(exact_times))
                    handles.append(self.diffusion_model.time_embedder.register_forward_hook(remember_time))
                    for layer in (*self.diffusion_model.blocks, self.diffusion_model.final_layer):
                        adaln = getattr(layer, "adaln_proj", None)
                        if adaln is not None and getattr(adaln, "apply_silu", True):
                            handles.append(adaln.linear.register_forward_pre_hook(precise_adaln_input))
                for name, by_hop in self._bank.items():
                    module = self.diffusion_model.get_submodule(name)
                    # Native SwiGLU may fuse the activation and fc2 matmul,
                    # bypassing Module.__call__ and therefore PyTorch hooks.
                    # Comfy's public bypass flag routes that call through the
                    # original linear, retaining its normal weight offloading.
                    if hasattr(module, "comfy_force_forward") or name.endswith(".mlp.fc2"):
                        forced.append((module, "comfy_force_forward" in vars(module),
                                       getattr(module, "comfy_force_forward", None)))
                        module.comfy_force_forward = True
                    handles.append(module.register_forward_hook(
                        self._linear_hook(frame, by_hop[selection.hop]), with_kwargs=True))

                def audio_clock_hook(module, args, output):
                    if _FORWARD.get() is not frame:
                        return output
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("PTD requires a tensor-valued native H3 audio head")
                    # Scale the adapted head before native H3 casts to the
                    # latent dtype. Original PTD retains fp32 output heads
                    # through this multiplication, then rounds to bf16.
                    factor = torch.as_tensor(audio_sigma_prime(selection.sigma),
                                             dtype=output.dtype, device=output.device)
                    return output * factor

                handles.append(self.diffusion_model.final_layer.audio_out.register_forward_hook(audio_clock_hook))
                self.interrupt_check()
                yield
                self.interrupt_check()
            finally:
                for handle in reversed(handles):
                    handle.remove()
                for module, had_instance_value, value in reversed(forced):
                    if had_instance_value:
                        module.comfy_force_forward = value
                    else:
                        del module.comfy_force_forward
                for module, had_instance_value, original in reversed(forwards):
                    if had_instance_value:
                        module.forward = original
                    else:
                        del module.forward
                _FORWARD.reset(token)


class _ModelFunctionWrapper:
    def __init__(self, controller: PTDController, previous=None):
        self.controller = controller
        self.previous = previous

    def __call__(self, model_function, kwargs):
        from .sampling import native_audio_shift

        selection = self.controller.current_selection()
        sigma = kwargs.get("timestep")
        if not isinstance(sigma, torch.Tensor) or sigma.numel() == 0:
            raise ValueError("PTD requires a tensor sigma from the native sampler")
        if not bool(torch.allclose(sigma, torch.full_like(sigma, selection.sigma),
                                   rtol=0.0, atol=2e-6)):
            raise ValueError("The model sigma does not match the active PTD hop")
        forwarded = dict(kwargs)
        conditions = dict(kwargs.get("c", {}))
        options = dict(conditions.get("transformer_options", {}))
        options["minimax_h3_sigma_shift_video"] = 1.0
        options["minimax_h3_sigma_shift_audio"] = native_audio_shift(selection.sigma)
        conditions["transformer_options"] = options
        payload = dict(conditions.get("minimax_payload", {}))
        payload["audio_scale"] = 1.0
        conditions["minimax_payload"] = payload
        forwarded["c"] = conditions
        with self.controller.forward_hooks(selection):
            if self.previous is not None:
                return self.previous(model_function, forwarded)
            return model_function(forwarded["input"], forwarded["timestep"], **conditions)

    def to(self, device):
        # Comfy calls this for offload/dtype transitions. Keep the bank on CPU.
        if self.previous is not None and hasattr(self.previous, "to"):
            self.previous = self.previous.to(device)
        return self

    def cleanup(self, **kwargs):
        if self.previous is not None and hasattr(self.previous, "cleanup"):
            self.previous.cleanup(**kwargs)

    def models(self):
        # Preserve any existing wrapper's auxiliary models so ComfyUI can
        # account for their memory and load/offload them with the backbone.
        if self.previous is not None and hasattr(self.previous, "models"):
            return self.previous.models()
        return []


def get_ptd_controller(model) -> PTDController:
    controller = model.model_options.get(CONTROLLER_KEY)
    if not isinstance(controller, PTDController):
        raise ValueError("Connect a MODEL produced by PTDLoader")
    return controller


def patch_ptd_model(model, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]):
    """Return a native MODEL clone carrying a validated PTD expert bank."""
    from .checkpoint import BUFFER_TARGETS, validate_adapter_tensors

    if CONTROLLER_KEY in model.model_options:
        raise ValueError("This MODEL already has PTD adapters; start from the base MODEL")
    recipe = metadata.get("ptd_recipe")
    if recipe not in ("h3_4step", "h3_6step"):
        raise ValueError("The PTD adapter file must identify h3_4step or h3_6step")
    hops = 4 if recipe == "h3_4step" else 6
    native = model.get_model_object("diffusion_model")
    required = ("blocks", "token_refiner", "condition_proj", "final_layer",
                "preprocess_text_embeds", "sigma_shift_video", "sigma_shift_audio")
    if not all(hasattr(native, name) for name in required):
        raise ValueError("PTDLoader requires a native MiniMax-H3 MODEL")
    if getattr(native, "use_adaln_curves", False):
        raise ValueError("PTD requires the original MiniMax-H3 base, not an AdaLN-curve variant")
    for name in ("video_out", "audio_out"):
        head = getattr(native.final_layer, name)
        if head.weight.shape[0] != head.out_features:
            raise ValueError("PTD requires the original MiniMax-H3 base, not a PDD head bank")
    pairs = validate_adapter_tensors(tensors, recipe,
                                    expected_blocks=len(native.blocks),
                                    expected_refiner_blocks=len(native.token_refiner.blocks))
    for key, target in BUFFER_TARGETS.items():
        if key not in tensors:
            continue
        parent, _, attr = target.rpartition(".")
        base = getattr(native.get_submodule(parent), attr)
        value = tensors[key]
        if (not isinstance(base, torch.Tensor) or base.device.type == "meta"
                or base.is_quantized or base.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
            raise ValueError(f"Cannot verify PTD base reference {target}; use the original floating-point base")
        if base.shape != value.shape or not torch.equal(base.detach().cpu(), value.to(device="cpu", dtype=base.dtype)):
            raise ValueError(f"The loaded MiniMax-H3 base does not match PTD reference {target}")
    controller = PTDController(native, tensors, pairs, recipe, hops)

    clone = model.clone()
    sampling = copy.deepcopy(model.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift") or not hasattr(sampling, "shift"):
        raise ValueError("PTD requires the native MiniMax-H3 audio/video flow sampler")
    sampling.audio_shift = sampling.shift
    clone.add_object_patch("model_sampling", sampling)

    text_dim = native.condition_proj.in_features

    def raw_text(text_states):
        if text_states.shape[-1] != text_dim:
            raise ValueError("PTD requires raw MiniMax-H3 text conditioning, not pre-refined embeddings")
        return text_states

    # Native H3 normally refines text once in extra_conds. Keep raw embeddings
    # instead: _embed_and_pack then applies the active hop's refiner each call.
    clone.add_object_patch("diffusion_model.preprocess_text_embeds", raw_text)
    previous = clone.model_options.get("model_function_wrapper")
    clone.set_model_unet_function_wrapper(_ModelFunctionWrapper(controller, previous))
    clone.model_options[CONTROLLER_KEY] = controller
    return clone
