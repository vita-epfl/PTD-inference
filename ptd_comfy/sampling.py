# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab.
# SPDX-License-Identifier: Apache-2.0

"""PTD's fixed hop integrator behind ComfyUI's public KSAMPLER interface.

The patched H3 model must expose physical audio/video latents (audio_scale=1)
with shared-clock mean velocities. Its controller converts H3's audio clock
and selects each hop's adapters; ComfyUI then returns ``denoised = x - t*h``.
This module implements the PTD recurrence, independently of ComfyUI internals.

Hop transitions follow the checkpoint's recipe: the paper's recipes carry the
Euler endpoint of every hop; the ``_sde`` recipes (``Recipe.renoise = 1``) carry,
from every hop but the last, the hop's clean endpoint re-noised to the next
knot with fresh noise (``renoised_transition``, the counterpart of
``ptd.sampler.hop_transition``), drawn from a generator seeded with the
sampler's ``seed`` (the workflow's Random Noise seed).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

from ptd.recipes import get_recipe
from ptd.schedule import Clock

T_MAX = 0.999
AUDIO_RELATIVE_SHIFT = 0.25


def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= T_MAX:
        raise ValueError(f"PTD shared time must lie in [0, {T_MAX}]")
    return value


def audio_sigma(time: float) -> float:
    """Audio's physical noise level on PTD's T=.999 shared video clock."""
    time = _time(time)
    if time == T_MAX:
        return T_MAX
    return AUDIO_RELATIVE_SHIFT * time * T_MAX / (T_MAX + (AUDIO_RELATIVE_SHIFT - 1.0) * time)


def audio_sigma_prime(time: float) -> float:
    """Derivative d(audio_sigma)/dt used to convert the audio mean head."""
    time = _time(time)
    denominator = T_MAX + (AUDIO_RELATIVE_SHIFT - 1.0) * time
    return AUDIO_RELATIVE_SHIFT * T_MAX**2 / denominator**2


def native_audio_shift(time: float) -> float:
    """T=1 shift producing PTD's audio sigma at this step (video shift=1).

    The native H3 time-shift function has a unit upper endpoint. Its ordinary
    constant 3/12 ratio is therefore not PTD's clock with upper endpoint .999.
    This ratio is applied only while evaluating the current hop.
    """
    time = _time(time)
    if time == 0.0:
        return AUDIO_RELATIVE_SHIFT
    if time == T_MAX:
        return 1.0
    sigma = audio_sigma(time)
    return sigma * (1.0 - time) / (time * (1.0 - sigma))


def ptd_sigmas(recipe: str, device=None) -> torch.Tensor:
    """Return the published four/six-step knots, without recomputing a grid."""
    return torch.tensor(get_recipe(recipe).grid, dtype=torch.float64, device=device)


def _streams(value):
    if getattr(value, "is_nested", False):
        streams = tuple(value.unbind())
        if len(streams) != 2:
            raise ValueError("PTD expects exactly video and audio latent streams")
        return streams
    if not isinstance(value, torch.Tensor):
        raise TypeError("PTD latents must be a Tensor or ComfyUI NestedTensor")
    return (value,)


def _map(value, operation: Callable):
    streams = _streams(value)
    result = [operation(stream) for stream in streams]
    return type(value)(result) if getattr(value, "is_nested", False) else result[0]


def _pair(left, right, operation: Callable):
    a, b = _streams(left), _streams(right)
    if len(a) != len(b) or bool(getattr(left, "is_nested", False)) != bool(getattr(right, "is_nested", False)):
        raise ValueError("model output must retain the latent stream structure")
    if any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError("model output shape differs from the latent state")
    result = [operation(x, y) for x, y in zip(a, b)]
    return type(left)(result) if getattr(left, "is_nested", False) else result[0]


def _clocks(value):
    """The clock of every stream: video then audio for nested latents; a single stream is on the shared (video) clock."""
    return (Clock(1.0), Clock(AUDIO_RELATIVE_SHIFT)) if getattr(value, "is_nested", False) else (Clock(1.0),)


def initial_state(noise):
    """Pure-noise PTD start: both clocks equal .999, with fp32 state carry.

    Multiplication occurs in float64, matching Clock.latents. Do not call this
    on the already scaled state passed to a bare sampling function.
    """
    return _map(noise, lambda stream: (stream.to(torch.float64) * T_MAX).to(torch.float32))


def denoised_to_velocity(state, denoised, time: float):
    """Recover shared-clock mean velocity from ComfyUI's x0 convention."""
    time = _time(time)
    if time == 0.0:
        raise ValueError("cannot recover a mean velocity at the clean endpoint")
    return _pair(state, denoised, lambda x, x0: (x.float() - x0.float()) / time)


def renoised_transition(state, velocity, time: float, next_time: float, fraction: float, generator=None):
    """The ``_sde`` carry from the hop at ``time`` to the knot ``next_time`` (``ptd.sampler.hop_transition``).

    Per stream, on its clock: the hop's clean endpoint ``x0 = x - (sigma / sigma')(time) h`` (float64 arithmetic,
    as the original) re-noised to ``sigma(next_time)`` with fresh float32 noise from ``generator``, drawn video first,
    then audio; ``fraction`` < 1 keeps ``sqrt(1 - fraction)`` of the noise the Euler endpoint carries.
    """
    time, next_time = _time(time), _time(next_time)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("the re-noised fraction must lie in (0, 1]")
    if next_time == 0.0:
        raise ValueError("the last hop carries its Euler endpoint and is never re-noised")
    _pair(state, velocity, lambda a, b: a)  # same stream structure and shapes
    first = _streams(state)[0]
    t_a = torch.tensor(time, dtype=torch.float64, device=first.device)
    t_e = torch.tensor(next_time, dtype=torch.float64, device=first.device)
    delta = (t_e - t_a).float()
    keep, new = math.sqrt(max(1.0 - fraction, 0.0)), math.sqrt(min(fraction, 1.0))
    result = []
    for x, h, clock in zip(_streams(state), _streams(velocity), _clocks(state)):
        x32, h32 = x.float(), h.float()
        x_e = x32 + delta * h32
        x0 = clock.flow_to_x0(x32, h32, t_a)
        sg = clock.sigma(t_e).to(torch.float32).clamp(min=1e-6)
        fresh = torch.randn(x_e.shape, generator=generator, device=x_e.device, dtype=torch.float32)
        if fraction >= 1.0:
            result.append((1.0 - sg) * x0 + sg * fresh)
        else:
            eps = (x_e - (1.0 - sg) * x0) / sg
            result.append((1.0 - sg) * x0 + sg * (keep * eps + new * fresh))
    return type(state)(result) if getattr(state, "is_nested", False) else result[0]


def _validate_grid(sigmas: torch.Tensor, recipe: str) -> None:
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.dtype not in (torch.float32, torch.float64):
        raise ValueError("PTD requires a one-dimensional float32/float64 recipe grid")
    expected = ptd_sigmas(recipe).to(dtype=sigmas.dtype)
    if sigmas.shape != expected.shape or not torch.equal(sigmas.detach().cpu(), expected):
        raise ValueError(f"PTD requires the complete, unmodified {recipe} grid; use PTD Sigmas")


def _validate_initial_inputs(model, extra_args) -> None:
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("PTD sampling does not support denoise masks or partial denoising")
    latent_image = getattr(model, "latent_image", None)
    if latent_image is not None and any(bool(torch.count_nonzero(x)) for x in _streams(latent_image)):
        raise ValueError("PTD requires an empty generated latent; supply image conditions through H3 conditioning")


@torch.no_grad()
def sample_ptd(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller, recipe: str | None = None):
    """Evaluate exactly one trained expert per hop and carry state in float32.

    ``model`` follows the k-diffusion sampler callback contract. A KSAMPLER's
    wrapper exposes its original ``noise``; use it to reproduce PTD's exact
    initial scaling instead of scaling ``x`` twice. For direct calls without
    that attribute, ``x`` is already the initial state. The controller's
    ``activate(hop, time)`` context must restore its state even on failure.
    For an ``_sde`` recipe the fresh noise of the re-noised transitions comes
    from a generator seeded with ``extra_args["seed"]`` (ComfyUI's sampler
    seed); without a seed it comes from the global RNG.
    """
    del disable  # ComfyUI owns progress reporting through callback.
    recipe = controller.recipe if recipe is None else recipe
    if controller.recipe != recipe:
        raise ValueError("sampler recipe differs from the loaded PTD checkpoint")
    _validate_grid(sigmas, recipe)
    extra_args = {} if extra_args is None else dict(extra_args)
    _validate_initial_inputs(model, extra_args)
    raw_noise = getattr(model, "noise", None)
    state = initial_state(raw_noise) if raw_noise is not None else _map(x, lambda value: value.float())
    # Validate the externally supplied initial shape, including packed vs nested.
    _pair(state, x, lambda a, b: a)
    first = _streams(state)[0]
    if first.ndim == 0 or first.shape[0] < 1:
        raise ValueError("PTD latents must have a nonempty batch dimension")
    if any(part.shape[0] != first.shape[0] or part.device != first.device for part in _streams(state)):
        raise ValueError("PTD latent streams must share a batch size and device")
    grid = ptd_sigmas(recipe, device=first.device)
    fraction = float(getattr(get_recipe(recipe), "renoise", 0.0))
    generator = None
    if fraction > 0.0 and extra_args.get("seed") is not None:
        generator = torch.Generator(device=first.device).manual_seed(int(extra_args["seed"]))
    for hop in range(len(grid) - 1):
        time = float(grid[hop])
        # Keep the exact stored knot in the controller, even if a caller supplied
        # the standard float32 ComfyUI SIGMAS representation.
        sigma_batch = grid[hop].expand(first.shape[0])
        with controller.activate(hop, time):
            denoised = model(state, sigma_batch, **extra_args)
        velocity = denoised_to_velocity(state, denoised, time)
        if callback is not None:
            callback({"x": state, "i": hop, "sigma": grid[hop], "sigma_hat": grid[hop], "denoised": denoised})
        if fraction > 0.0 and hop + 2 < len(grid):  # every transition but the last
            state = renoised_transition(state, velocity, time, float(grid[hop + 1]), fraction, generator)
        else:
            delta = (grid[hop + 1] - grid[hop]).float()
            state = _pair(state, velocity, lambda value, flow: value.float() + delta * flow.float())
    return state


def make_sampler(controller, recipe: str | None = None):
    """Construct a standard ComfyUI SAMPLER without importing ComfyUI at load."""
    from comfy.samplers import KSAMPLER

    recipe = controller.recipe if recipe is None else recipe
    if controller.recipe != recipe:
        raise ValueError("sampler recipe differs from the loaded PTD checkpoint")
    get_recipe(recipe)
    return KSAMPLER(sample_ptd, extra_options={"controller": controller, "recipe": recipe})
