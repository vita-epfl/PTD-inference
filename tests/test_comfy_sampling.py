# SPDX-FileCopyrightText: Copyright (c) 2026 EPFL VITA Lab.
# SPDX-License-Identifier: Apache-2.0

"""CPU numerical checks; no ComfyUI runtime, weights, network or GPU needed."""
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from ptd.recipes import get_recipe
from ptd.sampler import sample as reference_sample
from ptd.schedule import Clock
from ptd_comfy.sampling import (
    T_MAX, audio_sigma, audio_sigma_prime, denoised_to_velocity,
    initial_state, make_sampler, native_audio_shift, ptd_sigmas, sample_ptd,
)


class StreamPair:
    """Minimal external NestedTensor protocol, intentionally without arithmetic."""
    is_nested = True

    def __init__(self, tensors):
        self.values = tuple(tensors)

    def unbind(self):
        return self.values


class Controller:
    def __init__(self, recipe):
        self.recipe = recipe
        self.active = None
        self.visited = []

    @contextmanager
    def activate(self, hop, sigma):
        assert self.active is None
        self.active = (hop, sigma)
        self.visited.append(self.active)
        try:
            yield
        finally:
            self.active = None


@pytest.mark.parametrize('recipe', ['h3_4step', 'h3_6step'])
def test_exact_released_knots(recipe):
    grid = ptd_sigmas(recipe)
    assert grid.dtype == torch.float64
    assert tuple(grid.tolist()) == get_recipe(recipe).grid
    assert len(grid) == get_recipe(recipe).hops + 1
    assert grid[-1] == 0


@pytest.mark.parametrize('time', [0.0, .1, .545, .667, .75, .857, .923, .947, .968, .999])
def test_audio_clock_matches_original_and_native_effective_shift(time):
    t = torch.tensor(time, dtype=torch.float64)
    clock = Clock(.25)
    assert audio_sigma(time) == pytest.approx(float(clock.sigma(t)), abs=1e-15)
    assert audio_sigma_prime(time) == pytest.approx(float(clock.sigma_prime(t)), abs=1e-14)
    shift = native_audio_shift(time)
    mapped = shift * time / (1 + (shift - 1) * time)
    assert mapped == pytest.approx(audio_sigma(time), abs=1e-15)
    if 0 < time < T_MAX:
        epsilon = 1e-6
        difference = (audio_sigma(time + epsilon) - audio_sigma(time - epsilon)) / (2 * epsilon)
        assert audio_sigma_prime(time) == pytest.approx(difference, rel=1e-8)


def test_audio_endpoint_is_shared_noise_level_not_native_unit_clock():
    assert audio_sigma(T_MAX) == T_MAX
    assert audio_sigma_prime(T_MAX) == pytest.approx(4)
    assert native_audio_shift(T_MAX) == 1
    ordinary_native_audio = .25 * T_MAX / (1 - .75 * T_MAX)
    assert abs(ordinary_native_audio - audio_sigma(T_MAX)) > .002


@pytest.mark.parametrize('invalid', [-.1, 1, float('nan'), float('inf')])
def test_clock_rejects_invalid_times(invalid):
    for helper in [audio_sigma, audio_sigma_prime, native_audio_shift]:
        with pytest.raises(ValueError):
            helper(invalid)


def test_initial_state_uses_float64_scale_and_preserves_stream_shapes():
    video = torch.linspace(-3, 3, 101).reshape(1, 1, 101)
    audio = torch.tensor([[[[.1, .5], [-.2, -.7]]]], dtype=torch.bfloat16)
    result = initial_state(StreamPair([video, audio]))
    assert isinstance(result, StreamPair)
    for actual, noise in zip(result.unbind(), [video, audio]):
        assert actual.dtype == torch.float32
        assert torch.equal(actual, (noise.double() * .999).float())
    assert not torch.equal(result.unbind()[0], video * .999)


@pytest.mark.parametrize('recipe', ['h3_4step', 'h3_6step'])
@pytest.mark.parametrize('grid_dtype', [torch.float32, torch.float64])
def test_each_expert_once_fp32_carry_and_callback_contract(recipe, grid_dtype):
    controller = Controller(recipe)
    x = torch.tensor([[.999, -.7]], dtype=torch.float32)
    states = []
    callbacks = []

    def model(state, sigma, **kwargs):
        hop, time = controller.active
        assert state.dtype == torch.float32
        assert sigma.dtype == torch.float64
        assert float(sigma[0]) == get_recipe(recipe).grid[hop] == time
        assert kwargs['seed'] == 7
        states.append(state.clone())
        # An exact shared-clock mean head for each interval.
        velocity = torch.full_like(state, (hop + 1) / 16)
        return state - sigma[0] * velocity

    def callback(record):
        assert controller.active is None
        callbacks.append(record)

    args = {'seed': 7}
    actual = sample_ptd(model, x, ptd_sigmas(recipe).to(grid_dtype), extra_args=args,
                        controller=controller, callback=callback)
    expected = x.clone()
    knots = get_recipe(recipe).grid
    for hop, (start, end) in enumerate(zip(knots[:-1], knots[1:])):
        assert torch.allclose(states[hop], expected, atol=2e-7, rtol=0)
        expected = expected + torch.tensor(end - start, dtype=torch.float32) * ((hop + 1) / 16)
    assert torch.allclose(actual, expected, atol=2e-7, rtol=0)
    assert actual.dtype == torch.float32
    assert controller.visited == list(enumerate(knots[:-1]))
    assert [entry['i'] for entry in callbacks] == list(range(len(knots) - 1))
    assert all(set(entry) == {'x', 'i', 'sigma', 'sigma_hat', 'denoised'} for entry in callbacks)
    assert args == {'seed': 7}


def test_ksampler_initial_noise_is_scaled_once_using_original_raw_noise():
    controller = Controller('h3_4step')
    raw = torch.linspace(-3, 3, 101).reshape(1, 101)
    seen = []

    class Model:
        noise = raw
        latent_image = torch.zeros_like(raw)

        def __call__(self, state, sigma, **kwargs):
            seen.append(state.clone())
            return state  # zero mean velocity

    # KSAMPLER has already made this lower-precision multiplication; the function
    # must recover its original noise, rather than applying .999 to this twice.
    actual = sample_ptd(Model(), raw * .999, ptd_sigmas(controller.recipe), controller=controller)
    expected = (raw.double() * .999).float()
    assert torch.equal(seen[0], expected)
    assert torch.equal(actual, expected)


class ReferenceToy:
    """The original PTD protocol with nonlinear, hop-specific BF16 heads."""
    def __init__(self, recipe):
        self.N_HOPS = get_recipe(recipe).hops
        self.input_dtypes = []

    def latents(self, noise, time):
        return {'video': Clock().latents(noise['video'], time),
                'audio': Clock(.25).latents(noise['audio'], time)}

    def __call__(self, x, t, condition, hop):
        self.input_dtypes.append((x['video'].dtype, x['audio'].dtype))
        video = (x['video'] * .03125 + (hop + 1) * .015625).to(torch.bfloat16)
        audio = (x['audio'] * .0625 - (hop + 1) * .0078125).to(torch.bfloat16)
        # PTD rounds sigma_prime to the head's dtype before multiplying it.
        derivative = Clock(.25).sigma_prime(t).to(audio.dtype).reshape([-1] + [1] * (audio.ndim - 1))
        return {'video': video, 'audio': audio * derivative}


@pytest.mark.parametrize('recipe', ['h3_4step', 'h3_6step'])
def test_nested_streams_match_original_ptd_integrator_with_bf16_model_evaluation(recipe):
    noise = {'video': torch.tensor([[[[[-.7, .13], [.5, 1.2]]]]]),
             'audio': torch.tensor([[[[-.9, .17], [.3, 1.7]]]])}
    ref = ReferenceToy(recipe)
    expected = reference_sample(ref, noise, None, ptd_sigmas(recipe))
    native = ReferenceToy(recipe)
    controller = Controller(recipe)

    raw_noise = StreamPair([noise['video'], noise['audio']])
    empty_latent = StreamPair([torch.zeros_like(noise['video']), torch.zeros_like(noise['audio'])])

    class Model:
        noise = raw_noise
        latent_image = empty_latent

        def __call__(self, state, sigma, **kwargs):
            video, audio = state.unbind()
            hop, time = controller.active
            # BaseModel casts network inputs, but keeps its denoised arithmetic
            # in fp32 using the unrounded carried state.
            velocity = native({'video': video.bfloat16(), 'audio': audio.bfloat16()}, sigma, None, hop)
            return StreamPair([video - sigma[0] * velocity['video'].float(),
                               audio - sigma[0] * velocity['audio'].float()])

    actual = sample_ptd(Model(), initial_state(Model.noise), ptd_sigmas(recipe), controller=controller)
    for value, name in zip(actual.unbind(), ['video', 'audio']):
        assert value.dtype == torch.float32
        assert torch.allclose(value, expected[name], atol=3e-7, rtol=0)
    assert native.input_dtypes == ref.input_dtypes == [(torch.bfloat16, torch.bfloat16)] * get_recipe(recipe).hops


def test_active_hop_is_reset_after_model_failure():
    controller = Controller('h3_4step')

    def broken(state, sigma, **kwargs):
        if controller.active[0] == 1:
            raise RuntimeError('test model failure')
        return state

    with pytest.raises(RuntimeError, match='test model failure'):
        sample_ptd(broken, torch.zeros(1, 2), ptd_sigmas(controller.recipe), controller=controller)
    assert controller.active is None
    assert len(controller.visited) == 2


@pytest.mark.parametrize('change', ['short', 'middle', 'reverse', 'bfloat16', 'matrix'])
def test_changed_or_partial_schedule_is_rejected_before_model_call(change):
    controller = Controller('h3_4step')
    grid = ptd_sigmas(controller.recipe)
    if change == 'short':
        grid = grid[1:]
    elif change == 'middle':
        grid[1] += 1e-4
    elif change == 'reverse':
        grid = grid.flip(0)
    elif change == 'bfloat16':
        grid = grid.bfloat16()
    else:
        grid = grid.reshape(1, -1)
    with pytest.raises(ValueError, match='grid'):
        sample_ptd(None, torch.zeros(1, 2), grid, controller=controller)
    assert controller.visited == []


def test_wrong_recipe_mask_and_nonempty_latent_are_rejected():
    controller = Controller('h3_4step')
    grid = ptd_sigmas(controller.recipe)
    for model, args, recipe in [
        (None, {}, 'h3_6step'),
        (None, {'denoise_mask': torch.ones(1, 2)}, None),
        (SimpleNamespace(latent_image=torch.ones(1, 2)), {}, None),
    ]:
        with pytest.raises(ValueError):
            sample_ptd(model, torch.zeros(1, 2), grid, extra_args=args, recipe=recipe, controller=controller)
    assert controller.visited == []


def test_zero_time_and_mismatched_model_structure_are_rejected():
    with pytest.raises(ValueError, match='clean endpoint'):
        denoised_to_velocity(torch.ones(1, 2), torch.zeros(1, 2), 0)
    with pytest.raises(ValueError, match='structure'):
        denoised_to_velocity(torch.ones(1, 2), StreamPair([torch.ones(1, 2)] * 2), .5)
    with pytest.raises(ValueError, match='shape'):
        denoised_to_velocity(torch.ones(1, 2), torch.zeros(1, 3), .5)


def test_factory_uses_native_ksampler_interface_without_importing_runtime(monkeypatch):
    captured = {}

    class KSampler:
        def __init__(self, function, *, extra_options):
            captured.update(function=function, options=extra_options)

    comfy = ModuleType('comfy')
    samplers = ModuleType('comfy.samplers')
    samplers.KSAMPLER = KSampler
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.samplers', samplers)
    controller = Controller('h3_4step')
    result = make_sampler(controller)
    assert isinstance(result, KSampler)
    assert captured == {'function': sample_ptd, 'options': {'controller': controller, 'recipe': 'h3_4step'}}


@pytest.fixture
def native_comfy():
    """Optional real-core integration: put an installed ComfyUI on PYTHONPATH.

    Only the tiny network below is synthetic. CFGGuider, ModelPatcher, latent
    packing, condition batching, flow conventions and KSAMPLER run unchanged.
    """
    import importlib.util
    if importlib.util.find_spec('comfy') is None:
        pytest.skip('native ComfyUI integration requires ComfyUI on PYTHONPATH')
    import comfy.cli_args
    comfy.cli_args.args.cpu = True  # must precede model_management initialization
    import comfy.conds
    import comfy.model_management
    import comfy.model_patcher
    import comfy.model_sampling
    import comfy.nested_tensor
    import comfy.samplers
    import comfy.utils
    return SimpleNamespace(conds=comfy.conds, management=comfy.model_management,
                           patcher=comfy.model_patcher, sampling=comfy.model_sampling,
                           nested=comfy.nested_tensor, samplers=comfy.samplers, utils=comfy.utils)


def _native_tiny_guider(core, controller):
    class Sampling(core.sampling.ModelSamplingAV, core.sampling.CONST):
        pass

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = SimpleNamespace(sampling_settings={'shift': 12., 'audio_shift': 12.})
            self.model_sampling = Sampling(config)
            self.current_patcher = None
            self.calls = []
            self.latent_shapes = None
            self.register_buffer('dummy_weight', torch.zeros(1))

        def get_dtype(self):
            return torch.float32

        get_dtype_inference = get_dtype

        def memory_required(self, input_shape, cond_shapes=None):
            return 1024

        def extra_conds_shapes(self, **kwargs):
            return {}

        def extra_conds(self, **kwargs):
            return {'latent_shapes': core.conds.CONDConstant(kwargs['latent_shapes'])}

        def process_latent_in(self, value):
            return value

        def process_latent_out(self, value):
            return value

        def apply_model(self, x, sigma, *, latent_shapes, transformer_options):
            hop, time = controller.active
            assert transformer_options['integration_sentinel'] == 'preserved'
            assert tuple(transformer_options['sample_sigmas'].tolist()) == get_recipe(controller.recipe).grid
            assert x.dtype == torch.float32
            assert sigma.dtype == torch.float64
            self.calls.append((x.clone(), tuple(latent_shapes), float(sigma[0])))
            # A synthetic mean head through the real native flow x0 conversion.
            head = torch.full_like(x, (hop + 1) / 16)
            return self.model_sampling.calculate_denoised(sigma, head, x)

    tiny = TinyModel()
    cpu = torch.device('cpu')
    patcher = core.patcher.ModelPatcher(tiny, cpu, cpu)
    patcher.model_options['transformer_options']['integration_sentinel'] = 'preserved'
    guider = core.samplers.CFGGuider(patcher)
    guider.set_conds([[None, {}]], [[None, {}]])
    guider.set_cfg(1.)
    return guider, tiny, patcher


@pytest.mark.parametrize('recipe', ['h3_4step', 'h3_6step'])
def test_real_comfy_cfg_guider_and_ksampler_pack_nested_streams(native_comfy, recipe):
    core = native_comfy
    controller = Controller(recipe)
    guider, tiny, patcher = _native_tiny_guider(core, controller)
    video = torch.linspace(-3, 3, 24).reshape(1, 1, 2, 3, 4)
    audio = torch.tensor([[[[.1, .5], [-.2, -.7]]]])
    noise = core.nested.NestedTensor([video, audio])
    latent = core.nested.NestedTensor([torch.zeros_like(video), torch.zeros_like(audio)])
    callbacks = []
    try:
        result = guider.sample(noise, latent, make_sampler(controller), ptd_sigmas(recipe),
                               callback=lambda *args: callbacks.append(args), disable_pbar=True, seed=42)
        assert isinstance(result, core.nested.NestedTensor)
        assert [tuple(value.shape) for value in result.unbind()] == [tuple(video.shape), tuple(audio.shape)]
        assert all(value.dtype == torch.float32 for value in result.unbind())
        packed, shapes = core.utils.pack_latents([video, audio])
        expected = initial_state(packed)
        assert torch.equal(tiny.calls[0][0], expected)
        knots = get_recipe(recipe).grid
        for hop, (start, end) in enumerate(zip(knots[:-1], knots[1:])):
            assert tiny.calls[hop][1] == tuple(shapes)
            assert tiny.calls[hop][2] == start
            expected = expected + torch.tensor(end - start, dtype=torch.float32) * ((hop + 1) / 16)
        actual, _ = core.utils.pack_latents(result.unbind())
        assert torch.allclose(actual, expected, atol=8e-7, rtol=0)
        assert len(callbacks) == get_recipe(recipe).hops
        assert all(isinstance(entry[1], core.nested.NestedTensor) and isinstance(entry[2], core.nested.NestedTensor)
                   for entry in callbacks)
        assert controller.active is None
        # The real guider temporarily clones options. Our sigma state must not
        # leak into the reusable MODEL or change the user's sentinel.
        assert patcher.model_options['transformer_options'] == {'integration_sentinel': 'preserved'}
    finally:
        core.management.unload_all_models()


def test_real_comfy_mask_rejection_precedes_network_call(native_comfy):
    core = native_comfy
    controller = Controller('h3_4step')
    guider, tiny, _ = _native_tiny_guider(core, controller)
    noise = torch.zeros(1, 1, 2, 2)
    try:
        with pytest.raises(ValueError, match='denoise masks'):
            guider.sample(noise, noise, make_sampler(controller), ptd_sigmas(controller.recipe),
                           denoise_mask=torch.ones_like(noise), disable_pbar=True, seed=42)
        assert tiny.calls == []
        assert controller.active is None
    finally:
        core.management.unload_all_models()
