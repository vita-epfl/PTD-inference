# Release notes

## v0.1.0 release candidate

The `release/comfyui-inference` branch of [vita-epfl/PTD-inference](https://github.com/vita-epfl/PTD-inference) contains both standalone inference and native ComfyUI integration. See the [README](../README.md) for the included interfaces.

### Included

- The original standalone `ptd` inference package, FSDP2 launcher, four-step and six-step recipes, text/image-condition encoders and audio/video output.
- Native ComfyUI `MODEL`, `SAMPLER` and `SIGMAS` nodes for PTD hop adapters.
- CPU conversion tools for the original unpruned MiniMax-H3 base and trusted PTD DCP checkpoints.
- A standard UI workflow for four-step text-to-video with audio.
- Fixed recipe grids, per-hop expert routing, the PTD audio clock and float32 state carry.

The validated native ComfyUI scope is **four-step text-to-video**. Six-step and image-conditioned ComfyUI generation remain unvalidated on the full model. The existing standalone modes are retained; this validation does not qualify every standalone recipe or conditioning mode. The validated ComfyUI launch uses `--disable-dynamic-vram`; DynamicVRAM is unqualified. The compatibility baseline is [ComfyUI `5cc4b870`](https://github.com/Comfy-Org/ComfyUI/commit/5cc4b870094c89090db54bd8d12649dcaf89f2ba).

### Validation status

The supplied workflow passed full-model execution through the ComfyUI HTTP queue, with all 16 computational nodes executed: native text encoding, four-step PTD sampling, both native VAE decoders and Save Video. It produced 124 frames at 1344 × 768 and 24 FPS with 32 kHz stereo audio. Backend graph execution took 92.67 seconds, including in-graph model loading through saving; downloads and backend startup are excluded.

Media checks confirmed nonblack video, nonsilent audio and matching audio/video durations; three sampled frames were visually reviewed. Manual browser interaction and listening evaluation were not performed. GitHub CI passed 122 tests with no skips. Six-step and image-conditioned full-model generation remain unvalidated.

Numerical equivalence with standalone PTD is not claimed. See the [setup guide](COMFYUI.md#validation-and-resource-use) for the validation boundary and measured resource use.

### Distribution

For command-line use, follow [Standalone inference](INFERENCE.md). For ComfyUI, clone the same branch into `ComfyUI/custom_nodes/` and follow the [ComfyUI guide](COMFYUI.md). A standalone wheel does not install the complete custom-node checkout. Comfy Registry publication is not included in this release.

Model files are distributed separately. PTD weights are hosted at [epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights). The release retains MiniMax/Hugging Face attribution and EPFL VITA Lab's PTD adaptation and checkpoint-training attribution; see [NOTICE](../NOTICE) and [LICENSE](../LICENSE).
