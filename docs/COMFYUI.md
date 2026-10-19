# ComfyUI integration

This release supports four-step text-to-video generation through ComfyUI's native MiniMax-H3 backend. The supplied workflow has passed full-model execution from native text encoding through audio/video decoding and saving via the ComfyUI HTTP queue. Manual browser interaction was not tested. Six-step and image-conditioned generation are implemented but remain unvalidated on the full model.

## Installation

Use Python 3.10 or newer and a ComfyUI build with native MiniMax-H3 support. The compatibility baseline is [ComfyUI commit `5cc4b870`](https://github.com/Comfy-Org/ComfyUI/commit/5cc4b870094c89090db54bd8d12649dcaf89f2ba); later revisions have not been qualified for this release.

From the ComfyUI directory, clone the combined release branch and install using the Python environment that starts ComfyUI:

```sh
git clone --branch release/comfyui-inference --single-branch \
    https://github.com/vita-epfl/PTD-inference.git custom_nodes/ComfyUI-PTD
python -m pip install -e ./custom_nodes/ComfyUI-PTD
```

The validated GPU configuration uses estimate-based model loading. Start ComfyUI with `python main.py --disable-dynamic-vram`; DynamicVRAM has not been qualified for this release.

After startup, the `PTD` category should contain **Load PTD MiniMax-H3 Adapters**, **PTD Sampler**, and **PTD Sigmas**. This installation uses the checkout as a custom-node directory; installing the Python package alone does not register nodes in ComfyUI. The native path does not need the standalone Diffusers dependencies in `requirements.txt`. The same checkout also contains the [standalone inference scripts](INFERENCE.md), which use the optional `.[inference]` extra.

## Model files

PTD requires the original unpruned MiniMax-H3 transformer. Pruned, PDD, Turbo, AdaLN-curve and quantized transformer variants are not supported bases for these adapters.

The first two filenames below are example names for files created by the conversion commands. The text encoder and VAE filenames come from the [Comfy-Org text encoder directory](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders) and [VAE directory](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6/vae).

| ComfyUI directory | Filename |
| --- | --- |
| `models/diffusion_models/` | `minimax_h3_original.safetensors` |
| `models/ptd/` | `ptd_h3_4step.safetensors` |
| `models/text_encoders/` | `qwen3vl_32b_minimax_h3_bf16.safetensors` |
| `models/vae/` | `minimax_h3_video_vae_fp16.safetensors` |
| `models/vae/` | `minimax_h3_audio_vae_fp32.safetensors` |

Set the CLIP loader's type to `minimax` and the diffusion-model loader's weight dtype to `default`. The native text encoder and VAEs were validated at Comfy-Org model revision `e5eb578a89295337b8ff433a035929ce0279e0b6`. Model files are distributed separately; download the PTD checkpoint from [epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights).

### Convert the original base

From `ComfyUI/custom_nodes/ComfyUI-PTD`, stage the `transformer/` directory from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), revision `42ed227ee7df40d41602854ae760620d6eb651fe`, under `models/MiniMax-H3/`. Include its config, index and every indexed safetensors shard. Then run:

```sh
python scripts/convert_comfy_base.py \
    models/MiniMax-H3/transformer \
    ../../models/diffusion_models/minimax_h3_original.safetensors
```

Conversion preserves tensor precision and changes the parameter names/layout for native ComfyUI. It runs on CPU and refuses an existing output. Allow enough host RAM for the loaded transformer plus conversion intermediates, and disk space for the source shards and converted file.

### Convert the four-step PTD checkpoint

Download `ptd_h3_4step/` from the PTD weights repository into this checkout's `weights/` directory, keeping `.metadata` and every DCP shard. From the custom-node repository directory:

```sh
python scripts/convert_comfy_ptd.py \
    --checkpoint weights/ptd_h3_4step \
    --recipe h3_4step \
    --output ../../models/ptd/ptd_h3_4step.safetensors \
    --trust-dcp
```

`--trust-dcp` acknowledges that PyTorch DCP metadata is pickle-based; convert only a checkpoint whose source you trust. Runtime loading uses the resulting safetensors file. The converter keeps the adapter factors separate from the base. Use the PTD loader rather than a generic LoRA loader, because the active expert changes at every hop.

The converter also accepts `h3_6step` and its matching checkpoint. Six-step generation is outside the full-model validation completed for this release.

## Workflow

Drag [ptd_h3_t2v.json](../examples/comfyui/ptd_h3_t2v.json) onto the ComfyUI canvas and select the files above. The example uses 1344 × 768, 124 frames and 24 FPS, with audio. Keep the first/last-frame inputs disconnected for text-to-video.

1. **Load Diffusion Model → Load PTD MiniMax-H3 Adapters** creates the adapted native `MODEL`.
2. **MiniMax H3 Image to Video** receives the native CLIP and video VAE and produces positive conditioning plus an empty audio/video latent. With no input images, this node performs text-to-video setup.
3. Connect the same PTD `MODEL` to **Basic Guider**, **PTD Sampler** and **PTD Sigmas**. Send their outputs, **Random Noise**, and the empty latent into **SamplerCustomAdvanced**.
4. Send **SamplerCustomAdvanced's `output`** to **Separate AV Latent**. Decode the video with **VAE Decode** and the audio with **VAE Decode Audio**, using their matching VAEs.
5. Combine both streams in **Create Video** at 24 FPS, then use **Save Video**. The example prefix is `PTD/h3_4step` under ComfyUI's output directory.

**Basic Guider** applies no extra classifier-free guidance. Keep the complete PTD step schedule and an empty generated latent. Generic schedulers, truncated schedules, denoise masks and nonempty initial generated latents are unsupported. Image conditioning is not part of the validated release scope.

The PTD sampler selects one expert per hop on the checkpoint's fixed grid. It uses the shared clock ending at `0.999`, applies the corresponding audio clock, and carries inter-hop state in float32. ComfyUI controls the backbone's inference dtype and model loading/offloading.

## Validation and resource use

The supplied workflow executed all **16 computational nodes** through the ComfyUI HTTP queue: native text encoder → four-step PTD sampling → native video and audio VAEs → Save Video. The saved output contains **124 frames at 1344 × 768 and 24 FPS**, with **32 kHz stereo audio**. The frontend HTML endpoint also loaded successfully. This validates the backend graph; it was not a manual browser test.

Decoded media checks passed for frame dimensions/count/rate, nonblack video, nonsilent audio and matching audio/video durations. Frames 0, 62 and 123 were visually checked against the stream-at-sunrise prompt. No listening evaluation was performed.

Backend graph execution took **92.67 seconds**, including model loading within the graph, text encoding, sampling, decoding and saving. Downloads and backend startup are excluded. A separate sampler-only fixture on one GH200 measured **44.1 seconds**, including its first GPU transfer, with **69.7 GiB** peak PyTorch allocation and **77.0 GiB** reserved; that measurement excluded model-file loading and decoding. These measurements are not warm-run benchmarks or minimum-memory requirements. The environment used BF16 transformer blocks, PyTorch `2.10.0a0+b558c986e8.nv25.11`, CUDA 13.0 and the ComfyUI baseline above.

Five additional prompt comparisons against standalone PTD used cached text conditioning and the original VAE decoders. Numerical equivalence to standalone PTD is not claimed, and equal seeds need not produce identical outputs. Six-step and image-conditioned full-model generation remain unvalidated.

The CPU suite contains **122 tests**, covering conversion, adapter routing, both schedules and numerical checks, including the real ComfyUI sampler chain with a small synthetic network. The [CI workflow](../.github/workflows/tests.yml) requires every test to run without skips. The actual custom-node loader registers all three nodes; the workflow's 17 nodes, including its explanatory Note, and 20 connections match the tested schemas.

To run CPU tests from the checkout:

```sh
python -m pip install -e '.[test]'
PYTHONPATH=../.. python -m pytest tests/test_comfy_*.py -q
```

Tests requiring ComfyUI or Diffusers skip when those packages are unavailable. A minimal native installation therefore does not run every cross-implementation test. These tests do not replace a full-model workflow check.
