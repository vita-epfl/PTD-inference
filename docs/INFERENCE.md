# Standalone inference

This branch retains the original MiniMax-H3 inference code with EPFL VITA Lab's PTD adaptations. It runs independently of ComfyUI and includes four-step and six-step recipes, cached text conditioning, image conditioning and MP4 output with audio.

## Setup

Use Python 3.10 or newer with a CUDA-enabled PyTorch installation. The standalone launcher uses FSDP2 across NVIDIA GPUs and defaults to four processes. Set `NPROC` to the number of GPUs you intend to use, with sufficient memory for the 33B model. The ComfyUI memory measurements do not establish memory requirements for this path.

Clone the combined release branch and install its standalone dependencies:

```bash
git clone --branch release/comfyui-inference --single-branch \
    https://github.com/vita-epfl/PTD-inference.git
cd PTD-inference
python -m pip install -e '.[inference]'

hf auth login
hf download epfl-vita/PTD-weights --local-dir weights
hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
    --local-dir models/MiniMax-H3
```

Access to the PTD weights repository is required while it is private. Keep the checkpoint directories, including `.metadata` and every DCP shard, intact. The standalone path loads these checkpoints directly; ComfyUI conversion is unnecessary.

| Checkpoint directory | Recipe | Sampling steps |
| --- | --- | :---: |
| `weights/ptd_h3_4step/` | `h3_4step` | 4 |
| `weights/ptd_h3_6step/` | `h3_6step` | 6 |

Verify a checkpoint with `(cd weights && sha256sum -c checksums/ptd_h3_6step.files.sha256)`; replace `6step` with `4step` for the other checkpoint.

## Text-to-video

Store prompts as a JSON list of strings or objects with a `prompt` field. JSON preserves MiniMax-H3's multiline prompt format. The launcher prepares and reuses the condition cache, then writes MP4 files with audio.

```bash
bash scripts/sample.sh --recipe h3_6step --ckpt weights/ptd_h3_6step --model models/MiniMax-H3 \
    --prompts prompts.json --cond-cache cond_cache --output samples/h3_6step
```

For four-step inference, use `--recipe h3_4step` and `--ckpt weights/ptd_h3_4step`. Use `--seed` to offset the per-GPU seeds and `--save-latents` to retain final latents. See all sampling options with `python -m ptd.sample --help`.

## Image conditioning

Place `<key>.txt` and the corresponding images in a request directory:

- Image to video: `<key>.first.jpg`.
- First and last frame: `<key>.first.jpg` and `<key>.last.jpg`.
- Reference images: `<key>.ref1.jpg`, `<key>.ref2.jpg`, etc.

```bash
python scripts/encode_image_conditions.py --input-dirs requests/ --cache-dir cond_cache \
    --model models/MiniMax-H3 --write-prompts requests.json
```

Run the sampling command above with `--prompts requests.json` and the same condition cache. Sample reference-image requests separately and add `--transformer-subfolder transformer_ref`.

Cache keys use prompt text. Use separate caches for different conditions, or pass `--force` to `encode_image_conditions.py` when re-encoding the same prompt. Its `--force-canvas` option stretches mismatched keyframes to the recipe's canvas.

These existing inference modes are preserved. The newly completed native ComfyUI validation covers four-step text-to-video only; it does not qualify every standalone recipe or conditioning mode.
