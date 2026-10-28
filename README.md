# PTD Inference

Inference code for **Parametric Trajectory Distillation for Few-Step Video Generation (PTD)**, with standalone and native ComfyUI support for MiniMax-H3.

[Project page](https://alan-lanfeng.github.io/PTD/) · [Paper (arXiv)](https://arxiv.org/abs/2610.11498) · [Model weights](https://huggingface.co/epfl-vita/PTD-weights)

MiniMax developed MiniMax-H3 and its upstream inference code. **EPFL VITA Lab** made the PTD-specific adaptations and trained the checkpoints.

## Usage

Both interfaces are included in the `release/comfyui-inference` branch:

| Interface | Included code | Guide |
| --- | --- | --- |
| Command line | Original inference package, multi-GPU launcher, prompt and image-condition encoders | [Standalone inference](docs/INFERENCE.md) |
| ComfyUI | Native PTD nodes, checkpoint converters and an audio/video workflow | [ComfyUI setup](docs/COMFYUI.md) |

The standalone path provides the re-noised recipes `h3_4step_sde` and `h3_6step_sde` (the recipes of the checkpoints at [epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights)), the paper's Euler-transition recipes `h3_4step` and `h3_6step`, and image conditioning. The native ComfyUI four-step text-to-video workflow has passed full-model validation with the paper's four-step checkpoint; the re-noised transitions of the `_sde` adapters are implemented in the ComfyUI sampler and covered by the CPU tests, but, like six-step and image-conditioned ComfyUI generation, remain unvalidated on the full model. See [release notes](docs/RELEASE_NOTES.md) for the validation scope.

Model files are distributed separately. Both paths use the original MiniMax-H3 base and the VITA-trained checkpoints at [epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights).

## Citation

If you use PTD in your research, please cite:

```bibtex
@article{feng2026ptd,
  title   = {Parametric Trajectory Distillation for Few-Step Video Generation},
  author  = {Feng, Lan and Karkus, Peter and Igl, Maximilian and Berner, Julius and Chen, Yuxiao and Tan, Shuhan and Alahi, Alexandre and Ivanovic, Boris and Pavone, Marco},
  journal = {arXiv preprint arXiv:2610.11498},
  year    = {2026}
}
```

## License

Code: [Apache 2.0](LICENSE). [NOTICE](NOTICE) records MiniMax/Hugging Face attribution and EPFL VITA Lab's adaptations. The base model, PTD weights and outputs are covered by the MiniMax H3 Community License Agreement supplied with the weights.
