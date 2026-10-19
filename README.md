# PTD Inference

Standalone inference and native ComfyUI support for MiniMax-H3 with PTD.
MiniMax developed MiniMax-H3 and its upstream inference code. **EPFL VITA Lab** made the PTD-specific adaptations and trained the checkpoints for *Parametric Trajectory Distillation for Few-Step Video Generation*.

## Usage

Both interfaces are included in the `release/comfyui-inference` branch:

| Interface | Included code | Guide |
| --- | --- | --- |
| Command line | Original inference package, multi-GPU launcher, prompt and image-condition encoders | [Standalone inference](docs/INFERENCE.md) |
| ComfyUI | Native PTD nodes, checkpoint converters and an audio/video workflow | [ComfyUI setup](docs/COMFYUI.md) |

The standalone path retains the four-step and six-step recipes and image-conditioning support. The native ComfyUI four-step text-to-video workflow has passed full-model validation; six-step and image-conditioned ComfyUI generation remain unvalidated on the full model. See [release notes](docs/RELEASE_NOTES.md) for the validation scope.

Model files are distributed separately. Both paths use the original MiniMax-H3 base and the VITA-trained checkpoints at [epfl-vita/PTD-weights](https://huggingface.co/epfl-vita/PTD-weights).

## License

Code: [Apache 2.0](LICENSE). [NOTICE](NOTICE) records MiniMax/Hugging Face attribution and EPFL VITA Lab's adaptations. The base model, PTD weights and outputs are covered by the MiniMax H3 Community License Agreement supplied with the weights.
