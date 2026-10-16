# RL post-training

Run a bounded task with a checkable reward, save the result, and evaluate improvements, regressions, and reward exploitation.

## Supported stacks

Beam can be customized with the open-source training stacks below. Each stack's guide covers the task, reward, hardware, training configuration, evaluation, and checkpoint resume.

| Stack | Best for | Guide |
| --- | --- | --- |
| [Miles](https://miles.radixark.com/docs/models/reflection/beam) | Full-parameter and LoRA math RL with Megatron training and SGLang rollouts, using separate or shared training and rollout GPUs | [Beam math RL](miles/README.md) |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/beam-support/docs/guides/models/reflection/beam.md) | Full-parameter GRPO with AutoModel/SGLang on shared GPUs or Megatron/vLLM on dedicated GPUs | [Beam math RL](nemo-rl/README.md) |
| [SkyRL](https://docs.skyrl.ai/docs/examples/beam) | LoRA and full-parameter GRPO/DAPO with Megatron and vLLM, including MXFP8 | [Beam math RL](skyrl/README.md) |

The Miles guide covers dedicated and colocated full-parameter training and dedicated LoRA training, with AIME2025 results from the 96-H200 dedicated full-parameter run.

The NeMo RL guide covers shared-GPU configurations on 128 H200s and 64 B300s, a dedicated 128-H200 configuration, and AIME2025 results from the 128-H200 shared-GPU run.

The SkyRL guide covers LoRA on 16 or 32 B200 GPUs and BF16 or MXFP8 full fine-tuning on 64 B200 GPUs.

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) when adding a recipe.
