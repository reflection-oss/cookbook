# Customize Beam

Beam's weights are open, so you can make it yours. These recipes show the two ways to do it:

- **Adapt Beam itself** to your data and tasks with LoRA, full fine-tuning, or reinforcement learning (RL).
- **Use Beam as a teacher** to train a smaller model that is cheaper to serve.

Each recipe is a complete run, not a sketch. It uses public or synthetic data and the configurations and scripts we ran, and it records the hardware and runtime. Recipes that train for quality report before and after results, both measured the same way.

## Recipes

| Recipe | What it does |
| --- | --- |
| [Self-distillation](sft/sft_automodel_self_distillation/) | Beam fine-tunes on its own verified Triton kernels, with no teacher model and no human-written kernels |
| [SWE agent distillation](sft/distil-swe-agent/) | Gemma 4 E4B learns from Beam's passing traces in synthetic software-engineering environments |
| [Tool-calling reasoning](sft/reasoning-distillation/) | Gemma 4 E4B learns to think briefly before it calls a tool, from thoughts Beam wrote |
| [RL with Miles](rl/miles/) | GRPO on DAPO-Math-17K, full-parameter |
| [RL with NeMo RL](rl/nemo-rl/) | GRPO on DAPO-Math-17K, full-parameter |
| [RL with SkyRL](rl/skyrl/) | GRPO on DAPO-Math-17K, LoRA and full-parameter |

The training data for the SWE agent and tool-calling recipes comes from [`data-synthesis`](data-synthesis/). The self-distillation recipe generates its own data in its folder. In `data-synthesis`, one recipe has Beam write tool-calling thoughts for 3,941 public conversations. The other has Beam build software-engineering environments with hidden tests and solve them, and keeps only the trajectories that pass.

## A path from a low compute budget to full RL

The recipes form a ladder. Each step adds capability, and compute, over the one before it. Stop at the step that meets your quality bar:

1. **Generate data with Beam as the teacher.** Turn a small set of seed prompts or tasks into a training set. This spends inference compute, not training compute, and the result can train Beam or a smaller model.
2. **LoRA or QLoRA fine-tuning.** Adapt Beam on a smaller compute footprint by training adapters instead of the full model.
3. **Full fine-tuning.** When you need further quality gains and have the compute, train all parameters.
4. **RL post-training.** Optimize Beam against a verifiable reward for your task. This is the most capable step and the most compute-intensive.

The pipeline stays the same as you climb: prepare data, train, evaluate against the baseline. Only the training step changes.

### Compute budget

Sizing for Beam-1.0-501B-A23B (501B total parameters, 23B active). A node is 8x H200: 141 GB of HBM per GPU, about 1.1 TB per node.

| Step | Recipe | Trainable parameters | Compute footprint |
| --- | --- | --- | --- |
| Data generation | [`data-synthesis`](data-synthesis/) | None | Inference only, via the API or a self-hosted deployment |
| QLoRA, NF4 base | [`sft`](sft/) | ~29M to ~3.6B | 1x H200 node (measured) |
| LoRA, NVFP4 base | [`sft`](sft/) | ~29M to ~3.6B | 1x H200 node |
| LoRA, FP8 base | [`sft`](sft/) | ~29M to ~3.6B | 1x H200 node |
| LoRA, BF16 base | [`sft`](sft/) | ~29M to ~3.6B | 2x H200 nodes |
| Full fine-tuning | [`sft`](sft/) | All 501B | 3 to 8 H200 nodes |
| RL post-training | [`rl`](rl/) | All 501B | 4 to 8 H200 nodes |

Notes on the table:

- **Data generation** is not a training job. Call the API, or serve the model yourself with vLLM or SGLang.
- **Trainable parameters** for the adapter rows run from about 29M with adapters on the attention projections only, to about 3.6B with adapters on the attention projections and the fused expert parameters.
- **QLoRA** keeps the base weights in NF4, about 250 GB, so the whole model fits on one node with room for activations and the adapter optimizer.
- **NVFP4** matches the QLoRA footprint. FP4 tensor cores are native on Blackwell; on H200 the weights are dequantized for each matmul, so it fits like QLoRA but does not run faster.
- **FP8** halves the BF16 model footprint to about 0.5 TB and runs natively on Hopper and Blackwell FP8 tensor cores.
- **BF16 LoRA** keeps a frozen base of about 1.0 TB, which alone fills a node before activations and adapter states, so plan on two nodes.
- **Full fine-tuning** holds parameters, gradients, and optimizer states for all 501B parameters. Plan on three to eight nodes, depending on optimizer precision and how the stack shards the model.
- **RL post-training** holds the policy training state plus rollout generation and a reference model, so plan on four to eight nodes.
- **All training rows except QLoRA are estimates** from the memory footprint. Measured numbers replace them as each recipe is verified.

## What each folder provides

| Folder | What you get |
| --- | --- |
| [`data-synthesis`](data-synthesis/) | Beam as the teacher: recipes that turn seed prompts or tasks into verified training data for SFT and RL |
| [`sft`](sft/) | Supervised fine-tuning of Beam, and distillation of Beam into smaller models, with data preparation, training configuration, evaluation and export |
| [`rl`](rl/) | RL post-training with a verifiable reward, with reward design, training configuration, evaluation and checkpoint resume |

## Integration stacks

Beam is supported in the open-source training stacks below. Where a stack has its own Beam guide or examples, the link goes there. The recipes above use NeMo AutoModel, TRL, Miles, NeMo RL and SkyRL.

| Method | Stack | Best for |
| --- | --- | --- |
| SFT | [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/large-language-models/reflection/Beam-1.0-501B-A23B) | Hugging Face checkpoints with NVIDIA-optimized training, LoRA and full-parameter |
| SFT | [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/beam) | Large-scale full fine-tuning with Megatron-Core parallelism, with conversion to and from Hugging Face format |
| SFT | [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/beam) | Config-driven LoRA, QLoRA, and full fine-tuning with broad dataset and adapter support |
| SFT | [TRL](https://github.com/huggingface/trl) | Full fine-tuning of small open models such as Gemma 4 on Beam-generated data, with FSDP on one node |
| RL | [Miles](https://miles.radixark.com/docs/models/reflection/beam) | PyTorch-native RL post-training with SGLang rollouts and Megatron-LM or FSDP training, focused on MoE stability and low-precision training |
| RL | [NeMo RL](https://github.com/NVIDIA-NeMo/RL/blob/beam-support/docs/guides/models/reflection/beam.md) | Scalable RL post-training on NVIDIA hardware, from a single node to large clusters |
| RL | [SkyRL](https://docs.skyrl.ai/docs/examples/beam) | Modular RL for agentic and multi-turn tasks, with pluggable environments and inference backends |

More recipes land here as they are built and verified end to end. To contribute one, see [CONTRIBUTING.md](../../CONTRIBUTING.md).
