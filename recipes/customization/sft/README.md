# Supervised fine-tuning

Prepare data, fine-tune, save or export the model, and compare it with the baseline. Reports the task metric before and after, hardware, and runtime.

## Supported stacks

Beam works with the open-source training stacks below. Where a stack has its own Beam guide or examples, the link goes there, and the recipes in this folder show complete runs. For a smaller compute footprint, start with LoRA or QLoRA. When you need further quality gains and have the compute, move to full fine-tuning.

| Stack | Best for |
| --- | --- |
| [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/large-language-models/reflection/Beam-1.0-501B-A23B) | Hugging Face checkpoints with NVIDIA-optimized training, LoRA and full-parameter |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/beam) | Config-driven LoRA, QLoRA, and full fine-tuning with broad dataset and adapter support |
| [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/beam) | Large-scale full fine-tuning with Megatron-Core parallelism, with conversion to and from Hugging Face format |
| [TRL](https://github.com/huggingface/trl) | Full fine-tuning of small open models such as Gemma 4 on Beam-generated data, with FSDP on one node |

## Recipes

| Recipe | What it does |
| --- | --- |
| [`distil-swe-agent/`](distil-swe-agent/) | Full fine-tuning of Gemma 4 E4B with FSDP on Beam's golden traces from synthetic SWE environments, on one node, roughly doubling the held-out environments solved |
| [`reasoning-distillation/`](reasoning-distillation/) | Full fine-tuning of Gemma 4 (E4B, with E2B and 12B shown to generalise) with TRL and FSDP on tool-calling data with Beam-written reasoning, with BFCL v4 scores before and after |
| [`sft_automodel_self_distillation/`](sft_automodel_self_distillation/) | Self-distillation of Beam on Triton kernel writing: verified own samples as SFT data, full-parameter fine-tuning with NeMo AutoModel on 8 nodes of 8 H200, pass@1 before and after; ships the task corpus and the data-generation scripts |
