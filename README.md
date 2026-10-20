# Reflection Cookbook

Complete, runnable examples for customizing Reflection models: data generation, supervised fine-tuning, and RL post-training.

## How to use this repo

Every recipe lives in its own folder under `recipes/` with its own README, dependencies, sample input, expected output, and smoke test. Pick one and follow its README.

| Area | What you'll find |
| --- | --- |
| [`recipes/customization/`](recipes/customization/) | Overview of the customization recipes and the supported SFT and RL stacks |
| [`recipes/customization/data-synthesis/`](recipes/customization/data-synthesis/) | Data generation recipes for model customization, with Beam as the teacher |
| [`recipes/customization/sft/`](recipes/customization/sft/) | Supervised fine-tuning, with baseline comparison |
| [`recipes/customization/rl/`](recipes/customization/rl/) | RL post-training with a verifiable reward |

To add a recipe, see [CONTRIBUTING.md](CONTRIBUTING.md). If you use an AI coding assistant, point it at [`skills/community-contributions/SKILL.md`](skills/community-contributions/SKILL.md) and it will walk you through the process.

## Links

- Model weights and model card: [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B)

## Contact us

Email us at [developers@reflection.ai](mailto:developers@reflection.ai) or join our [Discord](https://discord.com/invite/reflectionai).

## License

Code in this repository is licensed under the [Apache 2.0 License](LICENSE). Model weights are covered by their own license.
