# Data synthesis

Data generation recipes for model customization. Use Beam as the teacher to turn a small set of seed prompts or tasks into training data, then feed the result into the [SFT](../sft/) or [RL](../rl/) recipes.

## Recipes

| Recipe | What it does |
| --- | --- |
| [`tool-calling-reasoning/`](tool-calling-reasoning/) | Beam writes a short thought before every assistant turn of 3,941 public tool-calling conversations, giving a fine-tuning set that teaches small models to think before they call a tool |
| [`synthetic-environment-gen/`](synthetic-environment-gen/) | Beam writes software-engineering environments through TMax, keeps only those a hidden test can verify, then solves them and keeps the passing trajectories as golden traces for training a small model |
