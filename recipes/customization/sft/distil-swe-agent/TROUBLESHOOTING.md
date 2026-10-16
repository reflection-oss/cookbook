# Troubleshooting

- **The fine-tuned model repeats itself until the output limit**: too large a share of rollouts in the data. Every student we trained only on traces rolled out through the evaluation harness did this; rebuild the mix with the corpus traces as the majority.
- **Out of memory**: `NUM_GPUS=4` doubles gradient accumulation at the same effective batch, or `MAX_LEN=32768` leaves the longest conversations out.
- **`EXPORT_FAILED`**: export the other epoch's checkpoint with `python export.py --src runs/e4b-distil/checkpoints/checkpoint-<step> --base models/gemma-4-E4B-it --out runs/e4b-distil/export`. In our runs the in-training checkpoints were always clean.
- **The server rejects the export's `config.json`**: a serving stack with an older Transformers than the training node. `export.py` copies the base model's config over the export; if you exported by hand, do the same.
- **Tempted to change the hyperparameters**: on our 60-environment subset, 3 epochs, learning rate 2e-5, a 1,000-character thought cap and uncapped thoughts were no better than the defaults, learning rate 3e-6 was clearly worse, and adding all 645 passing rollouts or weighting them twice did not help.
