"""Full fine-tune of a Gemma 4 checkpoint with TRL's SFTTrainer under FSDP.

    accelerate launch --config_file fsdp.yaml train.py \
      --model <base> --dataset <prepared> --output <dir>

The defaults are the recipe: learning rate 3e-6, one epoch, cosine schedule
with 3% warmup, effective batch 32 on 8 GPUs, bf16, gradient checkpointing.
The vision and audio towers are frozen because the data is text only.
Checkpoints hold the model only; export.py reads the end-of-epoch one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import datasets
import torch
import transformers
import trl

TRAINED_COLUMNS = ('input_ids', 'attention_mask', 'labels')
FROZEN_TOWERS = ('vision', 'audio', 'embed_vision', 'embed_audio')


def load_split(path: str | None) -> datasets.Dataset | None:
  """Load a prepared split with only the columns the collator uses."""
  if not path or not os.path.isdir(path):
    return None
  dataset = datasets.load_from_disk(path)
  extra = [c for c in dataset.column_names if c not in TRAINED_COLUMNS]
  return dataset.remove_columns(extra)


def main() -> None:
  """Train and write trainer_state.json and train_info.json to --output."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', required=True)
  parser.add_argument('--dataset', required=True)
  parser.add_argument('--eval-dataset', default=None)
  parser.add_argument('--output', required=True)
  parser.add_argument('--epochs', type=float, default=1.0)
  parser.add_argument('--lr', type=float, default=3e-6)
  parser.add_argument('--per-device-bs', type=int, default=2)
  parser.add_argument('--grad-accum', type=int, default=2)
  parser.add_argument('--warmup-ratio', type=float, default=0.03)
  parser.add_argument('--attn', default='sdpa')
  parser.add_argument('--logging-steps', type=int, default=5)
  parser.add_argument('--seed', type=int, default=44)
  args = parser.parse_args()
  train_ds, eval_ds = load_split(args.dataset), load_split(args.eval_dataset)
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
  model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation=args.attn
  )
  model.config.use_cache = False
  frozen = 0
  for name, param in model.named_parameters():
    if any(tower in name for tower in FROZEN_TOWERS):
      param.requires_grad_(False)
      frozen += param.numel()
  world = int(os.environ.get('WORLD_SIZE', '1'))
  effective_batch = args.per_device_bs * world * args.grad_accum
  total_steps = math.ceil(
    math.ceil(len(train_ds) / effective_batch) * args.epochs
  )
  warmup_steps = max(1, round(args.warmup_ratio * total_steps))
  if int(os.environ.get('RANK', '0')) == 0:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
      f'model: {type(model).__name__} | trainable {trainable / 1e9:.2f}B | '
      f'frozen towers {frozen / 1e9:.2f}B | examples {len(train_ds)} | '
      f'world {world} | steps {total_steps} | warmup {warmup_steps} | '
      f'effective batch {effective_batch}'
    )
  config = trl.SFTConfig(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type='cosine',
    warmup_steps=warmup_steps,
    per_device_train_batch_size=args.per_device_bs,
    gradient_accumulation_steps=args.grad_accum,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=args.logging_steps,
    save_strategy='epoch',
    save_total_limit=3,
    save_only_model=True,
    eval_strategy='epoch' if eval_ds is not None else 'no',
    per_device_eval_batch_size=2,
    report_to='none',
    seed=args.seed,
    dataset_kwargs={'skip_prepare_dataset': True},
    remove_unused_columns=False,
    packing=False,
    max_length=None,
    dataloader_num_workers=2,
    weight_decay=0.0,
    max_grad_norm=1.0,
  )
  trainer = trl.SFTTrainer(
    model=model,
    args=config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=transformers.DataCollatorForSeq2Seq(
      tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
    ),
    processing_class=tokenizer,
  )
  start = time.time()
  trainer.train()
  if trainer.is_world_process_zero():
    trainer.save_state()
    info = {'train_minutes': round((time.time() - start) / 60, 1)}
    info['args'] = vars(args)
    pathlib.Path(args.output, 'train_info.json').write_text(
      json.dumps(info, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
  main()
