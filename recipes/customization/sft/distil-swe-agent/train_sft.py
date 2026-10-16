"""Full fine-tune of Gemma 4 on the tokenized dataset, launched by train.sh.

The defaults are the recipe: 2 epochs, learning rate 1e-5, cosine schedule
with 3% warmup, one conversation per GPU per step and an effective batch of
32, bf16, gradient checkpointing. The vision and audio towers are frozen
(text-only data). One weights-only checkpoint is saved per epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch
from datasets import load_from_disk
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
)
from trl import SFTConfig, SFTTrainer

FROZEN_TOWERS = ('vision', 'audio')


def main() -> None:
  """Train and write `log_history.json` next to the checkpoints."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', required=True, help='base model folder')
  parser.add_argument('--data', type=pathlib.Path, required=True)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.add_argument('--epochs', type=float, default=2.0)
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--grad-accum', type=int, default=4)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()

  dataset = load_from_disk(str(args.data / 'dataset'))
  eval_dataset = None
  if (args.data / 'dataset_val').is_dir():
    eval_dataset = load_from_disk(str(args.data / 'dataset_val'))
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16, attn_implementation='sdpa'
  )
  model.config.use_cache = False
  for name, param in model.named_parameters():
    if any(tower in name for tower in FROZEN_TOWERS):
      param.requires_grad_(False)

  world = int(os.environ.get('WORLD_SIZE', '1'))
  config = SFTConfig(
    output_dir=str(args.output),
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type='cosine',
    warmup_steps=0.03,  # a float below 1 is a ratio of the total steps
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.grad_accum,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=5,
    save_strategy='epoch',
    save_only_model=True,
    eval_strategy='epoch' if eval_dataset is not None else 'no',
    report_to='none',
    seed=args.seed,
    dataset_kwargs={'skip_prepare_dataset': True},
    remove_unused_columns=False,
    packing=False,
    max_length=None,
    weight_decay=0.0,
    max_grad_norm=1.0,
  )
  trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(
      tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
    ),
    processing_class=tokenizer,
  )
  if trainer.is_world_process_zero():
    print(
      f'rows {len(dataset)} | gpus {world} | batch {world * args.grad_accum}'
    )
  trainer.train()
  if trainer.is_world_process_zero():
    (args.output / 'log_history.json').write_text(
      json.dumps(trainer.state.log_history, indent=1), encoding='utf-8'
    )


if __name__ == '__main__':
  main()
