"""Pre-tokenized JSONL dataset for NeMo AutoModel SFT.

Rows are {"id", "input_ids", "loss_mask"}: the exact token ids the sampled model consumed and produced, rendered once
by build_sft_rows.py with the checkpoint's chat template at reasoning_effort=max (rendered prompt == the server's
prompt tokens). Using them directly removes any re-rendering risk (template kwargs, header lines, reasoning channel)
from the training job.

Emits AutoModel's standard shifted next-token format (formatting_utils._package_tokenized_example):
  input_ids = ids[:-1];  labels[i] = ids[i+1] if loss_mask[i+1] else -100;  attention_mask = 1s.
No `loss_mask` key is emitted: pack_dataset would re-apply it index-wise to the already shifted labels.

Recipe usage (the module directory must be on PYTHONPATH; the Job mounts it at /config):
  dataset:
    _target_: pretok_dataset.PreTokenizedJsonlDataset
    path: data/self-distillation/train.tok.jsonl
    seq_length: 65536      # a longer row is an error here, never a silent drop in the packer
"""

import json

import numpy as np
import torch

IGNORE = -100


class PreTokenizedJsonlDataset(torch.utils.data.Dataset):
  def __init__(self, path, seq_length=None, limit=None, tokenizer=None):  # noqa: ARG002
    # tokenizer: accepted (AutoModel forwards the runtime tokenizer to targets that declare it) but unused: rows are token ids
    self.path, self.seq_length = path, seq_length
    self.ids, self.rows = [], []
    with open(path, encoding='utf-8') as f:
      for line in f:
        if not line.strip():
          continue
        r = json.loads(line)
        ids, m = r['input_ids'], r['loss_mask']
        rid = r.get('id')
        if len(ids) != len(m) or len(ids) < 2:
          raise ValueError(f'{path}:{rid}: input_ids/loss_mask length mismatch')
        if m[0] != 0 or m[-1] != 1:
          raise ValueError(
            f'{path}:{rid}: expected a prompt first and a supervised last token'
          )
        if seq_length is not None and len(ids) - 1 > seq_length:
          raise ValueError(
            f'{path}:{rid}: {len(ids)} tokens exceed seq_length {seq_length}'
          )
        self.ids.append(rid)
        self.rows.append((
          np.asarray(ids, dtype=np.int64),
          np.asarray(m, dtype=np.int8),
        ))
        if limit and len(self.rows) >= limit:
          break
    if not self.rows:
      raise ValueError(f'{path}: no rows')

  def __len__(self) -> int:
    return len(self.rows)

  def __getitem__(self, i) -> dict:
    ids, m = self.rows[i]
    labels = np.where(m[1:] != 0, ids[1:], IGNORE)
    return {
      'input_ids': ids[:-1].tolist(),
      'labels': labels.tolist(),
      'attention_mask': [1] * (len(ids) - 1),
    }

  def __iter__(self):
    for i in range(len(self.rows)):
      yield self[i]
