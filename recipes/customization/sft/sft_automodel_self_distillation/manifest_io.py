"""Shared helpers for the self-distillation scripts: load the manifest and hash content."""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib


def load_manifest(path: str | pathlib.Path) -> dict:
  """Load `dataset_manifest.json` or its gzip-compressed form `dataset_manifest.json.gz`."""
  p = pathlib.Path(path)
  if p.suffix == '.gz':
    with gzip.open(p, 'rt', encoding='utf-8') as fh:
      return json.load(fh)
  with p.open(encoding='utf-8') as fh:
    return json.load(fh)


def sha256(data: str | bytes) -> str:
  """Hex sha256 of text (UTF-8) or bytes."""
  if isinstance(data, str):
    data = data.encode('utf-8')
  return hashlib.sha256(data).hexdigest()
