# Common fixes

Symptoms we hit while building and evaluating the [TRL fine-tuning recipe](README.md), with their causes and fixes.

| Symptom | Cause | Fix |
| --- | --- | --- |
| Out of memory in the first steps | the per-device batch does not fit | `bash train.sh --per-device-bs 1 --grad-accum 4`; the effective batch stays 32 |
| `stock thinking gate not found in the chat template` | a chat template the recipe does not know | use the revision `download.sh` pins |
| The export prints `SMOKE_FAIL` | the checkpoint generates non-text; we saw it once in a save made after training ended | run `export.py` again on the epoch checkpoint; if it fails again, repeat the training run |
| The serving image cannot load the exported `config.json` | newer `transformers` versions rewrite Gemma 4's config on save | keep `BASE_MODEL` on the pinned base model; the export copies its config files over the export |
| `model_type gemma4 is not recognized` | `transformers` predates Gemma 4 | install the pinned version from `requirements.txt` |
| The held-out loss does not fall below the first logged training loss | the data is not what the recipe expects | check that every assistant turn has `reasoning_content`, that tool calls are structured `tool_calls`, and that `DATA` is the file you think it is |
| `inference_errors` above zero in the analysis | a serving worker restarted mid-entry and BFCL recorded `Error during inference`, which scores as wrong | remove those entries (snippet below), run `eval_run.sh` again; `bfcl generate` regenerates only what is missing. Across our runs this changed overall scores by at most 0.1 |
| A new registry entry breaks the overall number | BFCL writes display names to an unquoted CSV | keep display names free of commas |

<details>
<summary>Removing failed entries before a re-run</summary>

```python
import glob, json
for path in glob.glob('/data/bfcl-finetune/result/**/*_result.json', recursive=True):
  rows = [json.loads(x) for x in open(path) if x.strip()]
  keep = [r for r in rows if 'traceback' not in r and not str(r.get('result', '')).startswith('Error during inference')]
  if 'memory' not in path and len(keep) < len(rows):
    open(path, 'w').write(''.join(json.dumps(r) + '\n' for r in keep))
```

Memory entries are left alone because they depend on prerequisite snapshots.

</details>
