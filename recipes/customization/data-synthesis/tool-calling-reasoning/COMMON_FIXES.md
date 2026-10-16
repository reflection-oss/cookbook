# Common fixes

Symptoms we hit while building the data with the [tool-calling reasoning data recipe](README.md), with their causes and fixes.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `REFLECTION_API_KEY is not set` | the key is neither in `.env` in this folder nor exported | put it in `.env`; `main.py` loads `.env` from the current folder |
| `502`, `429 overloaded` or `InternalServerError` in the teacher stage | the API dropped requests | each conversation is retried eight times with backoff; a conversation that failed every attempt or lost a turn to an interruption is sent again on the next `bash run.sh` |
| The teacher stage is slow | Nemotron conversations are long, about 6,000 prompt and 10,000 completion tokens, a minute of generation at `max` effort | raise `WORKERS` up to your key's concurrency limit |
| `select-nemotron` fails while streaming | the pool is written only when the stream completes | run it again; once `work/nemotron_pool.jsonl` exists the command reads it instead of the Hub |
| ToolACE download fails | the Hub was checked on every run | `datasets` caches it under `~/.cache/huggingface`; set `HF_HUB_OFFLINE=1` after the first run |
| Fewer rows than expected after `assemble` | `rows_dropped_clarification_turn` is the act-don't-ask rule; `rows_dropped_missing_turn` counts conversations with a turn without an accepted thought | both are expected; `records_not_in_rows` in the audit summary counts records whose conversation is not in the rows file, which are skipped |
