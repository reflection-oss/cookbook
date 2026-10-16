# Beam Triton self-distillation: task corpus kit

Everything needed to rebuild the **task corpus** of the SFT set and to generate the training data yourself by sampling Beam.

| file | what |
|---|---|
| `dataset_manifest.json.gz` | gzip-compressed JSON, read by `manifest_io.load_manifest`. Per task: split (800 train / 200 val), difficulty class from our sampling run, family, provenance (`origin`) and the `rebuild` recipe: for the 652 library tasks the upstream files at their pinned commit (repo, commit, path, sha256) plus an edit script whose ops copy upstream lines by file and line number or insert our own lines; for the 128 KernelBook tasks the dataset row (uuid) and the `Model =` rule; for the 220 procedural tasks the shipped generator and its seed. Non-code task metadata (tolerances, shapes, descriptions, verifier fields) and the sha256 of every rebuilt module, task text and prompt. No task code and no third-party code is stored. Also the pinned prompt header, environment rewrite, verifier, sampling + SFT settings, the 22 library pins (URL, commit, licence) |
| `head_gpu.md` | the shared prompt header (sha256 pinned) |
| `verify_gpu.py` | the verifier used for every verdict (our grader, reformatted, with review fixes; the manifest records the sha256 of the shipped file and of the file as run): candidate compiled on CUDA with `TRITON_F32_DEFAULT=ieee`, reference on CPU, several seeds/shapes, static anti-cheat checks. Run it with the interpreter of a venv holding Python 3.12, torch 2.14.0+cu130, triton 3.8.0, numpy 2.5.3; its sandbox limits imports, sockets, memory and time but not filesystem access |
| `recreate_corpus.py` | rebuilds every task from its pointer: downloads the referenced upstream files at their pinned commits (GitHub raw), checks their sha256 and applies the edit scripts; fetches the KernelBook parquet from the Hugging Face hub and applies the row rule; regenerates the procedural family with the generator (needs torch); re-renders the task texts (`spec_render.py`) and the exact prompts of our sampling run; every module, text and prompt is checked against the recorded sha256; writes `corpus.jsonl`, `prompts/` and the split id lists of the rebuilt tasks; exits 1 if any task fails. `--libsrc DIR` and `--kernelbook PARQUET` use local copies; `--skip-procedural` / `--skip-kernelbook` for partial rebuilds |
| `sample_beam.py` | samples Beam on the prompts through any OpenAI-compatible endpoint (reasoning effort max, T=1, top_p=1, up to 160k tokens, streamed), parses the answer, grades it with `verify_gpu.py` on local GPUs, stops a task at its first PASS; resumable. Endpoint errors retry the same attempt and never consume the budget; exits 1 when the endpoint keeps failing |
| `build_sft_rows.py` | turns a run into SFT rows: first PASS sample per task, prompt looked up in `--prompts` and hash-checked, chat template at reasoning effort max, tokenized `train/val(.tok).jsonl`; rows over `--max-seq-tokens` (32,768, the packer budget) are excluded and listed |
| `generator/` | `gen_corpus.py` + `ops_library.py` regenerate the procedural family (seed 20260904) during the rebuild; `select_kernelbook.py` documents the KernelBook selection |
| `pretok_dataset.py` | the dataset class used by the training recipe: reads the pre-tokenized rows written by `build_sft_rows.py` and emits AutoModel's shifted next-token format; put this directory on `PYTHONPATH` when launching the training job |
| `spec_render.py` | renders the task text from a rebuilt module and its metadata (extracted from the generator, versions pinned) |
| `manifest_io.py` | loads the compressed manifest and hashes content; shared by the scripts above |
| `recipe_beam_triton_sft.yaml` | the NeMo AutoModel training config of the run |
| `smoke_test.py` | offline checks of the kit; `--online` also rebuilds three tasks from their pointers |

## Where the tasks come from

- **652 library tasks** (`origin.kind = library`): a task module written from a function or module of one of 22 open-source
  repositories at a pinned commit (`repo`, `commit_full`, `files[].resolved_path`, `files[].lines`, `files[].sha256_at_commit`,
  licence). The module is a self-contained plain-torch rewrite of the referenced code (einops/codegen inlined, deterministic,
  small shapes), fidelity-checked against the library at authoring time; it is not a verbatim extract, which is why the
  definitions ship. 7 tasks have agent-written paths that do not resolve uniquely at the commit (`found_at_commit: false`); the raw
  `source_file` string is kept for them.
- **128 KernelBook tasks** (`origin.kind = kernelbook`): human-written PyTorch modules from the GPUMODE/KernelBook dataset
  (`uuid`, GitHub repository + commit, licences); `reference_src` = the row's `python_code` plus a `Model = <entry_point>` line.
- **220 procedural tasks** (`origin.kind = procedural`): generated op chains (RMSNorm, softmax, cumsum, ...), no external source.

## Rebuild and run

```bash
python recreate_corpus.py --manifest dataset_manifest.json.gz --out corpus/  # fetch, rebuild, verify -> RECREATE_CORPUS_OK
vllm serve /models/Beam-1.0-501B-A23B --served-model-name beam --tensor-parallel-size 8 --max-model-len 262144 \
    --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser reflection --reasoning-parser reflection --port 8000
python sample_beam.py --corpus corpus/corpus.jsonl --prompts corpus/prompts --out run/ --base-url http://localhost:8000/v1 \
    --model beam --attempts 16 --concurrency 32 --verify-workers 8 --gpus 8                             # results/attempts.jsonl
python build_sft_rows.py --run run/ --prompts corpus/prompts --manifest dataset_manifest.json.gz --out data/self-distillation \
    --tokenizer /models/Beam-1.0-501B-A23B
```

`recreate_corpus.py` needs `pyarrow` and `huggingface_hub` for the KernelBook family and `torch` for the procedural family; `sample_beam.py` needs `requests`; `build_sft_rows.py` needs `transformers` (and `huggingface_hub` unless `--tokenizer` points at a local copy). The verifier runs in its own venv (`.venv/bin/python` next to `verify_gpu.py` by default, override with `--verify-python`) and executes model-generated code without filesystem isolation: use a dedicated machine or container. Our run: pass@1 51.7 % on the first attempt, 89.7 % of tasks solved within 16 attempts; the sampling is
stochastic, so a rerun reproduces the data generation, not our exact samples. `build_sft_rows.py` renders rows exactly like our training set (chat template at reasoning effort max, loss on the assistant tokens).
