# Task generators (as run)

The scripts that produced two of the three task families of the corpus, kept as they were run so the provenance recorded in
`dataset_manifest.json` can be traced back to code. `recreate_corpus.py` runs `gen_corpus.py` with its seed to regenerate the procedural
tasks; the KernelBook tasks are rebuilt from the dataset rows named in the manifest.

| file | what |
|---|---|
| `gen_corpus.py` + `ops_library.py` | the procedural family: op chains (normalisations, softmax, cumsum, ...) rendered as PyTorch reference modules plus specs, seed 20260904; needs torch and triton |
| `select_kernelbook.py` | the KernelBook family: selection and filtering of human-written PyTorch modules from the GPUMODE/KernelBook dataset (`dataset_permissive.parquet`); needs pyarrow and torch |

The library family (task modules written from pinned open-source repositories) was produced by agents working from the
inventories of those repositories; the manifest records, per task, the repository, commit, file and line range they were
written from.
