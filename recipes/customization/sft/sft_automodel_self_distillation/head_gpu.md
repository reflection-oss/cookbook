You are a GPU kernel engineer. Below is 1 independent task. Environment: Python 3.12, torch 2.14.0+cu130, triton 3.8.0, NVIDIA H200 GPU (CUDA).
float32 unless stated.

For EACH task, write a single Python file that defines one or more @triton.jit kernels and a class `ModelNew`
(same constructor signature as `Model`) whose `forward` returns the same result as `Model.forward` for any
valid input shape, computing the result with your Triton kernel(s). In `forward` you may allocate outputs,
compute grids, and call `.contiguous()`, but you must NOT compute the result with torch operations.
Handle arbitrary sizes with masking; do not assume powers of two.

Rules: answer directly. Do NOT use tools, run commands, or create/edit files. Code only.
Format, strictly: for each task, a line `### <task_id>` followed by exactly one ```python fenced block.

