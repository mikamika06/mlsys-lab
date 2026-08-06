We are observing unexpected `torch.float32` activations and parameters remaining in FP32 inside PyTorch `torch.cuda.amp.autocast()` blocks when training mixed-precision vision and transformer models. Engineers report that certain subgraphs fall back to full precision, but manual tensor inspection after execution is tedious and fails to track intermediate operation dtypes or diagnose why a particular region stayed FP32.

To debug this across arbitrary computation graphs, we need an automated static/dynamic autocast dtype oracle. The tool must trace a PyTorch module or callable, inspect the automatic mixed precision rules, determine the expected and actual intermediate dtypes across the graph, and report exactly why an execution region or op stayed in FP32 (e.g., due to strict FP32 op registration, explicit cast overrides, or mixed-dtype promotions).

Your task is to build the autocast oracle package:
1. Implement intermediate tensor dtype tracing and expected vs actual dtype evaluation under CUDA autocast rules.
2. Build a diagnostic engine that computes a per-intermediate dtype map and identifies the underlying cause whenever an autocast region or operation executes in FP32.
3. Write a regression test suite in `tests/test_regression.py` that verifies your oracle correctly flags improper autocast fallbacks and catches regressions where FP32 ops are incorrectly marked as autocasted.
