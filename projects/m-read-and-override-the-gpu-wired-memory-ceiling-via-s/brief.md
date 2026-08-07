# Overriding GPU Wired Memory Limits and Diagnosing Out-of-Memory Panics

## Symptom
During high-concurrency LLM inference runs using `mlx_lm.server`, workers suddenly experience unrecoverable kernel panics (`IOGPUMemory` panics) or immediate process crashes under heavy KV-cache allocation. The server log records driver memory events, sysctl memory constraints, and allocation requests just before the crash.

## System Behavior
On Apple Silicon unified memory architectures, macOS reserves a portion of total RAM for system and UI operations by imposing a ceiling on wired (non-pageable) GPU memory (`iogpu.wired_mem_limit` or `iogpu.wired_mem_limit_mb`). By default, `sysctl` calculates this limit as a deterministic fraction of `hw.memsize`. When total wired allocations exceed this dynamically computed threshold, the driver triggers an unrecoverable kernel assertion or panic rather than returning a clean allocation failure to user space.

## Goal
You will build a Python utility module `sysctl_mem` that:
1. Calculates default wired memory ceilings from physical memory capacity (`hw.memsize`) and provides validated `sysctl` command overrides.
2. Parses structured `mlx_lm.server` crash logs to extract allocation metrics, detect panic preconditions, and diagnose `IOGPUMemory` failure modes.
3. Provides regression tests in `tests/test_regression.py` that catch unsafe memory override limits before deployment.
