# Ticket: High Training Overhead and Memory Spikes in Custom Optimizer Loop

## Symptom
Our distributed training jobs are experiencing intermittent out-of-memory (OOM) errors during step transitions and showing unexpected latency overheads during optimizer updates. Profiling indicates that state tracking for Adam-style updates consumes significantly more VRAM than the static parameter shapes suggest, particularly during peak allocation windows. Additionally, setting gradients to zero after every step appears to trigger excessive allocation/deallocation thrashing on the CUDA caching allocator rather than releasing underlying memory blocks efficiently.

## Task
Implement a structured memory and execution profiler for PyTorch optimizer steps under `optmem/`.
1. Calculate exact state memory requirements across standard optimizer types (SGD with momentum, Adam, AdamW) given parameter tensors, accounting for tracked moment dynamics and state initialization state.
2. Build an optimizer step compiler abstraction that fuses/compiles optimizer step executions and measures peak memory allocation deltas compared to standard uncompiled step functions.
3. Quantify the exact memory allocation and deallocation footprint differences between setting parameter gradients to zero (`zero_grad(set_to_none=False)`) versus releasing gradient references (`zero_grad(set_to_none=True)`).
4. Provide regression tests in `tests/test_regression.py` that verify state memory accounting accuracy and detect regressions in `set_to_none` allocation behavior when state memory or gradient references are improperly retained.
