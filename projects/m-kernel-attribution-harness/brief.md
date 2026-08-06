# Ticket: Production Attention Kernels Slipping Into Fallbacks and O(N^2) Allocations

## Symptom
During recent large-sequence inference benchmarks, several models triggered severe memory spikes and latency degradation. Profiling shows that attention operations occasionally silently fall back to standard PyTorch matrix multiplication or unoptimized fused kernels, resulting in quadratic memory growth relative to sequence length.

Furthermore, post-mortem analysis of execution logs is difficult because kernel invocation records are fragmented across multiple tracing subsystems without clear attribution to specific source modules or call sites.

## Requirements
We need a diagnostic tool and attribution harness capable of parsing execution traces, detecting implicit quadratic attention patterns, and diagnosing fallback root causes:

1. **Kernel Attribution Harness**: Parse runtime execution events, trace visual frames/call stacks, and map raw CUDA/CPU kernel launch records back to high-level attention module invocations and call sites.
2. **Offline Quadratic Memory Detector**: Analyze execution trace memory allocations and kernel execution shapes without needing an active GPU to flag operations scaling quadratically $O(N^2)$ with sequence length rather than $O(N)$ memory usage.
3. **Fallback Cause Identification**: Implement an automated diagnostic engine that inspects kernel dispatch configurations, tensor strides, dtypes, and alignment constraints to identify why FlashAttention was bypassed (e.g., non-contiguous inputs, unsupported head dimensions, unaligned memory addresses, or mismatched dtypes).
4. **Safeguard Tests**: Implement regression tests in `tests/test_regression.py` that verify fallback cause rules and detect improper dispatch decisions.

Work in `kernel_attr/` to implement the harness, offline memory detector, and diagnostic module.
