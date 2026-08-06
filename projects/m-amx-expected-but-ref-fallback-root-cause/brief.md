# Diagnostic Ticket: unexpected reference fallback during CPU inference profiling

## Symptom
During low-level performance benchmarking of low-precision convolution and GEMM operators on modern Intel Sapphire Rapids servers (configured with Intel AMX hardware support), total model latency degrades significantly compared to baseline AVX-512 runs. Inspecting raw execution logs reveals unexpected invocations of unoptimized reference kernels (`ref:any` / `gemm:ref`) for certain inner matrix dimension $K$ sizes, alongside unpredictable ISA switches across dynamic matrix multiply shapes.

Furthermore, cumulative profiling of full model execution (such as standard ResNet-50 inference) shows disproportionate time spent in specific primitive kinds, but the exact bottleneck breakdown is obfuscated within unstructured `ONEDNN_VERBOSE` output.

## Task
You are tasked with analyzing raw `ONEDNN_VERBOSE` execution traces and building a unified diagnostic and fallback analysis module under `onednn_diag/`:

1. Parse verbose execution logs to detect the root cause of unexpected reference fallbacks (`onednn_diag/fallback.py`). Specifically, pinpoint missing SIMD/AMX ISA primitive implementations, unaligned memory layouts, or unsupported data type combinations that force oneDNN to drop back to scalar/reference execution.
2. Analyze kernel ISA selection behavior (`avx2`, `avx512_core`, `avx512_core_amx`) across dynamic $K$-sweeps (`onednn_diag/isa_sweep.py`). Determine exact transition thresholds and identify efficiency anomalies when switching between vector and matrix tile engines.
3. Compute primitive-kind time-dominance breakdowns (`onednn_diag/profiler.py`) from structured verbose trace inputs, identifying top bottleneck primitive kinds and calculating their percentage share of total primitive execution duration.
4. Implement a safeguard test suite (`tests/test_regression.py`) that detects masked ISA fallbacks and flags unoptimized reference kernel execution in production execution traces.
