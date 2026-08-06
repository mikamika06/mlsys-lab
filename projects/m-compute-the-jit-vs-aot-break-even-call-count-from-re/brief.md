# Latency Spikes and Compilation Overhead in Triton Kernel Deployment

## Symptom
Production model serving clusters running Triton-compiled kernels are experiencing severe latency spikes during low-volume batch execution and initial endpoint cold starts. Diagnostics reveal that dynamic switching between runtime JIT compilation (`triton.jit`) and pre-compiled ahead-of-time (AOT) C++ binaries is operating sub-optimally. Low-frequency microservice workloads are triggering full JIT compilation passes on single-invocation calls where the compilation latency overhead dominates total runtime. Conversely, high-throughput long-running pipelines are using pre-compiled AOT binaries with higher per-call dispatch overheads, failing to leverage runtime JIT kernel specializations that offer lower execution latencies.

## Task
Implement the `aotbreak` package to compute exact JIT vs. AOT break-even call counts from recorded execution trace profiles:
1. `aotbreak/profiler.py`: Implement `parse_overhead_records` to parse JIT compilation delays, AOT load overheads, and warm per-call execution latencies across workloads.
2. `aotbreak/breakeven.py`: Implement `compute_breakeven` to compute the minimum integer call threshold $N_{\text{break}}$, crossover total latency, and overhead delta.
3. `aotbreak/schedule.py`: Implement `select_strategy` to assign the optimal compilation mode and compute expected latency savings for scheduled invocation volumes.
4. `tests/test_regression.py`: Write an invariant test verifying that at $N_{\text{break}}$, the preferred execution mode strictly achieves lower or equal cumulative latency compared to the alternative mode.
