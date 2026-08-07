# Reconcile Per-Op Overhead and Detect Invalid Execution Targets

During recent benchmarking of CPU inference pipelines, performance engineers noticed severe discrepancies between total reported wall-clock time and the sum of individual execution times returned by per-op profiling. In several cases, individual node profiling was enabled across execution graphs, but the aggregated per-op time accounted for less than 60% of the top-level application duration. Engineers suspect uncaptured dispatch overhead, synchronization delays, or incorrect string-based execution target parameters causing silent fallback behavior or unhelpful errors when bad device targets are provided.

You are tasked with building a robust profiling log reconciliation and validation utility for OpenVINO benchmark reports. Your utility must parse structured performance reports, compute total wall-clock time and per-operator duration metrics, reconcile discrepancies by calculating relative overhead bounds, and validate targeted inference hardware strings.

## Symptoms
* Aggregated per-op profile metrics do not sum to total execution wall-clock time, leading to inaccurate bottleneck analysis in automated performance regression runs.
* Passing invalid execution targets (such as misspelled or unsupported CPU/GPU string identifiers) results in unvalidated input or unexpected runtime errors without proper diagnostic messages matching OpenVINO standards.
* The team lacks automated regression tests ensuring reconciliation functions correctly flag unaccounted pipeline delays.

## Task
1. Implement report parsing and time reconciliation functions to extract per-op timings, total wall-clock duration, and compute relative time discrepancies.
2. Implement device string validation that raises exact standard OpenVINO diagnostic errors when provided invalid target devices.
3. Write comprehensive regression tests in `tests/test_regression.py` that catch missing or incorrect reconciliation metrics when tested against invalid calculation variants.
