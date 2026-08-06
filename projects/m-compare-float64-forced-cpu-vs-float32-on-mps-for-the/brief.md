# MPS Execution and Precision Profiling Anomalies

## Symptom
Our telemetry pipeline for edge benchmarking reports conflicting performance and availability metrics on Apple Silicon targets. When running runtime checks on continuous integration workers, several workers crash or misreport device capabilities because they invoke backend functions without verifying whether PyTorch was compiled with MPS support (`is_built`) prior to querying hardware availability (`is_available`).

Furthermore, initial latency benchmarks show that matrix calculations executed on MPS finish in sub-microsecond wall-clock time regardless of matrix scale, whereas forced CPU execution in `float64` takes significantly longer. Downstream validation pipelines indicate that the MPS timing numbers are drastically inaccurate because asynchronous kernel dispatches return immediately before work finishes on the GPU stream. Additionally, precision comparisons between forced CPU `float64` baselines and MPS `float32` executions lack systematic error tracking, making it impossible to evaluate accuracy loss against a high-precision reference.

## Objective
Build the `mpsbench` package to handle backend support queries safely, profile synchronized execution timings across CPU and MPS targets, and measure relative error metrics against `float64` CPU baselines.

1. Implement `mpsbench/device.py` to check `is_built()` and `is_available()` safely.
2. Implement `mpsbench/bench.py` and `mpsbench/precision.py` to record synchronized wall-clock execution times and compute relative error metrics (`rel_err`).
3. Create a test in `tests/test_regression.py` that verifies synchronization behavior during timing checks.
