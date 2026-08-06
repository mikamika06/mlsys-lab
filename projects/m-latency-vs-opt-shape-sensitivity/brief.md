# TensorRT Profile Optimization & Latency Sensitivity

## Symptom
Our TensorRT engine deployment for a multi-batch dynamic shape workload shows severe latency degradation when handling inputs near the lower end of the profile range. A single wide profile configured with min shape `(1, 64, 512)`, opt shape `(16, 64, 512)`, and max shape `(32, 64, 512)` yields acceptable throughput at batch size 16, but small requests (batch size 1 to 4) run significantly slower than a standalone engine built specifically for small shapes. Profile benchmarking indicates shape-dependent kernel selection trade-offs that cause latency spikes across the broad dynamic range. Additionally, our pipeline configuration failed during execution-tensor vs shape-tensor validation because dimensions passed as shape tensors were treated as regular execution tensors.

## Task
You must diagnose and fix the shape sensitivity issue and correct tensor classification for TensorRT profiles.
1. Implement a diagnostic metric to measure profile shape sensitivity across dynamic ranges and classify engine inputs into execution tensors vs shape tensors based on TensorRT graph specifications.
2. Refactor the single wide optimization profile into two targeted profiles (a low-range profile and a high-range profile) with non-overlapping optimal shapes, ensuring latency ratio constraints are satisfied relative to a single-profile baseline.
3. Write a regression test suite in `tests/test_regression.py` that verifies profile boundaries, tensor classification, and shape-tensor invariants, and catches regressions when profiles are improperly merged or misconfigured.
