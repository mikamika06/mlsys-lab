We are implementing a compute-proxy simulator to reproduce Striped Attention's reported throughput ratio across GPUs and sequence lengths. Currently, our pipeline lacks a reliable way to simulate block-to-rank assignment policies and compute overlapping loads accurately, resulting in throughput speedup numbers that diverge significantly from published research values under varying sequence lengths and ring sizes.

Your task is to build a modular simulator that models Ring Attention with striped scheduling, implements the striped block allocation logic, computes communication-computation overlap metrics, and provides a robust regression test suite to catch policy bugs.

To complete this unit, you must implement three milestones:
1. Implement the striped block assignment policy mapping sequence chunks to devices to mirror Striped Attention layouts.
2. Build the compute-proxy throughput simulator that calculates execution times, overlapping metrics, and the final reported throughput ratio against baseline ring attention.
3. Write a comprehensive regression test suite in `tests/test_regression.py` that asserts invariant properties of the striped allocation schedule and throughput bounds, ensuring your test suite fails when a broken block assignment policy is injected.
