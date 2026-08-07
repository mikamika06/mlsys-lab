# SLA Compliance Classification Across Batch & Percentile Measurements

The CPU inference service team is reporting degraded service-level agreement (SLA) compliance on our real-time inference cluster. When request traffic spikes, latency tail behavior changes non-linearly across different batch sizes. Current ad-hoc operational scripts fail to accurately classify whether serving configurations meet tail latency constraints (such as P90, P95, and P99) under target batch size profiles and load scenarios.

Your task is to build a robust classification engine and latency profiler that models batch serving mechanics, evaluates percentile latency distributions against target SLAs, and detects non-compliant execution configurations before deployment.

## Requirements

1. **Batch Latency Profiling & Classification (`sla/profiler.py`)**:
   - Implement `classify_sla_compliance(batch_profiles, target_sla)` to analyze batch latency measurements across target percentiles (P50, P90, P95, P99).
   - Compute exact empirical percentiles from raw request latency distributions per batch size.
   - Evaluate whether each batch size setting satisfies max latency constraints for given SLA percentile targets. Return compliance classifications, violating percentiles, and maximum allowable batch size (`max_compliant_batch`).

2. **System Overhead & Cost Tradeoff Analysis (`sla/cost.py`)**:
   - Implement `compute_cost_efficiency(batch_profiles, target_sla, cost_per_cpu_second)` to compute the balance between SLA compliance, compute resource usage, and effective request throughput (QPS).
   - Implement `recommend_optimal_batch_size(batch_profiles, target_sla, cost_per_cpu_second)` to select the optimal batch size that maximizes compliant throughput while staying within SLA boundaries.

3. **Regression Safety Net (`tests/test_regression.py`)**:
   - Write comprehensive unit tests ensuring SLA compliance classification strictly flags percentile violations.
   - Your tests will be evaluated against broken classifiers that ignore tail percentile spikes (e.g., evaluating only median P50 instead of checking higher percentiles like P99).
