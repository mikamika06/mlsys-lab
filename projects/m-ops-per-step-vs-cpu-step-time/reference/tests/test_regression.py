"""Regression tests for launch-bound and compute-bound diagnosis."""

import sys
sys.path.insert(0, ".")

from launchbound.profiler import predict_small_batch_speedup
from launchbound.bounds import cut_ops_until_busy_fraction


def test_speedup_prediction():
    speedup = predict_small_batch_speedup(
        op_count=1000,
        baseline_batch_size=32,
        target_batch_size=4,
        cpu_launch_overhead_us=10.0,
        gpu_time_per_op_per_batch_us=1.0
    )
    assert abs(speedup - 3.2) < 1e-4, f"Expected speedup 3.2, got {speedup}"


def test_busy_fraction_threshold():
    pruned_ops = cut_ops_until_busy_fraction(
        initial_ops=500,
        target_busy_fraction=0.8,
        cpu_launch_overhead_us=5.0,
        gpu_time_per_op_us=20.0
    )
    assert pruned_ops == 500
