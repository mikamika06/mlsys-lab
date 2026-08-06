"""Regression tests for draft quantization crossover logic."""

import sys

sys.path.insert(0, ".")
from specquant.crossover import find_int8_crossover_size
from specquant.throughput import compute_throughput_ratio

SYSTEM_CONFIG = {
    "memory_bandwidth_gbps": 900.0,
    "kernel_launch_ms": 0.08,
    "dequant_overhead_ms": 0.06,
    "target_step_ms": 25.0
}
ALPHA_CONFIG = {
    "base_alpha_max": 0.85,
    "alpha_scale_m": 250.0
}
CANDIDATE_SIZES = [20, 50, 100, 150, 250, 500, 1000]
DRAFT_LEN = 5


def test_int8_payoff_above_crossover():
    crossover = find_int8_crossover_size(CANDIDATE_SIZES, DRAFT_LEN, SYSTEM_CONFIG, ALPHA_CONFIG)
    above_sizes = [s for s in CANDIDATE_SIZES if s >= crossover]
    assert len(above_sizes) > 0
    for s in above_sizes:
        ratio = compute_throughput_ratio(s, DRAFT_LEN, SYSTEM_CONFIG, ALPHA_CONFIG)
        assert ratio >= 1.0, f"Size {s} above crossover {crossover} has ratio {ratio} < 1.0"


def test_fp16_payoff_below_crossover():
    crossover = find_int8_crossover_size(CANDIDATE_SIZES, DRAFT_LEN, SYSTEM_CONFIG, ALPHA_CONFIG)
    below_sizes = [s for s in CANDIDATE_SIZES if s < crossover]
    for s in below_sizes:
        ratio = compute_throughput_ratio(s, DRAFT_LEN, SYSTEM_CONFIG, ALPHA_CONFIG)
        assert ratio < 1.0, f"Size {s} below crossover {crossover} has ratio {ratio} >= 1.0"
