"""Learner regression tests."""

import sys

sys.path.insert(0, ".")
from throughput.analyzer import evaluate_concurrency_capacity, locate_knee


def test_knee_detection_accuracy():
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    throughputs = [100.0, 190.0, 360.0, 680.0, 850.0, 880.0, 820.0, 700.0]
    knee = locate_knee(concurrency_levels, throughputs)
    assert knee == 16, f"Expected knee at 16, got {knee}"


def test_concurrency_capacity_threshold():
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    throughputs = [100.0, 190.0, 360.0, 680.0, 850.0, 880.0, 820.0, 700.0]
    ratio = evaluate_concurrency_capacity(concurrency_levels, throughputs, 16)
    assert ratio >= 0.90, f"Expected ratio >= 0.90, got {ratio}"
