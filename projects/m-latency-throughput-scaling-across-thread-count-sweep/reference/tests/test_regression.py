"""Regression tests for thread scaling and NUMA locality."""

import sys
sys.path.insert(0, ".")
from scaling.threads import find_oversubscription_point, analyze_thread_sweep
from scaling.numa import calculate_numa_ratios, evaluate_locality_efficiency


def test_oversubscription_detection():
    topology = {"p_cores": 8, "e_cores": 2}
    latencies = {1: 10.0, 2: 5.5, 4: 3.0, 8: 2.0, 9: 2.8, 10: 3.5, 12: 5.0}
    pt = find_oversubscription_point(topology, latencies)
    assert pt == 9, f"expected oversubscription point at 9, got {pt}"


def test_numa_locality_ratio():
    matrix = [
        [10, 21],
        [21, 10]
    ]
    ratio = calculate_numa_ratios(matrix)
    assert abs(ratio - 2.1) < 1e-5, f"expected ratio 2.1, got {ratio}"

    accesses = [(0, 0), (0, 0), (0, 1), (1, 1)]
    eff = evaluate_locality_efficiency(accesses, matrix)
    assert abs(eff - 0.75) < 1e-5, f"expected locality efficiency 0.75, got {eff}"
