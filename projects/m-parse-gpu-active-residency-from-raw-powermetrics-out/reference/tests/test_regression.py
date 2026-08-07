import sys

sys.path.insert(0, ".")
from profile_parser.parser import parse_gpu_active_residency

POWERMETRICS_TEST_LOG = """*** Sampled system activity
GPU HW active residency: 85.50%
ANE Power: 150 mW

*** Sampled system activity
GPU HW idle residency: 40.00%
ANE Power: 200 mW

*** Sampled system activity
GPU use: 60.00%
ANE Power: 0 mW
"""


def test_gpu_active_residency_invariant():
    residencies = parse_gpu_active_residency(POWERMETRICS_TEST_LOG)
    assert len(residencies) == 3, f"Expected 3 samples, got {len(residencies)}"
    assert abs(residencies[0] - 85.50) < 1e-4, f"Sample 0 mismatch: {residencies[0]}"
    assert abs(residencies[1] - 60.00) < 1e-4, f"Sample 1 idle compliment failed: {residencies[1]}"
    assert abs(residencies[2] - 60.00) < 1e-4, f"Sample 2 mismatch: {residencies[2]}"
    for r in residencies:
        assert 0.0 <= r <= 100.0, f"Residency out of bounds: {r}"
