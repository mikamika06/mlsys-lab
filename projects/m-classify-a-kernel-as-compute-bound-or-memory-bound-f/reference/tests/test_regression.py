import sys
sys.path.insert(0, ".")
from kernelstats.metrics import compute_arithmetic_intensity
from kernelstats.analyzer import classify_kernel


def test_arithmetic_intensity_positive():
    val = compute_arithmetic_intensity(1000, 200)
    assert val == 5.0, f"Expected 5.0, got {val}"


def test_classification_at_ridge():
    ridge = 10.5
    assert classify_kernel(12.0, ridge) == "compute-bound"
    assert classify_kernel(8.0, ridge) == "memory-bound"
    assert classify_kernel(10.5, ridge) == "compute-bound"


def test_zero_bytes_handling():
    val = compute_arithmetic_intensity(500, 0)
    assert val == float('inf')
