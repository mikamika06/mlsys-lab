import sys

sys.path.insert(0, ".")
from macroofline.bandwidth import achieved_bandwidth_gbps, bytes_transferred
from macroofline.roofline import arithmetic_intensity, fit_empirical_roofline

SAMPLES = [
    {"m": 1, "n": 4096, "k": 4096, "elapsed_sec": 0.0002, "itemsize": 2},
    {"m": 4096, "n": 4096, "k": 4096, "elapsed_sec": 0.005, "itemsize": 2},
]


def test_bandwidth_positive_and_scaling():
    """Verify bandwidth calculations return positive scaling values."""
    b = bytes_transferred(1024, 1024, 1024, 2)
    assert b > 0
    bw = achieved_bandwidth_gbps(b, 0.001)
    assert bw > 0


def test_arithmetic_intensity_monotonic():
    """Verify arithmetic intensity increases from vector to square matmul."""
    ai_small = arithmetic_intensity(1, 4096, 4096, 2)
    ai_large = arithmetic_intensity(4096, 4096, 4096, 2)
    assert ai_large > ai_small


def test_memory_bound_classification():
    """Verify low AI matrix shapes are identified as memory bound."""
    fit = fit_empirical_roofline(SAMPLES, 546.0)
    profiles = fit["profiles"]
    low_ai = profiles[0]
    high_ai = profiles[1]
    assert low_ai["is_memory_bound"] is True
    assert high_ai["is_memory_bound"] is False
