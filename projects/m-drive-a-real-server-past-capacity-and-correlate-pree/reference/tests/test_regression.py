import sys
sys.path.insert(0, ".")
from serverload.correlate import correlate
import numpy as np

def test_correlation_bounds():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = np.array([12.0, 22.0, 28.0, 42.0])
    c = correlate(x, y)
    assert -1.0 <= c <= 1.0

def test_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    c = correlate(x, y)
    assert abs(c - 1.0) < 1e-6
