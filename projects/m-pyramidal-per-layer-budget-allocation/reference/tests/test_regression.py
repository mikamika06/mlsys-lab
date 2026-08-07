import sys
sys.path.insert(0, ".")
from pyrkv.allocation import compute_pyramidal_allocation
from pyrkv.curve import compute_accuracy_curve

def test_pyramidal_allocation_sum():
    res = compute_pyramidal_allocation(12, 3600, 100)
    assert sum(res) == 3600

def test_accuracy_curve_monotonicity():
    def dummy(r):
        return r * 0.9
    curve = compute_accuracy_curve([0.2, 0.5, 1.0], dummy)
    scores = [s for _, s in curve]
    assert scores[0] <= scores[1] <= scores[2]
