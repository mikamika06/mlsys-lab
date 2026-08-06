import sys
import numpy as np

sys.path.insert(0, ".")
from decode_prof.analysis import find_crossover_batch_size, diagnose_occupancy_limiter


def test_crossover_within_valid_range():
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    tputs = np.array([10.0, 20.0, 40.0, 80.0, 100.0, 110.0, 112.0, 112.5])
    bw = np.array([100.0, 200.0, 400.0, 800.0, 1000.0, 1100.0, 1120.0, 1125.0])
    c = find_crossover_batch_size(batch_sizes, tputs, bw, 1500.0)
    assert c in batch_sizes, f"crossover {c} not in batch sizes"


def test_occupancy_bounds():
    batch_sizes = np.array([1, 2, 4])
    bw = np.array([100.0, 200.0, 400.0])
    res = diagnose_occupancy_limiter(batch_sizes, bw, 1000.0)
    assert np.all(res >= 0.0) and np.all(res <= 1.0), "occupancy ratio out of bounds"
