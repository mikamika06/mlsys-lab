import sys
sys.path.insert(0, ".")
from kvquant.sweep import sample_count_sweep
import numpy as np

def test_sweep_monotonicity():
    data = np.linspace(-5.0, 5.0, 100).tolist()
    counts = [10, 50, 100]
    res = sample_count_sweep(data, counts)
    vals = [res[c] for c in counts]
    assert len(vals) == len(counts)
    for v in vals:
        assert v > 0.0
