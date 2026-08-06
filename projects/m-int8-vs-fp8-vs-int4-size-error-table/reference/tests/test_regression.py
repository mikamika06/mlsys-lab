import sys

sys.path.insert(0, ".")
from qsim.calibrate import detect_poison
import numpy as np

def test_detects_poison():
    rs = np.random.RandomState(1337)
    acts = rs.randn(1000, 10)
    acts[500, 5] = 10000.0  # Massive outlier
    assert detect_poison(acts) is True

def test_passes_clean():
    rs = np.random.RandomState(42)
    acts = rs.randn(1000, 10)
    assert detect_poison(acts) is False
