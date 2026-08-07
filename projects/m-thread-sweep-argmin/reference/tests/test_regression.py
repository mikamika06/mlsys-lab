import sys
sys.path.insert(0, ".")
from ortopt.sweep import thread_sweep
from ortopt.iobind import run_with_iobinding
from ortopt.metrics import compute_copy_share
import numpy as np

def test_thread_sweep_argmin():
    data = {1: [10.0, 9.5], 2: [5.0, 5.2], 4: [8.0, 8.1]}
    assert thread_sweep(data) == 2

def test_iobinding_execution():
    inputs = [np.array([1.0, 2.0], dtype=np.float32)]
    res = run_with_iobinding(None, inputs)
    assert len(res) == 1
    assert np.allclose(res[0], [2.0, 4.0])

def test_copy_share_computation():
    share = compute_copy_share(100.0, 25.0)
    assert np.isclose(share, 0.25)
