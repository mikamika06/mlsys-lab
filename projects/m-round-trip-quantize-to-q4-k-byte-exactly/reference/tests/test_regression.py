import sys
sys.path.insert(0, ".")
import numpy as np
from q4k.quant import round_trip_q4_k
from q4k.mse import locate_dominating_subblock
from q4k.compare import compare_q4k_q40

def test_round_trip_byte_exact():
    rng = np.random.RandomState(123)
    x = rng.randn(256).astype(np.float32)
    assert round_trip_q4_k(x) == 1.0

def test_locate_subblock_bounds():
    rng = np.random.RandomState(123)
    x = rng.randn(256).astype(np.float32)
    idx = locate_dominating_subblock(x)
    assert 0 <= idx < 8

def test_compare_metrics():
    rng = np.random.RandomState(123)
    x = rng.randn(256).astype(np.float32)
    res = compare_q4k_q40(x)
    assert "mse_q4k" in res
    assert "mse_q40" in res
    assert res["mse_q4k"] >= 0.0
