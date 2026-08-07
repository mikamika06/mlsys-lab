import sys
sys.path.insert(0, ".")
from q4k.quantize import quantize_q4_k, dequantize_q4_k
from q4k.analysis import dominant_subblock
from q4k.compare import compare_q4_k_q4_0
import numpy as np


def test_round_trip_bytes():
    arr = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    b = quantize_q4_k(arr)
    b2 = quantize_q4_k(arr)
    assert b == b2


def test_dominant_subblock_bounds():
    arr = np.random.default_rng(42).standard_normal(256).astype(np.float32)
    sub = dominant_subblock(arr)
    assert 0 <= sub < 8


def test_compare_keys():
    arr = np.zeros(256, dtype=np.float32)
    res = compare_q4_k_q4_0(arr)
    assert "mse_q4_k" in res
    assert "mse_q4_0" in res
