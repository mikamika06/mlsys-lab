import sys
sys.path.insert(0, ".")
from zerotwo.memory import zero2_memory_breakdown
from zerotwo.comm import calc_bucket_count, toy_reduce_scatter
import numpy as np

def test_memory_monotonicity():
    m1 = zero2_memory_breakdown(1000000, 4)
    m2 = zero2_memory_breakdown(1000000, 8)
    assert m1["total"] > 0
    assert m2["total"] < m1["total"]

def test_bucket_count_positive():
    cnt = calc_bucket_count(100000, 2, 50000)
    assert cnt > 0

def test_reduce_scatter_shape():
    grads = [np.ones(16), np.ones(16)]
    out = toy_reduce_scatter(grads, 2)
    assert len(out) == 2
    assert out[0].shape[0] == 8
