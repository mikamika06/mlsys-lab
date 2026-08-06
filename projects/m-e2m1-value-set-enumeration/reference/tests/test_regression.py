import numpy as np
from e2m1.enumeration import enumerate_e2m1
from e2m1.quantize import quantize_e2m1

def test_enumeration_not_empty():
    res = enumerate_e2m1()
    assert len(res) == 16

def test_quantize_basic():
    x = [0.0, 1.0, 2.0, 6.0]
    q = quantize_e2m1(x)
    assert q.shape == (4,)
