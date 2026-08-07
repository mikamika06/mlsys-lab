import sys
sys.path.insert(0, ".")
from fallbackdiag.bits import effective_bits
from fallbackdiag.quant import select_group_size
import numpy as np

def test_effective_bits_greater_than_base():
    cfg = {"bits": 4, "group_size": 32, "has_zero_point": True}
    bits = effective_bits(cfg, (64, 64))
    assert bits > 4.0

def test_group_size_bounds():
    w = np.random.default_rng(0).normal(0, 1, (32, 32))
    gs = select_group_size(w, 1.0)
    assert gs in [32, 64, 128, 256]

def test_effective_bits_scaling():
    cfg1 = {"bits": 4, "group_size": 32, "has_zero_point": False}
    cfg2 = {"bits": 4, "group_size": 256, "has_zero_point": False}
    b1 = effective_bits(cfg1, (128, 128))
    b2 = effective_bits(cfg2, (128, 128))
    assert b1 > b2
