import sys
import numpy as np

sys.path.insert(0, ".")
from rotary.fusion import apply_fused_rotary
from rotary.sweep import optimal_num_splits
from rotary.bench import decode_latency_curve

def test_rotary_equivalence():
    q = np.ones((1, 1, 4, 32), dtype=np.float32)
    k = np.ones((1, 1, 4, 32), dtype=np.float32)
    cos = np.ones((1, 1, 4, 16), dtype=np.float32)
    sin = np.zeros((1, 1, 4, 16), dtype=np.float32)
    q_out, k_out = apply_fused_rotary(q, k, cos, sin)
    assert not np.allclose(q_out, 0.0)
    assert q_out.shape == q.shape

def test_sweep_bounds():
    assert optimal_num_splits(256) == 1
    assert optimal_num_splits(4096) == 8

def test_latency_curve_monotonic():
    curve = decode_latency_curve([128, 512, 2048])
    assert len(curve) == 3
    assert curve[0] < curve[1] < curve[2]
