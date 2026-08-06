import sys
sys.path.insert(0, ".")
from quantlib.layout import fix_qzeros
from quantlib.gidx import apply_gidx, invert_gidx
from quantlib.pack import compute_packing_sizes
import numpy as np


def test_fix_qzeros_shape_and_values():
    q = np.zeros((4, 4), dtype=np.int32)
    res = fix_qzeros(q, 32)
    assert res.shape == q.shape
    assert np.all(res == q + 1)


def test_gidx_roundtrip():
    w = np.arange(16, dtype=np.float32).reshape(2, 8)
    g = np.array([2, 0, 1, 3, 2, 0, 1, 3])
    p = apply_gidx(w, g)
    back = invert_gidx(p, g)
    assert np.allclose(w, back)


def test_packing_sizes_positive():
    res = compute_packing_sizes(1000, 4)
    assert res["aligned_bytes"] >= res["unaligned_bytes"]
    assert res["unaligned_bytes"] > 0
