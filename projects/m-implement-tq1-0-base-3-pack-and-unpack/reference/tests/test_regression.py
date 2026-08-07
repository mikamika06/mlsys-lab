import sys
sys.path.insert(0, ".")
from ternary.tq1 import pack_tq1_0, unpack_tq1_0
from ternary.analysis import compute_iq4_xs_size, compute_q4_k_s_size, measure_imatrix_effect
import numpy as np


def test_tq1_roundtrip():
    vals = np.array([-1, 0, 1, 1, 0, -1, 0, 1], dtype=np.int8)
    packed = pack_tq1_0(vals)
    unpacked = unpack_tq1_0(packed, len(vals))
    np.testing.assert_array_equal(vals, unpacked)


def test_sizes():
    el = 1024
    sz_iq = compute_iq4_xs_size(el)
    sz_qk = compute_q4_k_s_size(el)
    assert sz_iq > 0
    assert sz_qk > 0


def test_imatrix():
    cb = [0.1, -0.2, 0.5]
    iw = [1.0, 10.0, 1.0]
    res = measure_imatrix_effect(cb, iw)
    assert res > 0.0
