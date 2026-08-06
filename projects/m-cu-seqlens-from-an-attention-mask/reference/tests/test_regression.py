import numpy as np
from varlen.seqlens import compute_cu_seqlens
from varlen.packing import unpad, pad
from varlen.leak import detect_leakage


def test_round_trip():
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int32)
    hidden = np.random.randn(2, 3, 4)
    unp = unpad(hidden, mask)
    rep = pad(unp, mask)
    assert np.allclose(rep[mask.astype(bool)], hidden[mask.astype(bool)])


def test_leakage_detection():
    cu = np.array([0, 2, 5], dtype=np.int32)
    mat = np.zeros((5, 5))
    mat[0:2, 0:2] = 1.0
    mat[2:5, 2:5] = 1.0
    assert detect_leakage(mat, cu) is False
    mat[0, 4] = 0.5
    assert detect_leakage(mat, cu) is True
