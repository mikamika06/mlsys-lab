import sys
import numpy as np

sys.path.insert(0, ".")
from quant.derive import derive_symmetric, derive_affine
from quant.analysis import blockwise_size_ratio
from quant.sweep import block_size_sweep


def test_derive_symmetric_bounds():
    w = np.array([-10.0, 5.0, 2.0, -3.0], dtype=np.float32)
    scale, zp = derive_symmetric(w, bits=4)
    assert scale > 0
    assert zp == 0


def test_derive_affine_bounds():
    w = np.array([0.0, 2.0, 4.0, 10.0], dtype=np.float32)
    scale, zp = derive_affine(w, bits=4)
    assert scale > 0
    assert 0 <= zp <= 15


def test_blockwise_size_ratio_positive():
    ratio = blockwise_size_ratio((4096, 4096), 128, bits=4)
    assert 0.0 < ratio < 1.0


def test_block_size_sweep_structure():
    w = np.linspace(-1, 1, 1000, dtype=np.float32)
    res = block_size_sweep(w, [32, 64, 128], bits=4)
    assert len(res) == 3
    for bs, err in res:
        assert bs in [32, 64, 128]
        assert err >= 0.0
