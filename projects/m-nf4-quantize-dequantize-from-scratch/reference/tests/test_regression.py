import sys
sys.path.insert(0, ".")
import numpy as np
from quant.codebook import get_nf4_codebook, get_fp4_codebook
from quant.quantize import quantize_blockwise, dequantize_blockwise


def test_nf4_codebook_properties():
    cb = get_nf4_codebook()
    assert len(cb) == 16
    assert cb[0] == -1.0
    assert cb[-1] == 1.0


def test_quantize_roundtrip_shape():
    w = np.random.randn(128).astype(np.float32)
    q, s = quantize_blockwise(w, block_size=32, fmt="nf4")
    dq = dequantize_blockwise(q, s, block_size=32, fmt="nf4", original_shape=w.shape)
    assert dq.shape == w.shape


def test_error_comparison():
    w = np.random.normal(0, 1, 256).astype(np.float32)
    q_nf4, s_nf4 = quantize_blockwise(w, 64, "nf4")
    dq_nf4 = dequantize_blockwise(q_nf4, s_nf4, 64, "nf4", w.shape)
    err_nf4 = np.mean((w - dq_nf4) ** 2)
    assert err_nf4 < 1.0
