import numpy as np
from bnb_sim.quant import quantize_int8, dequantize_int8
from bnb_sim.matmul import mixed_precision_matmul

def test_quantize_roundtrip_preserves_outliers():
    mat = np.random.randn(10, 10).astype(np.float32)
    mat[2, 3] = 100.0
    q, scales, outliers = quantize_int8(mat, threshold=6.0)
    recon = dequantize_int8(q, scales, outliers)
    assert outliers[2, 3] == 100.0
    assert np.abs(recon[2, 3] - 100.0) < 1e-4

def test_matmul_shape():
    A = np.random.randn(4, 10).astype(np.float32)
    B = np.random.randn(10, 5).astype(np.float32)
    res = mixed_precision_matmul(A, B)
    assert res.shape == (4, 5)
