import sys
import numpy as np

sys.path.insert(0, ".")
from int8_matmul.dequant import derive_vector_scales
from int8_matmul.outliers import compute_outlier_curve
from int8_matmul.matmul import decomposed_matmul


def test_derive_vector_scales_shape():
    t = np.random.randn(16, 64)
    scales = derive_vector_scales(t)
    assert scales.shape == (16, 1)


def test_compute_outlier_curve_length():
    t = np.random.randn(32, 32)
    thresholds = [3.0, 6.0, 9.0]
    curve = compute_outlier_curve(t, thresholds)
    assert len(curve) == len(thresholds)


def test_decomposed_matmul_finite():
    x = np.random.randn(8, 16)
    w = np.random.randn(16, 8)
    res = decomposed_matmul(x, w, threshold=6.0)
    assert np.all(np.isfinite(res))
