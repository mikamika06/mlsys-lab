import numpy as np
from int8_matmul.outliers import compute_outlier_curve as ref_compute_curve
from int8_matmul.dequant import derive_vector_scales as ref_derive_scales
from int8_matmul.matmul import decomposed_matmul as ref_matmul


def get_test_tensor():
    np.random.seed(42)
    return np.random.randn(32, 64)


def get_thresholds():
    return [2.0, 4.0, 6.0, 8.0]


def get_matrices():
    np.random.seed(1337)
    x = np.random.randn(16, 32)
    w = np.random.randn(32, 16)
    return x, w
