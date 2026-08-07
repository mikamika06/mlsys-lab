import sys
import numpy as np

sys.path.insert(0, ".")
from bnb_quant.core import quantize_blockwise
from bnb_quant.outliers import identify_outliers
from bnb_quant.dequant import dequantize_blockwise


def test_quantization_roundtrip_shape():
    tensor = np.random.default_rng(42).standard_normal((64, 64))
    res = quantize_blockwise(tensor, 32)
    assert res["quantized"].shape[1] == 32


def test_outlier_detection_basic():
    tensor = np.zeros((10, 5), dtype=np.float32)
    tensor[0, 2] = 100.0
    outliers = identify_outliers(tensor, threshold=2.0)
    assert outliers[2] == True


def test_dequantization_mse():
    tensor = np.random.default_rng(42).standard_normal((32, 32))
    res = quantize_blockwise(tensor, 64)
    deq = dequantize_blockwise(res, 64)
    mse = np.mean((tensor - deq) ** 2)
    assert mse < 0.1
