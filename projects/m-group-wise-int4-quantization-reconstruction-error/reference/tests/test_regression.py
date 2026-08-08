import numpy as np
import sys
sys.path.insert(0, ".")

from quant.group_quant import quantize_group_int4, dequantize_group_int4
from quant.metrics import classify_saturation, compute_reconstruction_mse


def test_saturation_classification():
    data = np.array([-10.0, -5.0, 0.0, 2.0, 5.0, 12.0], dtype=np.float32)
    res = classify_saturation(data, group_size=3, asymmetric=False)
    assert res["saturated_count"] > 0
    assert res["total_count"] == 6
    assert res["saturated_count"] + res["unsaturated_count"] == res["total_count"]


def test_int4_bounds_respected():
    data = np.linspace(-20, 20, 32, dtype=np.float32)
    q, s, zp = quantize_group_int4(data, group_size=8, asymmetric=True)
    assert np.all(q >= -8)
    assert np.all(q <= 7)


def test_reconstruction_mse_non_negative():
    data = np.random.randn(16, 16).astype(np.float32)
    q, s, zp = quantize_group_int4(data, group_size=16)
    rec = dequantize_group_int4(q, s, zp, group_size=16)
    mse = compute_reconstruction_mse(data, rec)
    assert mse >= 0.0
