import sys
import numpy as np

sys.path.insert(0, ".")
from mlxdiag.drift import evaluate_weight_drift


def test_quant_drift_safety_threshold():
    np.random.seed(42)
    weights = np.random.randn(128, 128).astype(np.float32)

    res_strict = evaluate_weight_drift(weights, bits=4, group_size=64, max_allowed_mse=0.001)
    assert res_strict["exceeds_threshold"] is True, "Expected 4-bit quantization drift to exceed strict MSE threshold"

    res_loose = evaluate_weight_drift(weights, bits=8, group_size=64, max_allowed_mse=0.1)
    assert res_loose["exceeds_threshold"] is False, "Expected 8-bit quantization drift to remain within loose MSE threshold"
