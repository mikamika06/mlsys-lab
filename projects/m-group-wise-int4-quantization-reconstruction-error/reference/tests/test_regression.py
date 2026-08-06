import numpy as np
import sys
sys.path.insert(0, ".")

from quant.group_int4 import compute_reconstruction_mse


def test_smaller_group_size_reduces_or_maintains_mse():
    np.random.seed(42)
    x = np.random.randn(128, 128)
    mse_large = compute_reconstruction_mse(x, group_size=128)["total_mse"]
    mse_small = compute_reconstruction_mse(x, group_size=32)["total_mse"]
    assert mse_small <= mse_large + 1e-6, f"Expected {mse_small} <= {mse_large}"


def test_zero_tensor_handling():
    x = np.zeros((16, 16))
    res = compute_reconstruction_mse(x, group_size=16)
    assert res["total_mse"] == 0.0, "Zero tensor must produce zero MSE"
