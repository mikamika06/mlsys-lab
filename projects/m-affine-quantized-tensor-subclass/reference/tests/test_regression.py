import numpy as np
from qtensor.compare import get_rel_err


def test_asymmetric_dequantize_accuracy():
    np.random.seed(42)
    w = np.random.randn(64, 128).astype(np.float32)
    err = get_rel_err(w, "edge_device")
    assert err < 0.2, f"Relative error {err} too high for edge_device!"
