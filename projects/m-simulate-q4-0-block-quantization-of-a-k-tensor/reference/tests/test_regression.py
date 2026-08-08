import sys
import numpy as np

sys.path.insert(0, ".")
from q4_0.quant import quantize, dequantize


def test_roundtrip_error_bounds():
    tensor = np.linspace(-1.5, 1.5, 64, dtype=np.float32)
    q = quantize(tensor)
    dq = dequantize(q)
    err = float(np.max(np.abs(tensor - dq)))
    assert err < 0.20, f"error {err} exceeds threshold"


def test_output_shape_preserved():
    tensor = np.zeros((2, 64), dtype=np.float32)
    q = quantize(tensor)
    dq = dequantize(q)
    assert dq.shape == (2, 64)


def test_block_count():
    tensor = np.ones(128, dtype=np.float32)
    q = quantize(tensor)
    assert q["scales"].shape == (4,)
    assert q["packed"].shape == (4, 16)
