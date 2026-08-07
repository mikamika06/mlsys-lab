import sys
import numpy as np

sys.path.insert(0, ".")
from quant.q4_0 import quantize_q4_0, dequantize_q4_0, max_abs_err


def test_roundtrip_error_bounded():
    t = np.linspace(-3.0, 3.0, 64, dtype=np.float32)
    b = quantize_q4_0(t)
    recon = dequantize_q4_0(b, t.shape)
    err = max_abs_err(t, recon)
    assert err < 0.5


def test_output_length():
    t = np.zeros(64, dtype=np.float32)
    b = quantize_q4_0(t)
    assert len(b) == 36


def test_zero_tensor():
    t = np.zeros(32, dtype=np.float32)
    b = quantize_q4_0(t)
    recon = dequantize_q4_0(b, t.shape)
    assert np.all(recon == 0.0)
