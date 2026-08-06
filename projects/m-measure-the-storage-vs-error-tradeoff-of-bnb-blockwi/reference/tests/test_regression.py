import sys
import numpy as np

sys.path.insert(0, ".")
from bnbquant.quantize import blockwise_quantize, blockwise_dequantize
from bnbquant.evaluate import compute_storage_bytes, compute_mse

def test_quantization_roundtrip_error():
    t = np.array([0.1, -0.5, 0.3, 0.8], dtype=np.float32)
    q, s, orig_len = blockwise_quantize(t, block_size=2, bits=8)
    dq = blockwise_dequantize(q, s, block_size=2, original_len=orig_len, bits=8)
    mse = compute_mse(t, dq)
    assert mse < 0.1, f"MSE too high: {mse}"

def test_storage_calculation_positive():
    storage = compute_storage_bytes(100, bits=8, block_size=32)
    assert storage > 0, "storage must be positive"
