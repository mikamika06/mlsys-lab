import numpy as np
from reference.quant.q4_0 import quantize_q4_0, dequantize_q4_0, max_abs_err

np.random.seed(42)
TENSORS = [
    np.random.randn(32).astype(np.float32) * 2.0,
    np.random.randn(64).astype(np.float32) * 5.0,
    np.random.randn(128).astype(np.float32) * 1.5,
    np.zeros(32, dtype=np.float32)
]
