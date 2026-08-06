import numpy as np
from quant.blockwise import quantize_blockwise, dequantize_blockwise

def test_blockwise_roundtrip():
    tensor = np.random.randn(32, 32).astype(np.float32)
    q, s, shape = quantize_blockwise(tensor, 16)
    dq = dequantize_blockwise(q, s, 16, shape)
    assert dq.shape == tensor.shape
    rel_err = np.linalg.norm(tensor - dq) / (np.linalg.norm(tensor) + 1e-12)
    assert rel_err < 0.2
