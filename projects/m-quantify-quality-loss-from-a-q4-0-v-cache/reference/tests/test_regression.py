import numpy as np
import sys

sys.path.insert(0, ".")
from vcache.quant_eval import evaluate_v_cache_loss, quantize_q4_0, dequantize_q4_0


def test_q4_0_loss_bounds():
    np.random.seed(42)
    v = np.random.randn(16, 8, 128, 64).astype(np.float32)
    loss = evaluate_v_cache_loss(v, block_size=32)
    assert 0.0 < loss < 0.2, f"Loss {loss} out of expected bounds for q4_0"


def test_q4_0_differs_from_fp16():
    np.random.seed(42)
    v = np.random.randn(4, 4, 32, 64).astype(np.float32)
    q_data = quantize_q4_0(v, block_size=32)
    dequant = dequantize_q4_0(q_data, v.shape, block_size=32)
    diff = np.abs(v - dequant)
    assert np.max(diff) > 1e-4, "q4_0 dequantization unexpectedly identical to original"
