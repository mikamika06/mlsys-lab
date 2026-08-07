import numpy as np
from gguf_pipeline.quantizer import quantize_q8_0, dequantize_q8_0
from gguf_pipeline.evaluator import compute_kl_divergence

def test_quantization_reconstruction():
    np.random.seed(42)
    x = np.random.randn(32, 32).astype(np.float32)
    q = quantize_q8_0(x)
    deq = dequantize_q8_0(q["qdata"], q["scales"])
    err = np.mean(np.abs(x - deq))
    assert err < 0.1, f"Reconstruction error too high: {err}"

def test_kl_divergence_non_negative():
    np.random.seed(42)
    p = np.random.randn(10, 50)
    q = np.random.randn(10, 50)
    kld = compute_kl_divergence(p, q)
    assert kld >= 0.0, f"KL divergence must be non-negative, got {kld}"
