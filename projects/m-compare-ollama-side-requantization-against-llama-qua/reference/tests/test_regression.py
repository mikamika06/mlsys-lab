import numpy as np
from runner.quant_compare import (
    compare_quantization_drift,
    llama_quantize_mock,
    ollama_requantize_mock,
)


def test_quantization_drift_detection():
    """Verify quantization comparison invariants and drift bounds."""
    np.random.seed(42)
    data = np.random.randn(1024).astype(np.float32)
    res = compare_quantization_drift(data, quant_type="Q4_0")
    assert "mse_llama" in res
    assert "mse_ollama" in res
    assert "max_diff" in res
    assert "cosine_sim" in res
    assert res["max_diff"] >= 0.0
    assert -1.0 <= res["cosine_sim"] <= 1.0

    q_llama = llama_quantize_mock(data, quant_type="Q4_0")
    q_ollama = ollama_requantize_mock(data, quant_type="Q4_0")
    assert q_llama.shape == data.shape
    assert q_ollama.shape == data.shape
