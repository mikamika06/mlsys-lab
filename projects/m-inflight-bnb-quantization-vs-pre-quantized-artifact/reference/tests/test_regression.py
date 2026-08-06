import sys
import numpy as np

sys.path.insert(0, ".")
from bnb_quant.loader import load_model_weights, quantize_fp16_to_int8


def test_quantization_layout_alignment():
    np.random.seed(42)
    sample_weight = np.random.randn(128, 256).astype(np.float32)

    qweight, scales = quantize_fp16_to_int8(sample_weight)

    assert qweight.shape == sample_weight.shape
    assert scales.shape == (sample_weight.shape[0], 1)

    reconstructed = qweight.astype(np.float32) * scales
    diff = np.abs(sample_weight - reconstructed)
    assert np.max(diff) <= 2.5
