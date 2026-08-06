import numpy as np
from gptq.scales import compute_group_scales
from gptq.loop import gptq_quantize_with_recompute


def test_scales_shape_and_positivity():
    w = np.random.randn(16, 32).astype(np.float32)
    scales = compute_group_scales(w, group_size=16, bits=4)
    assert scales.shape == (16, 2)
    assert np.all(scales > 0)


def test_quantization_output_shape():
    np.random.seed(0)
    w = np.random.randn(16, 32).astype(np.float32)
    h = np.dot(w.T, w) + np.eye(32, dtype=np.float32) * 0.1
    out = gptq_quantize_with_recompute(w, h, group_size=16, bits=4)
    assert out.shape == w.shape
    assert not np.array_equal(out, w)
