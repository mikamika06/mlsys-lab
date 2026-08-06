import numpy as np
import sys
sys.path.insert(0, ".")
from quant.legacy import quantize_q8_0, dequantize_q8_0, block_properties, compute_rmse

def test_q8_0_roundtrip():
    rng = np.random.default_rng(123)
    x = rng.standard_normal(256).astype(np.float32)
    encoded = quantize_q8_0(x)
    decoded = dequantize_q8_0(encoded, x.shape)
    assert compute_rmse(x, decoded) < 0.05

def test_block_properties_structure():
    props = block_properties()
    assert "Q8_0" in props
    assert props["Q8_0"]["bytes_per_block"] == 34
    assert props["Q8_0"]["bpw"] == 8.5
