import sys
sys.path.insert(0, ".")
import numpy as np
from lora_merge.merger import safe_merge

def test_safe_merge_tolerance():
    np.random.seed(42)
    bw = np.random.randn(32, 32).astype(np.float32) * 0.1
    la = np.random.randn(4, 32).astype(np.float32)
    lb = np.random.randn(32, 4).astype(np.float32)
    merged = safe_merge(bw, la, lb, 16.0, 4)
    scale = 16.0 / 4
    delta = (lb @ la) * scale
    expected = bw + delta
    assert np.max(np.abs(merged - expected)) < 1e-6
