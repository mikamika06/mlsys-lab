import sys
import numpy as np
sys.path.insert(0, ".")
from bertfuse.fp16 import evaluate_fp16

def test_fp16_error_bound():
    g = {"weights": np.ones((4, 4), dtype=np.float32)}
    res = evaluate_fp16(g, threshold=1e-2)
    assert res["error"] <= 1e-2
