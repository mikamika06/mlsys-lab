import sys
sys.path.insert(0, ".")
from imageconv.scale import get_scale_bias
from imageconv.convert import verify_drift
from imageconv.metrics import size_ratio

def test_scale_bias_length():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale, bias = get_scale_bias(mean, std)
    assert len(scale) == 3
    assert len(bias) == 3

def test_verify_drift_bounds():
    import numpy as np
    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    assert verify_drift(a, b, 1e-5) is True

def test_size_ratio_positive():
    r = size_ratio(100, 50)
    assert r > 1.0
