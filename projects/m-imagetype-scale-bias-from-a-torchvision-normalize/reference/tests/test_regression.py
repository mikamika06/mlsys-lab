import sys
sys.path.insert(0, ".")
from core_image.normalize import compute_scale_bias

def test_scale_bias_lengths():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale, bias = compute_scale_bias(mean, std)
    assert len(scale) == 3
    assert len(bias) == 3

def test_scale_positive():
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale, bias = compute_scale_bias(mean, std)
    assert all(s > 0 for s in scale)

def test_bias_computation():
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    scale, bias = compute_scale_bias(mean, std)
    assert all(b == 0.0 for b in bias)
