import sys
sys.path.insert(0, ".")

from export.normalize import normalize_to_scale_bias


def test_scale_bias_invariants():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale, bias = normalize_to_scale_bias(mean, std)

    assert len(scale) == 3
    assert len(bias) == 3

    for m, s, sc, b in zip(mean, std, scale, bias):
        assert sc > 0.0
        assert b < 0.0
        reconstructed_zero = (0.0 * sc * 255.0) + b
        expected_zero = (0.0 - m) / s
        assert abs(reconstructed_zero - expected_zero) < 1e-6
