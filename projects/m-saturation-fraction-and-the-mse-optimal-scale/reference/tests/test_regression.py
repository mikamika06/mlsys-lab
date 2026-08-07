import numpy as np
from quantlib.saturation import compute_mse_optimal_scale, saturation_fraction
from quantlib.scaling import decide_scaling_mode
from quantlib.detection import detect_inverted_scale


def test_optimal_scale_positive():
    """Test optimal scale is positive."""
    x = np.array([1.0, 2.0, 100.0, 3.0], dtype=np.float32)
    scale = compute_mse_optimal_scale(x, max_val=448.0)
    assert scale > 0.0


def test_saturation_fraction_bounds():
    """Test saturation fraction bounds."""
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    frac = saturation_fraction(x, scale=0.01, max_val=448.0)
    assert 0.0 <= frac <= 1.0
    assert frac == 1.0


def test_decide_scaling_mode_output():
    """Test decide scaling mode output."""
    history = [np.array([1.0, 2.0], dtype=np.float32), np.array([100.0, 200.0], dtype=np.float32)]
    mode = decide_scaling_mode(history, threshold=2.0)
    assert mode in ("static", "dynamic")


def test_detect_inverted_scale_behavior():
    """Test detect inverted scale behavior."""
    ref_s = 2.5
    inv_s = 1.0 / ref_s
    assert detect_inverted_scale(inv_s, ref_s) is True
    assert detect_inverted_scale(ref_s, ref_s) is False
