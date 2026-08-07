import numpy as np
from quantopt.calibration import calibrate_w8a8
from quantopt.analysis import why_fp16
from quantopt.search import find_budget_config

def test_calibration_zero_point_is_zero():
    t = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    _, zp = calibrate_w8a8(t)
    assert zp == 0, f"expected zero point 0, got {zp}"

def test_calibration_scale_positive():
    t = np.array([-5.0, 5.0], dtype=np.float32)
    scale, _ = calibrate_w8a8(t)
    assert scale > 0.0, f"scale must be positive, got {scale}"

def test_analysis_sensitivity():
    l = {"name": "test", "params": 100, "sensitivity": 0.8, "supported_bits": [4, 8, 16]}
    assert why_fp16(l) == "high_sensitivity"

def test_search_respects_limits():
    layers = [{"name": "l1", "params": 5000, "supported_bits": [8, 16]}]
    cfg = find_budget_config(layers, 1000)
    assert cfg["l1"] == 16
