import sys
sys.path.insert(0, ".")
from int8_eng.tuning import find_sensitive_layers, calibrate

def test_sensitivity_check():
    data = {"layer1": 0.005, "layer2": 0.05}
    sens = find_sensitive_layers(data, threshold=0.03)
    assert "layer2" in sens
    assert "layer1" not in sens

def test_calibration_validity():
    model = {"calibrated": False}
    calibrated_model = calibrate(model, [1.0, 2.0, 3.0])
    assert calibrated_model["calibrated"] is True
    assert calibrated_model["scale"] == 2.0
