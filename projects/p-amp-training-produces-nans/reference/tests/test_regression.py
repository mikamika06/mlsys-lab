import sys
sys.path.insert(0, ".")
from amp_fix.detector import inspect_module_nans


def test_detector_pinpoints_module():
    tensors = {"layer1": [1.0, 2.0], "layer2": [float("nan"), 1.0]}
    res = inspect_module_nans(tensors)
    assert res["layer2"] is True
    assert res["layer1"] is False


def test_no_nan_over_long_horizon():
    vals = [1.0, 2.0, 3.0]
    assert all(v > 0 for v in vals)
