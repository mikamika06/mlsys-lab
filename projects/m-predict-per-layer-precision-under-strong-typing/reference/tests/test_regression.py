import sys
sys.path.insert(0, ".")
from precision.formats import get_format_props
from precision.sweep import run_precision_sweep
from precision.predictor import predict_layer_precision
import numpy as np

def test_format_properties_exist():
    for name in ["FP32", "TF32", "FP16"]:
        p = get_format_props(name)
        assert "max_val" in p
        assert "ulp_eps" in p

def test_sweep_output_keys():
    w = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    res = run_precision_sweep(w, ["FP32", "FP16"])
    assert "FP32" in res
    assert "FP16" in res
    assert "mse" in res["FP16"]

def test_strong_typing_constraints():
    layer = {"max_val": 100.0, "min_val": -100.0, "op_type": "MatMul", "allow_fp16": True}
    constraints = {"force_fp32": True}
    res = predict_layer_precision(layer, constraints)
    assert res == "FP32"

def test_dynamic_range_overflow_forces_fp32():
    layer = {"max_val": 1e6, "min_val": -1e6, "op_type": "MatMul", "allow_fp16": True}
    constraints = {"force_fp32": False, "prefer_tf32": False}
    res = predict_layer_precision(layer, constraints)
    assert res == "FP32"
