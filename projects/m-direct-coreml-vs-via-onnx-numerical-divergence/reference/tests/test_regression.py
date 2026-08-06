import numpy as np
from coreml_audit.divergence import max_abs_error
from coreml_audit.census import node_census
from coreml_audit.min_config import minimal_op_config

def test_min_config_tolerance():
    models = [{
        "id": "m1",
        "direct": np.zeros((1, 4), dtype=np.float32),
        "onnx": np.ones((1, 4), dtype=np.float32) * 0.1,
        "nodes": [{"op": "MatMul", "provider": "ANE"}]
    }]
    cfg = minimal_op_config(models)
    assert cfg["max_tolerance"] >= 0.0
    assert "MatMul" in cfg["allowed_ops"]

def test_node_census_sorting():
    nodes = [
        {"op": "Add", "provider": "CPU"},
        {"op": "MatMul", "provider": "ANE"},
        {"op": "Add", "provider": "CPU"}
    ]
    res = node_census(nodes)
    assert len(res) == 2
    assert res[0]["op"] == "Add"
    assert res[0]["count"] == 2

def test_divergence_value():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.1, 1.9], dtype=np.float32)
    assert abs(max_abs_error(a, b) - 0.1) < 1e-5
