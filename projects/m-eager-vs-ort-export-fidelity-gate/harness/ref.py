import numpy as np

FIDELITY_TESTS = [
    {
        "eager": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
        "ort": [np.array([1.00001, 2.00001, 3.00001], dtype=np.float32)],
        "rtol": 1e-3,
        "atol": 1e-3,
        "expected": True,
    },
    {
        "eager": [np.array([10.0, 20.0], dtype=np.float32)],
        "ort": [np.array([10.5, 20.0], dtype=np.float32)],
        "rtol": 1e-3,
        "atol": 1e-3,
        "expected": False,
    },
    {
        "eager": [np.array([[1.0, 2.0]], dtype=np.float32)],
        "ort": [np.array([[1.0, 2.0]], dtype=np.float32)],
        "rtol": 1e-5,
        "atol": 1e-5,
        "expected": True,
    },
]

TRACEBACK_TESTS = [
    ("torch._dynamo.exc.Unsupported: dynamic control flow detected", "graph_break"),
    ("RuntimeError:aten::native_dropout is not supported in ONNX export", "unsupported_type"),
    ("RuntimeError: shape mismatch in MatMul node inputs", "shape_mismatch"),
    ("Some random python error traceback occurred", "unknown"),
]

CONSTANT_TESTS = [
    {
        "nodes": [{"op": "Constant", "val": 128, "is_baked": True}, {"op": "Add"}],
        "expected": 1,
    },
    {
        "nodes": [{"op": "Param", "name": "batch_size"}, {"op": "Mul"}],
        "expected": 0,
    },
]
