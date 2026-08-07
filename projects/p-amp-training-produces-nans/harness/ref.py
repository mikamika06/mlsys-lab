import numpy as np


def get_mock_graph():
    return {
        "embedding": np.array([1.0, 2.0]),
        "attention": np.array([0.5, float("nan")]),
        "mlp": np.array([1.1, 1.2])
    }


def get_mock_ops():
    return [
        {"name": "matmul", "precision": "fp16", "sensitive": False},
        {"name": "exp", "precision": "fp16", "sensitive": True},
        {"name": "add", "precision": "fp32", "sensitive": False}
    ]


def get_mock_scaler_state():
    return {"scale": 65536.0, "backoff_factor": 0.5, "growth_factor": 2.0}
