import numpy as np

SAMPLE_GRAPHS = [
    [
        {"name": "in1", "op": "MatMul", "inputs": ["X", "W1"], "output": "h1", "weights": (128, 64)},
        {"name": "relu1", "op": "ReLU", "inputs": ["h1"], "output": "h1_act"},
        {"name": "out", "op": "Add", "inputs": ["h1_act", "B1"], "output": "Y", "weights": (64,)},
    ],
    [
        {"name": "conv_like", "op": "MatMul", "inputs": ["A", "B"], "output": "C", "weights": (256, 256)},
        {"name": "reshape", "op": "Reshape", "inputs": ["C"], "output": "D", "attributes": {"shape": (1, 256, 256)}},
    ]
]


def generate_parity_data():
    np.random.seed(42)
    x = np.random.randn(16, 128).astype(np.float32)
    w1 = np.random.randn(128, 64).astype(np.float32)
    b1 = np.random.randn(64).astype(np.float32)

    h1 = np.matmul(x, w1)
    h1_act = np.maximum(h1, 0)
    y = h1_act + b1

    pt_outputs = {
        "X": x,
        "W1": w1,
        "B1": b1,
        "h1": h1 + np.random.normal(0, 1e-6, h1.shape),
        "h1_act": h1_act,
        "Y": y + np.random.normal(0, 1e-6, y.shape),
    }

    ir_graph = [
        {"op": "MatMul", "inputs": ["X", "W1"], "output": "h1"},
        {"op": "ReLU", "inputs": ["h1"], "output": "h1_act"},
        {"op": "Add", "inputs": ["h1_act", "B1"], "output": "Y"},
    ]
    return pt_outputs, ir_graph
