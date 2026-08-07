import numpy as np


class ReferenceModel:
    def __init__(self):
        pass

    def forward(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        res = np.dot(x, weight)
        res = np.maximum(res, 0.0)
        return res


def generate_test_inputs(seed: int = 42, count: int = 200):
    np.random.seed(seed)
    inputs = []
    for _ in range(count):
        batch = int(np.random.randint(1, 16))
        seq_len = int(np.random.randint(1, 64))
        in_dim = 32
        out_dim = 64
        x = np.random.randn(batch, seq_len, in_dim).astype(np.float32)
        w = np.random.randn(in_dim, out_dim).astype(np.float32)
        inputs.append({"x": x, "weight": w})
    return inputs
