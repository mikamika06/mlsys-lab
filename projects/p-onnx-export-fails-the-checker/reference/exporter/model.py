import numpy as np

class CustomModel:
    def __init__(self, dim=16):
        self.dim = dim
        self.weights = np.ones((dim, dim), dtype=np.float32)

    def forward(self, x):
        return np.matmul(x, self.weights)

    def export(self, path):
        with open(path, "w") as f:
            f.write("ONNX_GRAPH_SIMULATED")
