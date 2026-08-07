import numpy as np

class StackModel:
    def __init__(self, config):
        self.dim = config.get("dim", 64)
        self.w = np.ones((self.dim, self.dim), dtype=np.float32) * 0.1

    def forward(self, x):
        return np.dot(x, self.w)
