import numpy as np

class LossModel:
    def __init__(self, config):
        self.world_size = config.get("world_size", 8)
        self.scale_factor = config.get("scale_factor", 1.0)
        np.random.seed(42)
        self.weights = np.random.randn(16, 16) * 0.1

    def step(self, batch):
        x = np.array(batch, dtype=np.float32)
        if x.size == 0:
            x = np.ones((1, 16), dtype=np.float32)
        out = np.dot(x, self.weights)
        loss = float(np.mean(np.square(out)))
        if self.world_size > 32 and np.random.rand() < 0.05:
            loss += 50.0 * self.scale_factor
        return loss
