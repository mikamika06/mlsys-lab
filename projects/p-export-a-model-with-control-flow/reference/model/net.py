import numpy as np

class ConditionalModel:
    def __init__(self):
        self.threshold = 0.5

    def forward(self, x):
        batch_size = x.shape[0]
        out = np.zeros_like(x)
        for i in range(batch_size):
            val = np.sum(x[i])
            if val > self.threshold:
                out[i] = x[i] * 2.0
            else:
                out[i] = x[i] * 0.5
        return out

    def export_check(self, x):
        res = self.forward(x)
        return 1, res
