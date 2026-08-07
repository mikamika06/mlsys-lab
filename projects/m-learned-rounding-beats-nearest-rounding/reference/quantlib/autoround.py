import numpy as np


class AutoRoundModifier:
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def optimize(self, calibration_data):
        optimized_weights = []
        for w in self.model["weights"]:
            scale = self.model["scale"]
            zp = self.model["zero_point"]
            v = w / scale + zp
            q_base = np.floor(v)
            r = np.clip(v - q_base, 0.0, 1.0)
            for _ in range(20):
                grad = 2.0 * ((q_base + r - zp) * scale - w) * scale
                r = r - 0.02 * grad
                r = np.clip(r, 0.0, 1.0)
            final_q = np.clip(q_base + np.round(r), -8, 7)
            optimized_weights.append((final_q - zp) * scale)
        return optimized_weights
