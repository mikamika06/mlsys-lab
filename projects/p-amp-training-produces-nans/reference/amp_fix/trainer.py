import numpy as np


class SensitiveModelTrainer:
    def __init__(self, model):
        self.model = model
        self.scale = 1024.0
        self.memory_footprint = 100.0

    def train_steps(self, data_stream, num_steps):
        step_count = 0
        for batch in data_stream:
            if step_count >= num_steps:
                break
            out = self.model(batch)
            if not np.isfinite(out).all():
                self.scale *= 0.5
            else:
                step_count += 1
        return step_count
