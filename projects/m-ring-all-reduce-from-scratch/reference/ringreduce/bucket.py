import numpy as np


class GradientBucket:
    def __init__(self, capacity_bytes):
        self.capacity_bytes = capacity_bytes
        self.current_bytes = 0
        self.tensors = []

    def add(self, tensor):
        t = np.array(tensor, dtype=np.float32)
        nbytes = t.nbytes
        if self.current_bytes + nbytes > self.capacity_bytes and self.tensors:
            return False
        self.tensors.append(t)
        self.current_bytes += nbytes
        return True

    def flush(self):
        if not self.tensors:
            return None
        combined = np.concatenate([t.flatten() for t in self.tensors])
        self.tensors = []
        self.current_bytes = 0
        return combined
