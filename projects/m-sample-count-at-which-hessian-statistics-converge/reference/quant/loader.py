import numpy as np


class DeterministicLoader:
  def __init__(self, data: np.ndarray, seed: int, batch_size: int):
    self.data = data
    self.seed = int(seed)
    self.batch_size = int(batch_size)
    self.rng = np.random.default_rng(self.seed)
    self.indices = np.arange(len(self.data))
    self.rng.shuffle(self.indices)
    self.pointer = 0

  def __iter__(self):
    self.pointer = 0
    return self

  def __next__(self):
    if self.pointer >= len(self.data):
      raise StopIteration
    end = min(self.pointer + self.batch_size, len(self.data))
    batch_idx = self.indices[self.pointer:end]
    self.pointer = end
    return self.data[batch_idx]
