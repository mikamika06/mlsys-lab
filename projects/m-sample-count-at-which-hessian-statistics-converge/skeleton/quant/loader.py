import numpy as np


class DeterministicLoader:
  def __init__(self, data: np.ndarray, seed: int, batch_size: int):
    raise NotImplementedError

  def __iter__(self):
    raise NotImplementedError

  def __next__(self):
    raise NotImplementedError
