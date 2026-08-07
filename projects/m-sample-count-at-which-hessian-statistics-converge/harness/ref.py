import numpy as np


def generate_data(seed: int, n_samples: int, dim: int):
  rng = np.random.default_rng(seed)
  return rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))
