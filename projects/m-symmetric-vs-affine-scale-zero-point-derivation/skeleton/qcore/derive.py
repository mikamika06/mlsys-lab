import numpy as np


def derive_symmetric_params(w: np.ndarray, num_bits: int = 4):
  """Derive symmetric scale and zero-point for weights."""
  raise NotImplementedError


def derive_affine_params(w: np.ndarray, num_bits: int = 4):
  """Derive affine scale and zero-point for weights."""
  raise NotImplementedError
