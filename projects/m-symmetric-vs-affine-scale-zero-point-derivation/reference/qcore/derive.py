import numpy as np


def derive_symmetric_params(w: np.ndarray, num_bits: int = 4):
  """Derive symmetric scale and zero-point for weights."""
  qmax = (1 << (num_bits - 1)) - 1
  max_val = float(np.max(np.abs(w)))
  max_val = max(max_val, 1e-8)
  scale = float(max_val / qmax)
  zero_point = 0
  return scale, zero_point


def derive_affine_params(w: np.ndarray, num_bits: int = 4):
  """Derive affine scale and zero-point for weights."""
  qmin = 0
  qmax = (1 << num_bits) - 1
  rmin = float(np.min(w))
  rmax = float(np.max(w))
  if rmax == rmin:
    rmax = rmin + 1e-8
  scale = float((rmax - rmin) / (qmax - qmin))
  zero_point = int(np.round(-rmin / scale))
  zero_point = int(np.clip(zero_point, qmin, qmax))
  return scale, zero_point
