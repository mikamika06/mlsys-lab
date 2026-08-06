import itertools
import math
import numpy as np


def per_axis_qint8(W: np.ndarray, axis: int = 0):
  """
  Symmetric int8 quantization with one scale per index along `axis`
  (per-output-channel when axis=0): scale = absmax / 127 (no
  zero-point), codes = clip(round(W / scale), -127, 127).
  Returns (codes, scale, dequant).
  """
  W = np.asarray(W, dtype=np.float64)
  shape = W.shape
  ndim = W.ndim

  other_axes = [a for a in range(ndim) if a != axis]
  scale_shape = tuple(shape[a] if a == axis else 1 for a in range(ndim))
  scale = np.empty(scale_shape, dtype=np.float64)

  for k in range(shape[axis]):
    current_max = 0.0
    if not other_axes:
      val = math.fabs(W[k])
      if val > current_max:
        current_max = val
    else:
      for other_indices in itertools.product(*(range(shape[a]) for a in other_axes)):
        full_idx = list(range(ndim))
        for a_idx, a in enumerate(other_axes):
          full_idx[a] = other_indices[a_idx]
        full_idx[axis] = k
        val = math.fabs(W[tuple(full_idx)])
        if val > current_max:
          current_max = val

    if current_max == 0.0:
      current_max = 1.0

    scale_val = current_max / 127.0
    scale_idx = list(range(ndim))
    for a in other_axes:
      scale_idx[a] = 0
    scale_idx[axis] = k
    scale[tuple(scale_idx)] = scale_val

  codes = np.empty(shape, dtype=np.float64)
  deq = np.empty(shape, dtype=np.float64)

  for idx in itertools.product(*(range(s) for s in shape)):
    scale_idx = list(range(ndim))
    for a in other_axes:
      scale_idx[a] = 0
    scale_idx[axis] = idx[axis]
    scale_val = scale[tuple(scale_idx)]

    w_val = W[idx]
    quotient = w_val / scale_val
    rounded = float(round(quotient))
    clipped = 127.0 if rounded > 127.0 else (-127.0 if rounded < -127.0 else rounded)

    codes[idx] = clipped
    deq[idx] = clipped * scale_val

  return codes, scale, deq
