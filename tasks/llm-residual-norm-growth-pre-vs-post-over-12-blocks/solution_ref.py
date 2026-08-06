import math
import numpy as np


def residual_norm_growth(block_fn, x):
  y = x
  for _ in range(12):
    y = block_fn(y)

  sum_sq_y = 0.0
  for val in y.flat:
    v = float(val)
    sum_sq_y += v * v
  norm_y = math.sqrt(sum_sq_y)

  sum_sq_x = 0.0
  for val in x.flat:
    v = float(val)
    sum_sq_x += v * v
  norm_x = math.sqrt(sum_sq_x)

  return float(norm_y / norm_x)
