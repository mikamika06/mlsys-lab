import math
import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _snap_e2m1(y):
  y_arr = np.asarray(y, dtype=np.float64)
  shape = y_arr.shape
  flat = y_arr.ravel()
  out_list = []
  for val in flat:
    abs_val = abs(val)
    min_diff = float("inf")
    best_idx = 0
    for k in range(len(_MAG)):
      diff = abs(abs_val - _MAG[k])
      if diff < min_diff:
        min_diff = diff
        best_idx = k
    if val > 0:
      s = 1.0
    elif val < 0:
      s = -1.0
    else:
      s = 0.0
    out_list.append(s * _MAG[best_idx])
  return np.array(out_list, dtype=np.float64).reshape(shape)


def mxfp4_full_block_quantize(W: np.ndarray) -> dict:
  x = np.asarray(W, dtype=np.float64)
  shape = x.shape
  nrows = shape[0]
  ncols = shape[1]

  scale_list = []
  for i in range(nrows):
    m = 0.0
    for j in range(ncols):
      val = abs(x[i, j])
      if val > m:
        m = val
    amax_i = m
    r = amax_i if amax_i > 0 else 6.0
    ratio_val = r / 6.0
    log_val = math.log2(ratio_val)
    ceil_val = math.ceil(log_val)
    e_val = max(0, ceil_val)
    scale_list.append(2.0**e_val)

  scale = np.array(scale_list, dtype=np.float64)

  y_list = []
  for i in range(nrows):
    row_y = []
    for j in range(ncols):
      row_y.append(x[i, j] / scale[i])
    y_list.append(row_y)
  y = np.array(y_list, dtype=np.float64)

  codes = _snap_e2m1(y)

  dequant_list = []
  for i in range(nrows):
    row_dq = []
    for j in range(ncols):
      row_dq.append(codes[i, j] * scale[i])
    dequant_list.append(row_dq)
  dequant = np.array(dequant_list, dtype=np.float64)

  return {"scale": scale, "codes": codes, "dequant": dequant}
