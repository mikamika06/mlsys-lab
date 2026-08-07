import math

_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _snap_e2m1(y):
  out_list = []
  for row in y:
    row_out = []
    for val in row:
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
      row_out.append(s * _MAG[best_idx])
    out_list.append(row_out)
  return out_list


def mxfp4_full_block_quantize(W: list[list[float]]) -> dict:
  nrows = len(W)
  ncols = len(W[0]) if nrows > 0 else 0

  scale_list = []
  for i in range(nrows):
    m = 0.0
    for j in range(ncols):
      val = abs(W[i][j])
      if val > m:
        m = val
    amax_i = m
    r = amax_i if amax_i > 0 else 6.0
    ratio_val = r / 6.0
    log_val = math.log2(ratio_val)
    ceil_val = math.ceil(log_val)
    e_val = max(0, ceil_val)
    scale_list.append(2.0**e_val)

  y_list = []
  for i in range(nrows):
    row_y = []
    scale_i = scale_list[i]
    for j in range(ncols):
      row_y.append(W[i][j] / scale_i)
    y_list.append(row_y)

  codes = _snap_e2m1(y_list)

  dequant_list = []
  for i in range(nrows):
    row_dq = []
    scale_i = scale_list[i]
    for j in range(ncols):
      row_dq.append(codes[i][j] * scale_i)
    dequant_list.append(row_dq)

  return {"scale": scale_list, "codes": codes, "dequant": dequant_list}
