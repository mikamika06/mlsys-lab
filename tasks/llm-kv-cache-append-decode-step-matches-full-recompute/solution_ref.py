import math
import numpy as np


def _softmax(x):
  x_arr = np.asarray(x, dtype=np.float64)
  if x_arr.ndim == 1:
    m = -float("inf")
    for val in x_arr:
      if val > m:
        m = val
    exp_vals = []
    s = 0.0
    for val in x_arr:
      ev = math.exp(val - m)
      exp_vals.append(ev)
      s += ev
    res = [ev / s for ev in exp_vals]
    return np.asarray(res, dtype=np.float64)
  else:
    rows, cols = x_arr.shape
    res = []
    for i in range(rows):
      m = -float("inf")
      for j in range(cols):
        val = x_arr[i, j]
        if val > m:
          m = val
      row_exps = []
      s = 0.0
      for j in range(cols):
        ev = math.exp(x_arr[i, j] - m)
        row_exps.append(ev)
        s += ev
      row_res = [ev / s for ev in row_exps]
      res.append(row_res)
    return np.asarray(res, dtype=np.float64)


def decode_steps(x, Wq, Wk, Wv):
  x_arr = np.asarray(x, dtype=np.float64)
  Wq_arr = np.asarray(Wq, dtype=np.float64)
  Wk_arr = np.asarray(Wk, dtype=np.float64)
  Wv_arr = np.asarray(Wv, dtype=np.float64)

  k_cache = []
  v_cache = []
  outputs = []
  scale = math.sqrt(Wq_arr.shape[1])

  T = x_arr.shape[0]
  D_in = x_arr.shape[1]
  D_out = Wq_arr.shape[1]

  for i in range(T):
    token = x_arr[i]

    q = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wq_arr[k, j]
      q[j] = acc

    k_vec = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wk_arr[k, j]
      k_vec[j] = acc

    v_vec = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wv_arr[k, j]
      v_vec[j] = acc

    k_cache.append(k_vec)
    v_cache.append(v_vec)

    t_curr = len(k_cache)

    scores = [[0.0] * t_curr]
    for c in range(t_curr):
      acc = 0.0
      for k in range(D_out):
        acc += q[k] * k_cache[c][k]
      scores[0][c] = acc / scale

    weights = _softmax(scores)

    out_row = [0.0] * D_out
    for d in range(D_out):
      acc = 0.0
      for c in range(t_curr):
        acc += weights[0, c] * v_cache[c][d]
      out_row[d] = acc
    outputs.append(out_row)

  return np.asarray(outputs, dtype=np.float64)
