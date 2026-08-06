import math


def _softmax(x: list[float]) -> list[float]:
  m = -float("inf")
  for val in x:
    if val > m:
      m = val
  exp_vals = []
  s = 0.0
  for val in x:
    ev = math.exp(val - m)
    exp_vals.append(ev)
    s += ev
  return [ev / s for ev in exp_vals]


def decode_steps(
    x: list[list[float]],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
) -> list[list[float]]:
  k_cache = []
  v_cache = []
  outputs = []

  T = len(x)
  D_in = len(x[0]) if T > 0 else 0
  D_out = len(Wq[0]) if len(Wq) > 0 else 0

  scale = math.sqrt(D_out)

  for i in range(T):
    token = x[i]

    q = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wq[k][j]
      q[j] = acc

    k_vec = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wk[k][j]
      k_vec[j] = acc

    v_vec = [0.0] * D_out
    for j in range(D_out):
      acc = 0.0
      for k in range(D_in):
        acc += token[k] * Wv[k][j]
      v_vec[j] = acc

    k_cache.append(k_vec)
    v_cache.append(v_vec)

    t_curr = len(k_cache)

    scores = [0.0] * t_curr
    for c in range(t_curr):
      acc = 0.0
      for k in range(D_out):
        acc += q[k] * k_cache[c][k]
      scores[c] = acc / scale

    weights = _softmax(scores)

    out_row = [0.0] * D_out
    for d in range(D_out):
      acc = 0.0
      for c in range(t_curr):
        acc += weights[c] * v_cache[c][d]
      out_row[d] = acc
    outputs.append(out_row)

  return outputs
