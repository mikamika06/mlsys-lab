import numpy as np


def vector_wise_int8_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
  X = np.asarray(X, dtype=np.float64)
  W = np.asarray(W, dtype=np.float64)

  n, k = X.shape
  k_w, m = W.shape

  sx = []
  for i in range(n):
    m_val = 0.0
    for j in range(k):
      v = abs(X[i, j])
      if v > m_val:
        m_val = v
    sx.append(m_val / 127.0)

  sw = []
  for j in range(m):
    m_val = 0.0
    for i in range(k):
      v = abs(W[i, j])
      if v > m_val:
        m_val = v
    sw.append(m_val / 127.0)

  safe_sx = [1.0 if val == 0.0 else val for val in sx]
  safe_sw = [1.0 if val == 0.0 else val for val in sw]

  Xq = []
  for i in range(n):
    row = []
    s = safe_sx[i]
    for j in range(k):
      val = X[i, j] / s
      rounded = round(val)
      clipped = max(-127, min(127, rounded))
      row.append(int(clipped))
    Xq.append(row)

  Wq = []
  for i in range(k):
    row = []
    for j in range(m):
      s = safe_sw[j]
      val = W[i, j] / s
      rounded = round(val)
      clipped = max(-127, min(127, rounded))
      row.append(int(clipped))
    Wq.append(row)

  acc = []
  for i in range(n):
    row = []
    for j in range(m):
      s_acc = 0
      for l in range(k):
        s_acc += Xq[i][l] * Wq[l][j]
      row.append(s_acc)
    acc.append(row)

  Y = []
  for i in range(n):
    row = []
    for j in range(m):
      val = float(acc[i][j]) * (sx[i] * sw[j])
      row.append(val)
    Y.append(row)

  return np.asarray(Y, dtype=np.float64)
