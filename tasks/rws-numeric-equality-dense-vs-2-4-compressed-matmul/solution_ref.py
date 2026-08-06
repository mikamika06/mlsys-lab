import numpy as np


def dense_vs_compressed24_matmul_error(W: np.ndarray, X: np.ndarray) -> float:
  W = np.asarray(W, dtype=np.float64)
  X = np.asarray(X, dtype=np.float64)
  m = W.shape[0]
  n = W.shape[1]

  if X.ndim == 1:
    Y_dense = np.zeros(m, dtype=np.float64)
    for i in range(m):
      acc = 0.0
      for j in range(n):
        acc += W[i, j] * X[j]
      Y_dense[i] = acc
  else:
    k = X.shape[1]
    Y_dense = np.zeros((m, k), dtype=np.float64)
    for i in range(m):
      for jj in range(k):
        acc = 0.0
        for j in range(n):
          acc += W[i, j] * X[j, jj]
        Y_dense[i, jj] = acc

  W_recon_list = [[0.0] * n for _ in range(m)]
  n_groups = n // 4
  for g in range(n_groups):
    col_start = g * 4
    for i in range(m):
      for j in range(4):
        val = W[i, col_start + j]
        if val != 0.0:
          W_recon_list[i][col_start + j] = val
  W_recon = np.array(W_recon_list, dtype=np.float64)

  if X.ndim == 1:
    Y_compressed = np.zeros(m, dtype=np.float64)
    for i in range(m):
      acc = 0.0
      for j in range(n):
        acc += W_recon[i, j] * X[j]
      Y_compressed[i] = acc
  else:
    k = X.shape[1]
    Y_compressed = np.zeros((m, k), dtype=np.float64)
    for i in range(m):
      for jj in range(k):
        acc = 0.0
        for j in range(n):
          acc += W_recon[i, j] * X[j, jj]
        Y_compressed[i, jj] = acc

  max_err = 0.0
  if Y_dense.ndim == 1:
    for i in range(m):
      diff = Y_dense[i] - Y_compressed[i]
      if diff < 0.0:
        diff = -diff
      if diff > max_err:
        max_err = diff
  else:
    shape0, shape1 = Y_dense.shape
    for i in range(shape0):
      for j in range(shape1):
        diff = Y_dense[i, j] - Y_compressed[i, j]
        if diff < 0.0:
          diff = -diff
        if diff > max_err:
          max_err = diff

  return float(max_err)
