import math
import numpy as np


def cholesky_inverse(H: np.ndarray) -> np.ndarray:
  """
  Invert an SPD matrix H via its Cholesky factor: H = L L^T, then
  H^-1 = (L^-1)^T (L^-1). Never forms H^-1 through a generic elimination
  path; only ever solves against the triangular factor L.
  """
  H_arr = np.asarray(H, dtype=np.float64)
  n = H_arr.shape[0]

  H_list = [[H_arr[i, j] for j in range(n)] for i in range(n)]

  L = [[0.0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
    for j in range(i + 1):
      s = H_list[i][j]
      for k in range(j):
        s -= L[i][k] * L[j][k]
      if i == j:
        L[i][j] = math.sqrt(s)
      else:
        L[i][j] = s / L[j][j]

  L_inv = [[0.0 for _ in range(n)] for _ in range(n)]
  for j in range(n):
    for i in range(n):
      s = 1.0 if i == j else 0.0
      for k in range(i):
        s -= L[i][k] * L_inv[k][j]
      L_inv[i][j] = s / L[i][i]

  res = [[0.0 for _ in range(n)] for _ in range(n)]
  for r in range(n):
    for c in range(n):
      s = 0.0
      for k in range(n):
        s += L_inv[k][r] * L_inv[k][c]
      res[r][c] = s

  return np.array(res, dtype=np.float64)
