def dense_vs_compressed24_matmul_error(W: list[list[float]], X: list[list[float]]) -> float:
  m = len(W)
  n = len(W[0])

  if isinstance(X[0], (int, float)):
    Y_dense = [0.0] * m
    for i in range(m):
      acc = 0.0
      for j in range(n):
        acc += W[i][j] * X[j]
      Y_dense[i] = acc
  else:
    k = len(X[0])
    Y_dense = [[0.0] * k for _ in range(m)]
    for i in range(m):
      for jj in range(k):
        acc = 0.0
        for j in range(n):
          acc += W[i][j] * X[j][jj]
        Y_dense[i][jj] = acc

  W_recon = [[0.0] * n for _ in range(m)]
  n_groups = n // 4
  for g in range(n_groups):
    col_start = g * 4
    for i in range(m):
      for j in range(4):
        val = W[i][col_start + j]
        if val != 0.0:
          W_recon[i][col_start + j] = val

  if isinstance(X[0], (int, float)):
    Y_compressed = [0.0] * m
    for i in range(m):
      acc = 0.0
      for j in range(n):
        acc += W_recon[i][j] * X[j]
      Y_compressed[i] = acc
  else:
    k = len(X[0])
    Y_compressed = [[0.0] * k for _ in range(m)]
    for i in range(m):
      for jj in range(k):
        acc = 0.0
        for j in range(n):
          acc += W_recon[i][j] * X[j][jj]
        Y_compressed[i][jj] = acc

  max_err = 0.0
  if isinstance(X[0], (int, float)):
    for i in range(m):
      diff = Y_dense[i] - Y_compressed[i]
      if diff < 0.0:
        diff = -diff
      if diff > max_err:
        max_err = diff
  else:
    shape0 = len(Y_dense)
    shape1 = len(Y_dense[0])
    for i in range(shape0):
      for j in range(shape1):
        diff = Y_dense[i][j] - Y_compressed[i][j]
        if diff < 0.0:
          diff = -diff
        if diff > max_err:
          max_err = diff

  return float(max_err)
