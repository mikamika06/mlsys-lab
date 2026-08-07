def compressed_matmul(values: list[list[float]], idx: list[list[int]], X: list[list[float]]) -> list[list[float]]:
    """
    Reconstruct the dense weight matrix from NVIDIA-style compressed 2:4
    storage (2 nonzero values per group of 4 columns, plus a 2-bit index
    per value giving its position 0..3 within the group), then compute
    the dense matmul W @ X.
    """
    d_out = len(values)
    half = len(values[0])
    d_in = half * 2
    n = len(X[0])

    # Reconstruct dense weight matrix W
    W = [[0.0 for _ in range(d_in)] for _ in range(d_out)]
    for r in range(d_out):
        for k in range(half):
            g = k // 2
            col = 4 * g + idx[r][k]
            W[r][col] = values[r][k]

    # Compute Y = W @ X
    Y = [[0.0 for _ in range(n)] for _ in range(d_out)]
    for r in range(d_out):
        for c in range(n):
            total = 0.0
            for j in range(d_in):
                total += W[r][j] * X[j][c]
            Y[r][c] = total

    return Y
