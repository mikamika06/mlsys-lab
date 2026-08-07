def apply_wanda_mask(W: list[list[float]], M: list[list[float]], X: list[list[float]]):
    """
    Apply a (precomputed) Wanda pruning mask M to weights W, and report
    both the pruned layer's output and how much it deviates from the
    unpruned layer's output on the same activations.

    Y = (W ⊙ M) @ X
    R = W @ X - Y     (the output residual introduced by pruning)

    W, M: (d_out, d_in). X: (d_in, n).
    Returns (Y, R), each (d_out, n).
    """
    d_out = len(W)
    d_in = len(W[0])
    n = len(X[0])

    Y = []
    R = []
    for i in range(d_out):
        y_row = []
        r_row = []
        for j in range(n):
            y_val = 0.0
            wx_val = 0.0
            for k in range(d_in):
                y_val += (W[i][k] * M[i][k]) * X[k][j]
                wx_val += W[i][k] * X[k][j]
            y_row.append(y_val)
            r_row.append(wx_val - y_val)
        Y.append(y_row)
        R.append(r_row)

    return Y, R
