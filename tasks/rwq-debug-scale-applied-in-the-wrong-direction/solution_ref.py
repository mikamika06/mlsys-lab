def apply_migration_scale(X: list[list[float]], W: list[list[float]], s: list[float]) -> tuple:
    """
    AWQ-style migration scale: shrink activations, grow the matching weight
    rows, so the product X @ W is unchanged.
    """
    b = len(X)
    d_in = len(s)
    d_out = len(W[0])

    X_prime = [[X[i][j] / s[j] for j in range(d_in)] for i in range(b)]
    W_prime = [[W[j][k] * s[j] for k in range(d_out)] for j in range(d_in)]

    return X_prime, W_prime
