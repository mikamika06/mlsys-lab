def imatrix_from_calibration(X: list[list[float]]) -> list[float]:
    """Per-input-channel importance: sum over calibration tokens of activation^2.

    X has shape (n_tokens, n_channels); returns a 1-D list of length n_channels.
    """
    n_tokens = len(X)
    n_channels = len(X[0]) if n_tokens > 0 else 0
    result = [0.0] * n_channels
    for j in range(n_channels):
        acc = 0.0
        for i in range(n_tokens):
            val = X[i][j]
            acc += val * val
        result[j] = acc
    return result
