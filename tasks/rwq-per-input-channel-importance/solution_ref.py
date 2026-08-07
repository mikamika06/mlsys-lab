def per_input_channel_importance(X: list[list[list[float]]]) -> list[float]:
    """
    Compute the mean absolute activation for each input channel.

    Parameters
    ----------
    X : list[list[list[float]]]
        Input tensor of shape (B, T, C).

    Returns
    -------
    list[float]
        1‑D list of length C containing the per‑channel importance.
    """
    B = len(X)
    T = len(X[0])
    C = len(X[0][0])
    out = [0.0] * C
    total_elements = B * T
    for c in range(C):
        acc = 0.0
        for b in range(B):
            for t in range(T):
                val = X[b][t][c]
                if val < 0.0:
                    val = -val
                acc += val
        out[c] = acc / total_elements
    return out
