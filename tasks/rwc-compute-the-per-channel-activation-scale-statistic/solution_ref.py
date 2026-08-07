def compute_activation_scale(X: list[list[list[float]]]) -> list[float]:
    """
    Compute per‑channel mean absolute activation.

    Parameters
    ----------
    X : list[list[list[float]]]
        3‑D list of shape (batch, seq_len, channels).

    Returns
    -------
    list[float]
        1‑D list of length `channels` containing the statistic.
    """
    batch_size = len(X)
    seq_len = len(X[0])
    channels = len(X[0][0])
    total_elements = batch_size * seq_len

    result = [0.0] * channels

    for c in range(channels):
        acc = 0.0
        for b in range(batch_size):
            for s in range(seq_len):
                val = X[b][s][c]
                if val < 0.0:
                    acc -= val
                else:
                    acc += val
        result[c] = acc / total_elements

    return result
