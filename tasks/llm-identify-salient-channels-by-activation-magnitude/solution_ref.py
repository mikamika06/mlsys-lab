import math

def salient_channels(X: list[list[float]], fraction: float = 0.1) -> list[int]:
    """
    Return indices of the top fraction of channels by mean absolute activation.
    """
    if not (0 <= fraction <= 1):
        raise ValueError("fraction must be in [0, 1]")
    n_samples = len(X)
    n_channels = len(X[0]) if n_samples > 0 else 0
    k = math.ceil(fraction * n_channels)
    if k == 0 or n_channels == 0:
        return []

    mean_abs = [0.0] * n_channels
    for c in range(n_channels):
        s = 0.0
        for r in range(n_samples):
            v = X[r][c]
            if v < 0:
                v = -v
            s += v
        mean_abs[c] = s / n_samples

    idx = list(range(n_channels))
    for i in range(1, n_channels):
        key = idx[i]
        j = i - 1
        while j >= 0:
            if mean_abs[key] > mean_abs[idx[j]] or (mean_abs[key] == mean_abs[idx[j]] and key < idx[j]):
                idx[j + 1] = idx[j]
                j -= 1
            else:
                break
        idx[j + 1] = key

    topk = idx[:k]
    for i in range(1, k):
        key = topk[i]
        j = i - 1
        while j >= 0 and topk[j] > key:
            topk[j + 1] = topk[j]
            j -= 1
        topk[j + 1] = key

    return topk
