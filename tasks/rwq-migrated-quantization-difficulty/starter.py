import numpy as np


def channel_peakiness_before_after(X, W, alpha=0.5):
    """SmoothQuant channel-imbalance ("quantization difficulty") before/after migration.

    X: (n_tokens, C) activation samples. W: (n_out, C) weight matrix, column
    j aligned with activation channel j. alpha: SmoothQuant migration
    strength in [0, 1].

    Let amax_X[j] = max over tokens of |X[:, j]| and amax_W[j] = max over
    rows of |W[:, j]|.

        ratio_before[j] = amax_X[j] / mean(amax_X)
        s[j]            = amax_X[j] ** alpha / amax_W[j] ** (1 - alpha)
        ratio_after[j]  = (amax_X[j] / s[j]) / mean(amax_X / s)

    Returns (ratio_before, ratio_after): float64 arrays of shape (C,).
    """
    raise NotImplementedError('your code here')
