import numpy as np

_EPS = 1e-12


def channel_peakiness_before_after(X, W, alpha=0.5):
    """SmoothQuant channel-imbalance ("quantization difficulty") before/after migration.

    A per-tensor activation quantizer sets one scale from the single largest
    channel. If a handful of channels have activation magnitude far above the
    rest, that shared scale crushes every other channel -- this is the
    "quantization difficulty" SmoothQuant migrates from activations to
    weights.

    Parameters
    ----------
    X : np.ndarray, shape (n_tokens, C)
        Activation samples (rows = tokens, columns = channels).
    W : np.ndarray, shape (n_out, C)
        Weight matrix; column j lines up with activation channel j.
    alpha : float
        SmoothQuant migration strength in [0, 1].

    Returns
    -------
    ratio_before : np.ndarray, float64, shape (C,)
        amax_X[j] / mean(amax_X), i.e. how far channel j's peak activation
        magnitude sits above the average channel's peak magnitude.
    ratio_after : np.ndarray, float64, shape (C,)
        The same ratio computed after per-channel migration:
        amax_X[j] / s[j], where
        s[j] = amax_X[j] ** alpha / amax_W[j] ** (1 - alpha).
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    n_tokens, C = X.shape
    n_out = W.shape[0]

    amax_X_list = []
    for j in range(C):
        m = abs(X[0, j])
        for i in range(1, n_tokens):
            val = abs(X[i, j])
            if val > m:
                m = val
        amax_X_list.append(m)

    amax_W_list = []
    for j in range(C):
        m = abs(W[0, j])
        for i in range(1, n_out):
            val = abs(W[i, j])
            if val > m:
                m = val
        amax_W_list.append(m)

    sum_amax_X = 0.0
    for val in amax_X_list:
        sum_amax_X += val
    mean_amax_X = sum_amax_X / C
    denom_before = max(mean_amax_X, _EPS)

    ratio_before_list = []
    for val in amax_X_list:
        ratio_before_list.append(val / denom_before)

    s_list = []
    for j in range(C):
        num = max(amax_X_list[j], _EPS) ** alpha
        den = max(amax_W_list[j], _EPS) ** (1.0 - alpha)
        s_list.append(num / den)

    amax_X_smoothed_list = []
    for j in range(C):
        amax_X_smoothed_list.append(amax_X_list[j] / s_list[j])

    sum_smoothed = 0.0
    for val in amax_X_smoothed_list:
        sum_smoothed += val
    mean_smoothed = sum_smoothed / C
    denom_after = max(mean_smoothed, _EPS)

    ratio_after_list = []
    for val in amax_X_smoothed_list:
        ratio_after_list.append(val / denom_after)

    ratio_before = np.asarray(ratio_before_list, dtype=np.float64)
    ratio_after = np.asarray(ratio_after_list, dtype=np.float64)

    return ratio_before, ratio_after
