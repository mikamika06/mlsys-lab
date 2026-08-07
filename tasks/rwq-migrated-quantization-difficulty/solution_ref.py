_EPS = 1e-12


def channel_peakiness_before_after(X: list[list[float]], W: list[list[float]], alpha: float = 0.5) -> tuple[list[float], list[float]]:
    """SmoothQuant channel-imbalance ("quantization difficulty") before/after migration.

    A per-tensor activation quantizer sets one scale from the single largest
    channel. If a handful of channels have activation magnitude far above the
    rest, that shared scale crushes every other channel -- this is the
    "quantization difficulty" SmoothQuant migrates from activations to
    weights.

    Parameters
    ----------
    X : list of list of float, shape (n_tokens, C)
        Activation samples (rows = tokens, columns = channels).
    W : list of list of float, shape (n_out, C)
        Weight matrix; column j lines up with activation channel j.
    alpha : float
        SmoothQuant migration strength in [0, 1].

    Returns
    -------
    ratio_before : list of float, shape (C,)
        amax_X[j] / mean(amax_X), i.e. how far channel j's peak activation
        magnitude sits above the average channel's peak magnitude.
    ratio_after : list of float, shape (C,)
        The same ratio computed after per-channel migration:
        amax_X[j] / s[j], where
        s[j] = amax_X[j] ** alpha / amax_W[j] ** (1 - alpha).
    """
    n_tokens = len(X)
    C = len(X[0])
    n_out = len(W)

    amax_X_list = []
    for j in range(C):
        m = abs(X[0][j])
        for i in range(1, n_tokens):
            val = abs(X[i][j])
            if val > m:
                m = val
        amax_X_list.append(m)

    amax_W_list = []
    for j in range(C):
        m = abs(W[0][j])
        for i in range(1, n_out):
            val = abs(W[i][j])
            if val > m:
                m = val
        amax_W_list.append(m)

    sum_amax_X = 0.0
    for val in amax_X_list:
        sum_amax_X += val
    mean_amax_X = sum_amax_X / C
    denom_before = max(mean_amax_X, _EPS)

    ratio_before = []
    for val in amax_X_list:
        ratio_before.append(val / denom_before)

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

    ratio_after = []
    for val in amax_X_smoothed_list:
        ratio_after.append(val / denom_after)

    return ratio_before, ratio_after
