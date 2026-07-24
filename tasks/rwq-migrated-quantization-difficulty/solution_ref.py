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

    amax_X = np.max(np.abs(X), axis=0)
    amax_W = np.max(np.abs(W), axis=0)

    ratio_before = amax_X / max(float(np.mean(amax_X)), _EPS)

    s = (np.maximum(amax_X, _EPS) ** alpha) / (np.maximum(amax_W, _EPS) ** (1.0 - alpha))
    amax_X_smoothed = amax_X / s
    ratio_after = amax_X_smoothed / max(float(np.mean(amax_X_smoothed)), _EPS)

    return ratio_before, ratio_after
