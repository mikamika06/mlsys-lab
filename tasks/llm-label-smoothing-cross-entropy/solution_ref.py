import numpy as np

def label_smoothed_cross_entropy(logits: np.ndarray,
                                 targets: np.ndarray,
                                 eps: float = 0.1) -> float:
    """
    Compute the average label‑smoothed cross‑entropy loss.

    Parameters
    ----------
    logits : np.ndarray, shape (N, K)
        Raw model outputs for N examples and K classes.
    targets : np.ndarray, shape (N,)
        Integer class indices in [0, K-1].
    eps : float, default 0.1
        Label‑smoothing factor.

    Returns
    -------
    loss : float
        Mean smoothed cross‑entropy over the batch.
    """
    N, K = logits.shape

    # stable log‑softmax computation
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - logits_max)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_softmax = logits - logits_max - np.log(sum_exp)

    # smoothed target distribution
    y_onehot = np.zeros_like(logits)
    y_onehot[np.arange(N), targets] = 1.0
    y_smooth = (1 - eps) * y_onehot + eps / K

    loss_per_sample = -np.sum(y_smooth * log_softmax, axis=1)
    return float(np.mean(loss_per_sample))
