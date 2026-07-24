import numpy as np

def label_smoothed_fused_ce(logits, targets, eps=0.1):
    """
    Numerically stable label-smoothed cross-entropy.

    Parameters
    ----------
    logits  : np.ndarray, shape (N, K) — unnormalized scores
    targets : np.ndarray, shape (N,)   — integer class indices
    eps     : float                    — smoothing factor in [0, 1]

    Returns
    -------
    float — mean cross-entropy over the batch
    """
    N, K = logits.shape
    # Build smoothed target distribution for each sample
    q_smooth = np.full((N, K), eps / K, dtype=np.float64)
    q_smooth[np.arange(N), targets] += 1.0 - eps
    # Fused log-sum-exp for numerical stability
    m = np.max(logits, axis=1, keepdims=True)
    log_Z = np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))
    log_p = logits - m - log_Z
    # Cross-entropy: -sum_k q_k * log p_k, averaged over batch
    losses = -np.sum(q_smooth * log_p, axis=1)
    return float(np.mean(losses))
