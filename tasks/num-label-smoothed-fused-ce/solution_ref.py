import numpy as np
import math

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
    q_smooth = np.full((N, K), eps / K, dtype=np.float64)
    for i in range(N):
        q_smooth[i, targets[i]] += 1.0 - eps

    m = np.empty((N, 1), dtype=np.float64)
    log_Z = np.empty((N, 1), dtype=np.float64)
    log_p = np.empty((N, K), dtype=np.float64)
    losses = np.empty((N,), dtype=np.float64)

    for i in range(N):
        max_val = logits[i, 0]
        for j in range(1, K):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        m[i, 0] = max_val

        sum_exp = 0.0
        for j in range(K):
            sum_exp += math.exp(logits[i, j] - m[i, 0])
        log_Z[i, 0] = math.log(sum_exp)

        loss_i = 0.0
        for j in range(K):
            val = logits[i, j] - m[i, 0] - log_Z[i, 0]
            log_p[i, j] = val
            loss_i -= q_smooth[i, j] * val
        losses[i] = loss_i

    total_loss = 0.0
    for i in range(N):
        total_loss += losses[i]
    return float(total_loss / N)
