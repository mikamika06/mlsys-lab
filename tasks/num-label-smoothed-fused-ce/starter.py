import math

def label_smoothed_fused_ce(logits: list[list[float]], targets: list[int], eps: float=0.1) -> float:
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
    raise NotImplementedError('your code here')
