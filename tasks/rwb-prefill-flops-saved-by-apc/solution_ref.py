import numpy as np

def flops_saved_by_apc(reused_counts: np.ndarray, per_token_flop: float) -> float:
    """
    Compute the total FLOPs saved by APC.

    Parameters
    ----------
    reused_counts : np.ndarray
        1‑D array of non‑negative integers representing how many times each token was reused.
    per_token_flop : float
        FLOPs saved per reuse of a single token.

    Returns
    -------
    float
        Total FLOPs saved.
    """
    counts = np.asarray(reused_counts, dtype=np.int64)
    total_reuses = 0
    for i in range(len(counts)):
        total_reuses += counts[i]
    return float(total_reuses * per_token_flop)
