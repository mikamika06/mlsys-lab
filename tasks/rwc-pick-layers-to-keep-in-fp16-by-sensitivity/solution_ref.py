import numpy as np

def select_fp16_layers(errors: np.ndarray, k: int) -> list[int]:
    """
    Return the indices of the top‑k layers with largest FP8‑KV error.

    Parameters
    ----------
    errors : np.ndarray
        1‑D array of per‑layer errors.
    k : int
        Number of layers to keep in FP16 (0 ≤ k ≤ len(errors)).

    Returns
    -------
    list[int]
        Indices sorted by decreasing error; ties broken by increasing index.
    """
    if k == 0:
        return []
    # Stable tie‑break: lower index first when errors equal.
    order = np.lexsort((np.arange(len(errors)), -errors))
    topk = order[:k]
    return list(topk)
