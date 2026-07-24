import numpy as np

def count_live_params_and_sparsity(mask: np.ndarray) -> tuple[int, float]:
    """
    Count the number of non‑zero entries in a mask and compute its sparsity.

    Parameters
    ----------
    mask : np.ndarray
        2‑D array containing zeros and ones (or any truthy values).

    Returns
    -------
    live_count : int
        Number of non‑zero elements.
    sparsity : float
        Fraction of zero elements in the mask, in [0,1].
    """
    mask = np.asarray(mask)
    live_count = int(np.count_nonzero(mask))
    sparsity = float((mask.size - live_count) / mask.size)
    return live_count, sparsity
