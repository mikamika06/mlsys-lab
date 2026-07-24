import numpy as np

def classify_prefill_decode(budgets: np.ndarray,
                            decode_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per‑step prefill allocation and a mask indicating whether any
    prefill tokens were allocated.

    Parameters
    ----------
    budgets : np.ndarray
        1‑D array of non‑negative integers representing the token budget at each step.
    decode_counts : np.ndarray
        1‑D array of the same shape as `budgets` containing the number of decode tokens required per step.

    Returns
    -------
    prefill_tokens : np.ndarray
        Integer array of the same shape with $p_i = \max(0, b_i - d_i)$.
    is_prefill : np.ndarray
        Boolean array that is True iff $p_i > 0$.
    """
    budgets_arr = np.asarray(budgets)
    decode_arr = np.asarray(decode_counts)

    if budgets_arr.shape != decode_arr.shape:
        raise ValueError("budgets and decode_counts must have the same shape")

    prefill_tokens = budgets_arr - decode_arr
    prefill_tokens = np.maximum(prefill_tokens, 0).astype(np.int64)
    is_prefill = prefill_tokens > 0

    return prefill_tokens, is_prefill
