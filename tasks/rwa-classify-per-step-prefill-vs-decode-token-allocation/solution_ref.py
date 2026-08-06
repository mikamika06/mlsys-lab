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

    n = budgets_arr.shape[0]
    prefill_list = []
    is_prefill_list = []

    for i in range(n):
        diff = int(budgets_arr[i]) - int(decode_arr[i])
        val = diff if diff > 0 else 0
        prefill_list.append(val)
        is_prefill_list.append(val > 0)

    prefill_tokens = np.array(prefill_list, dtype=np.int64)
    is_prefill = np.array(is_prefill_list, dtype=bool)

    return prefill_tokens, is_prefill
