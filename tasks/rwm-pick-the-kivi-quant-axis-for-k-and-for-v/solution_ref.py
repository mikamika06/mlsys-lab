import numpy as np

def pick_kivi_quant_axis(K: np.ndarray, V: np.ndarray) -> tuple[str, str]:
    """
    For each of K and V choose the axis (channel or token) that yields a lower
    group‑quantisation mean squared error.  The MSE is computed by summing the
    population variances over groups.
    """
    def _label(arr: np.ndarray) -> str:
        channel_var = np.var(arr, axis=0, ddof=0).sum()
        token_var   = np.var(arr, axis=1, ddof=0).sum()
        return "channel" if channel_var <= token_var else "token"

    k_axis = _label(K)
    v_axis = _label(V)
    return k_axis, v_axis
