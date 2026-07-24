import numpy as np

def select_keep_sets(width_importance: np.ndarray,
                     layer_importance: np.ndarray,
                     d_target: int,
                     L_target: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return indices of the top‑d_target widths and top‑L_target layers.
    Indices are sorted in descending order of importance.
    """
    w_idx = np.argsort(-width_importance)[:d_target]
    l_idx = np.argsort(-layer_importance)[:L_target]
    return w_idx.astype(np.int64), l_idx.astype(np.int64)
