import numpy as np

def select_keep_sets(width_importance: np.ndarray,
                     layer_importance: np.ndarray,
                     d_target: int,
                     L_target: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return indices of the top‑d_target widths and top‑L_target layers.
    Indices are sorted in descending order of importance.
    """
    w_pairs = []
    for i in range(len(width_importance)):
        w_pairs.append((width_importance[i], i))
    w_sorted = sorted(w_pairs, key=lambda x: x[0], reverse=True)
    w_idx_list = []
    for i in range(d_target):
        w_idx_list.append(w_sorted[i][1])

    l_pairs = []
    for i in range(len(layer_importance)):
        l_pairs.append((layer_importance[i], i))
    l_sorted = sorted(l_pairs, key=lambda x: x[0], reverse=True)
    l_idx_list = []
    for i in range(L_target):
        l_idx_list.append(l_sorted[i][1])

    return np.array(w_idx_list, dtype=np.int64), np.array(l_idx_list, dtype=np.int64)
