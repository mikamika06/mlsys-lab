import numpy as np

def prune_ffn_neurons(
    up_proj: np.ndarray,
    down_proj: np.ndarray,
    target_width: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Return the indices of the most important neurons and the corresponding sliced
    projection matrices.

    Neuron importance is defined as the sum of absolute weights in both
    projections.  The top `target_width` neurons are kept.
    """
    importance = np.abs(up_proj).sum(axis=1) + np.abs(down_proj).sum(axis=0)
    idx = np.argsort(-importance)[:target_width]
    idx_sorted = np.sort(idx)

    up_sliced = up_proj[idx_sorted, :]
    down_sliced = down_proj[:, idx_sorted]

    return list(idx_sorted), up_sliced, down_sliced
