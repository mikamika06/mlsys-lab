import numpy as np

def importance_scores(
    structures: dict[int, tuple[np.ndarray, np.ndarray, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute three importance signals for each structure.

    Parameters
    ----------
    structures : dict[int, tuple[np.ndarray, np.ndarray, float]]
        Mapping from a structure identifier to a tuple of:
          * activations (1-D array)
          * gradients (1-D array, same shape as activations)
          * weight scalar

    Returns
    -------
    mean_abs : np.ndarray
        Mean absolute activation per structure.
    l2_norm : np.ndarray
        Euclidean norm of the activation vector.
    taylor_imp : np.ndarray
        Mean absolute product of gradient and weight.
    """
    keys = sorted(structures.keys())
    mean_abs = np.empty(len(keys), dtype=np.float64)
    l2_norm = np.empty(len(keys), dtype=np.float64)
    taylor_imp = np.empty(len(keys), dtype=np.float64)

    for idx, k in enumerate(keys):
        act, grad, w = structures[k]
        act = np.asarray(act, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)

        mean_abs[idx] = np.mean(np.abs(act))
        l2_norm[idx] = np.linalg.norm(act)
        taylor_imp[idx] = np.mean(np.abs(grad * w))

    return mean_abs, l2_norm, taylor_imp
