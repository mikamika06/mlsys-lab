import numpy as np

def knorm_press(keys: np.ndarray,
                values: np.ndarray,
                capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the top‑capacity key/value pairs sorted by descending key L2 norm.
    The relative order of selected tokens is preserved from the input.
    """
    n = keys.shape[0]
    if capacity <= 0:
        return keys[:0], values[:0]
    if capacity >= n:
        return keys.copy(), values.copy()
    norms = np.linalg.norm(keys, axis=1)
    # indices of top‑capacity by descending norm
    top_idx = np.argpartition(-norms, capacity - 1)[:capacity]
    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    return keys[mask], values[mask]
