import numpy as np

def lookup_embeddings(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ids_arr = np.asarray(ids, dtype=np.int64)
    return np.take(weights, ids_arr, axis=0).astype(np.float64)
