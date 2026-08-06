import math
import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Scale an embedding matrix by 1/sqrt(d), where d is the feature dimension.
    The result is always float64 regardless of input dtype.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    rows, cols = emb.shape
    scale = math.sqrt(cols)
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = emb[i, j] / scale
    return out
