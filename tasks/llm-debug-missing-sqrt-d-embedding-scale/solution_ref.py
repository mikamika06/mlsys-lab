import numpy as np

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Scale an embedding matrix by 1/sqrt(d), where d is the feature dimension.
    The result is always float64 regardless of input dtype.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    d = emb.shape[1]
    return emb / np.sqrt(d)
