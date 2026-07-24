import numpy as np

def tied_head_logits(embedding_matrix: np.ndarray) -> np.ndarray:
    """Return logits computed with weight tying."""
    return embedding_matrix.astype(np.float64) @ embedding_matrix.T
