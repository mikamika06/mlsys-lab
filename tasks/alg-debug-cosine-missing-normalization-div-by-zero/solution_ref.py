import numpy as np

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Return the matrix of cosine similarities between rows of A and B.
    Handles zero‑norm vectors by returning 0.0 for any pair involving a zero vector.
    """
    dot = A @ B.T
    norm_a = np.linalg.norm(A, axis=1)
    norm_b = np.linalg.norm(B, axis=1)
    denom = np.outer(norm_a, norm_b)
    # Avoid division by zero: where denom == 0 set similarity to 0.0
    sim = np.divide(dot, denom,
                    out=np.zeros_like(dot, dtype=np.float64),
                    where=denom != 0)
    return sim.astype(np.float64)
