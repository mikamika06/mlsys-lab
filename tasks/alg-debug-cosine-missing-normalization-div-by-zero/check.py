import numpy as np
from mlsys import scorers

def _reference_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity with proper normalization and zero‑norm handling."""
    dot = A @ B.T  # shape (n_q, n_c)
    norm_a = np.linalg.norm(A, axis=1)          # shape (n_q,)
    norm_b = np.linalg.norm(B, axis=1)          # shape (n_c,)
    denom = np.outer(norm_a, norm_b)            # shape (n_q, n_c)
    # Where denominator is zero (i.e., a zero vector), set similarity to 0.0
    sim = np.divide(dot, denom,
                    out=np.zeros_like(dot, dtype=np.float64),
                    where=denom != 0)
    return sim

def grade(sol, fx) -> dict:
    # Test case 1: normal query vs two candidates
    queries1 = np.array([[1, 1]])
    candidates = np.array([[100, 0], [1, 1]])

    # Test case 2: zero‑norm query
    queries2 = np.array([[0, 0]])

    # Combine into a single test set
    queries = np.vstack([queries1, queries2])

    ref = _reference_cosine(queries, candidates)
    try:
        got = sol.cosine_similarity(queries, candidates)
    except Exception as e:
        return {"argmax_agreement": 0.0}

    # Ensure output is float64
    if got.dtype != np.float64:
        return {"argmax_agreement": 0.0}

    # Compute argmax agreement
    agree = scorers.argmax_agreement(ref, got)
    return {"argmax_agreement": agree}
