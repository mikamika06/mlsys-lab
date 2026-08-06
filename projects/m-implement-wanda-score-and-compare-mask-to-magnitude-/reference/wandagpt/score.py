import numpy as np


def wanda_score(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute Wanda score: |W_ij| * ||X_j||_2."""
    x_norm = np.linalg.norm(X, axis=0)
    return np.abs(W) * x_norm[None, :]


def create_mask_from_score(score: np.ndarray, sparsity: float) -> np.ndarray:
    """Create binary mask keeping highest score weights."""
    if sparsity <= 0.0:
        return np.ones_like(score, dtype=bool)
    if sparsity >= 1.0:
        return np.zeros_like(score, dtype=bool)

    k = int(np.round(score.size * (1.0 - sparsity)))
    k = max(0, min(score.size, k))
    if k == 0:
        return np.zeros_like(score, dtype=bool)
    if k == score.size:
        return np.ones_like(score, dtype=bool)

    flat = score.ravel()
    threshold = np.partition(flat, score.size - k)[score.size - k]
    return score >= threshold


def magnitude_mask(W: np.ndarray, sparsity: float) -> np.ndarray:
    """Create binary mask based on absolute weight values."""
    return create_mask_from_score(np.abs(W), sparsity)
