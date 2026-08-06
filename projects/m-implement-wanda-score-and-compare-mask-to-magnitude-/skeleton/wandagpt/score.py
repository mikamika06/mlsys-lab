import numpy as np


def wanda_score(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute Wanda score for weight matrix W and activation matrix X."""
    raise NotImplementedError


def create_mask_from_score(score: np.ndarray, sparsity: float) -> np.ndarray:
    """Create binary mask keeping highest scoring entries."""
    raise NotImplementedError


def magnitude_mask(W: np.ndarray, sparsity: float) -> np.ndarray:
    """Create binary mask keeping highest magnitude weights."""
    raise NotImplementedError
