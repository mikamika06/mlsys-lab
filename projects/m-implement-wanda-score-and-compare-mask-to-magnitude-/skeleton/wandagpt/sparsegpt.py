import numpy as np


def simplified_sparsegpt(
    W: np.ndarray, X: np.ndarray, sparsity: float, damping: float = 1e-4
) -> np.ndarray:
    """Perform single-layer OBS weight updates to prune W given activations X."""
    raise NotImplementedError
