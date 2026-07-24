import numpy as np


def flash_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tile_size: int) -> np.ndarray:
    """Compute tiled attention with a running softmax state."""
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    m = np.full(n, -np.inf, dtype=np.float64)
    l = np.zeros(n, dtype=np.float64)
    O = np.zeros((n, d), dtype=np.float64)

    for start in range(0, K.shape[0], tile_size):
        end = min(K.shape[0], start + tile_size)

        scores = Q @ K[start:end].T * scale

        tile_max = np.max(scores, axis=1)
        new_m = np.maximum(m, tile_max)

        alpha = np.exp(m - new_m)
        exp_scores = np.exp(scores - new_m[:, None])

        O = O + exp_scores @ V[start:end]
        l = l * alpha + np.sum(exp_scores, axis=1)

        m = new_m

    return O / l[:, None]
