import numpy as np


def absorb_query_weight(W_Q: np.ndarray, W_UK: np.ndarray) -> np.ndarray:
    return W_Q.astype(np.float64).T @ W_UK.astype(np.float64)
