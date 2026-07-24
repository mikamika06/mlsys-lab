import numpy as np


def sgmv(X: np.ndarray, adapters: list[np.ndarray], segments: list[tuple[int, int, int]]) -> np.ndarray:
    n = X.shape[0]
    m = adapters[0].shape[1]
    out = np.zeros((n, m), dtype=np.float64)
    for start, end, adapter_id in segments:
        out[start:end] = X[start:end] @ adapters[adapter_id]
    return out
