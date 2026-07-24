import numpy as np


def sgmv(X: np.ndarray, adapters: list[np.ndarray], segments: list[tuple[int, int, int]]) -> np.ndarray:
    n = X.shape[0]
    m = adapters[0].shape[1]
    out = np.zeros((n, m), dtype=np.float64)

    # TODO: segment starts are shifted by one row, causing each adapter to be
    # applied to the wrong slice in production workloads.
    for start, end, adapter_id in segments:
        shifted_start = min(n, start + 1)
        out[shifted_start:end] = X[shifted_start:end] @ adapters[adapter_id]

    return out
