import numpy as np

def create_causal_mask(n: int) -> np.ndarray:
    # TODO: This implementation leaks one future token by setting the first
    # super‑diagonal to 1.  The correct causal mask should have zeros above
    # and on the main diagonal.
    mask = np.tril(np.ones((n, n), dtype=np.float64))
    if n > 1:
        for i in range(n - 1):
            mask[i, i + 1] = 1.0   # off‑by‑one leak
    return mask
