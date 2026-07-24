import numpy as np


def sample_sequence(logits: np.ndarray, temperature: float, seed: int) -> np.ndarray:
    """Reproduce a seeded temperature-sampled id sequence via inverse-CDF draws.

    One np.random.default_rng(seed) is created, and one uniform is consumed per
    decode step, in order. Returns an int64 array of shape (T,).
    """
    logits = np.asarray(logits, dtype=np.float64)
    T, V = logits.shape
    rng = np.random.default_rng(seed)
    ids = np.empty(T, dtype=np.int64)
    for t in range(T):
        z = logits[t] / temperature
        z = z - np.max(z)
        e = np.exp(z)
        p = e / e.sum()
        cdf = np.cumsum(p)
        u = rng.random()
        idx = int(np.searchsorted(cdf, u, side="right"))
        if idx >= V:
            idx = V - 1
        ids[t] = idx
    return ids
