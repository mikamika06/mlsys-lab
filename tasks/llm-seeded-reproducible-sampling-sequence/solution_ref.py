import math
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
        
        max_z = z[0]
        for i in range(1, V):
            if z[i] > max_z:
                max_z = z[i]
                
        z_shifted = z - max_z
        
        e = np.empty(V, dtype=np.float64)
        for i in range(V):
            e[i] = math.exp(z_shifted[i])
            
        sum_e = 0.0
        for i in range(V):
            sum_e += e[i]
            
        p = np.empty(V, dtype=np.float64)
        for i in range(V):
            p[i] = e[i] / sum_e
            
        cdf = np.empty(V, dtype=np.float64)
        acc = 0.0
        for i in range(V):
            acc += p[i]
            cdf[i] = acc
            
        u = rng.random()
        
        idx = V
        for i in range(V):
            if cdf[i] > u:
                idx = i
                break
                
        if idx >= V:
            idx = V - 1
        ids[t] = idx
    return ids
