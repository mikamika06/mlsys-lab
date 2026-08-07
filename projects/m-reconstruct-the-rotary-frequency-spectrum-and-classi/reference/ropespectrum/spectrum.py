import numpy as np

def reconstruct_spectrum(dim, base):
    indices = np.arange(0, dim, 2, dtype=np.float64)
    return 1.0 / (base ** (indices / dim))

def classify_dims(dim, base, threshold_freq):
    freqs = reconstruct_spectrum(dim, base)
    return ["high" if f >= threshold_freq else "low" for f in freqs]
