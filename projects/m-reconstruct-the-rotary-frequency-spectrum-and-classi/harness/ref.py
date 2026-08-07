import numpy as np

CONFIGS = [
    {"dim": 64, "base": 10000.0, "max_seq_len": 4096},
    {"dim": 128, "base": 500000.0, "max_seq_len": 32768},
    {"dim": 256, "base": 1000000.0, "max_seq_len": 131072}
]

def reconstruct_spectrum(dim, base):
    indices = np.arange(0, dim, 2, dtype=np.float64)
    freqs = 1.0 / (base ** (indices / dim))
    return freqs

def classify_dims(dim, base, threshold_freq):
    freqs = reconstruct_spectrum(dim, base)
    classes = []
    for f in freqs:
        if f >= threshold_freq:
            classes.append("high")
        else:
            classes.append("low")
    return classes

def simulate_overflow(pos_ids, dim, base, max_len):
    freqs = reconstruct_spectrum(dim, base)
    angles = np.outer(pos_ids, freqs)
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    overflow = pos_ids > max_len
    return embeddings, overflow
