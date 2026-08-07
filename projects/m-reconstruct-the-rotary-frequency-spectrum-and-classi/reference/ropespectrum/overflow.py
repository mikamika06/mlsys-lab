import numpy as np
from ropespectrum.spectrum import reconstruct_spectrum

def simulate_overflow(pos_ids, dim, base, max_len):
    freqs = reconstruct_spectrum(dim, base)
    angles = np.outer(pos_ids, freqs)
    embeddings = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    overflow = pos_ids > max_len
    return embeddings, overflow
