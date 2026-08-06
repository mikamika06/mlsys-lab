import numpy as np

def compute_token_mask(vocab_size, allowed_tokens):
    m = np.zeros(vocab_size, dtype=bool)
    m[allowed_tokens] = True
    return m
