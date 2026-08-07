import numpy as np

def online_softmax(chunks):
    m = np.full((chunks[0].shape[0], 1), -np.inf)
    l = np.zeros((chunks[0].shape[0], 1))
    for c in chunks:
        block_max = np.max(c, axis=1, keepdims=True)
        new_m = np.maximum(m, block_max)
        alpha = np.exp(m - new_m)
        l = l * alpha + np.sum(np.exp(c - new_m), axis=1, keepdims=True)
        m = new_m
    return m, l
