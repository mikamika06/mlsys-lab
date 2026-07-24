import numpy as np

def combine_softmax_stats(block_a, block_b):
    m1, s1, w1 = block_a
    m2, s2, w2 = block_b
    m = max(m1, m2)
    shift1 = m1 - m
    shift2 = m2 - m
    s = s1 * np.exp(shift1) + s2 * np.exp(shift2)
    w = w1 * np.exp(shift1) + w2 * np.exp(shift2)
    return (float(m), float(s), float(w))
