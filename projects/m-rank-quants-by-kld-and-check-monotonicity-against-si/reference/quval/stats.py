import numpy as np


def derive_sample_size(ppl_a, ppl_b, variance, alpha=0.05, power=0.8):
    diff = abs(ppl_a - ppl_b)
    if diff == 0:
        return 1000000
    z_alpha = 1.96
    z_beta = 0.842
    n = (2 * variance * (z_alpha + z_beta) ** 2) / (diff ** 2)
    return int(np.ceil(n))
