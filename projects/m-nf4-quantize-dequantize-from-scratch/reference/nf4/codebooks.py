import numpy as np


def norm_ppf(p):
    t = np.sqrt(-2.0 * np.log(p if p < 0.5 else 1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    val = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    return -val if p < 0.5 else val


def build_int4_codebook():
    return np.linspace(-1.0, 1.0, 16)


def build_fp4_codebook():
    return np.array([
        -1.0, -0.666, -0.5, -0.333, -0.25, -0.166, -0.083, -0.0,
         0.0,  0.083,  0.166,  0.25,  0.333,  0.5,  0.666,  1.0
    ])


def build_nf4_codebook():
    p_neg = np.linspace(0.03, 0.5, 8)
    d = p_neg[1] - p_neg[0]
    p_pos = np.linspace(0.5 + d, 0.97, 8)
    p_all = np.concatenate([p_neg, p_pos])
    quantiles = np.array([norm_ppf(p) for p in p_all])
    return quantiles / np.max(np.abs(quantiles))
