import numpy as np


def norm_ppf(p):
    """
    Returns the quantile function (inverse CDF) of the standard normal distribution.
    Use this to build the NF4 codebook.
    """
    t = np.sqrt(-2.0 * np.log(p if p < 0.5 else 1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    val = t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)
    return -val if p < 0.5 else val


def build_int4_codebook():
    """
    Return a 16-element numpy array of linearly spaced values from -1.0 to 1.0.
    """
    raise NotImplementedError()


def build_fp4_codebook():
    """
    Return the standard 16-element FP4 codebook.
    """
    return np.array([
        -1.0, -0.666, -0.5, -0.333, -0.25, -0.166, -0.083, -0.0,
         0.0,  0.083,  0.166,  0.25,  0.333,  0.5,  0.666,  1.0
    ])


def build_nf4_codebook():
    """
    Build the 16-element NF4 codebook.
    1. Generate 8 evenly spaced probabilities for the negative half:
       from 0.03 to 0.5 (inclusive).
    2. The step size `d` is the difference between adjacent probabilities in step 1.
       Generate 8 evenly spaced probabilities for the positive half:
       from 0.5 + d to 0.97 (inclusive).
    3. Concatenate to get 16 probabilities.
    4. Pass each probability through `norm_ppf(p)`.
    5. Divide all 16 values by the maximum absolute value.
    """
    raise NotImplementedError()
