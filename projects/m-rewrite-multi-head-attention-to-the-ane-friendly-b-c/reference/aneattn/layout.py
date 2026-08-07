import numpy as np


def to_ane_friendly(x):
    b, h, s, d = x.shape
    return x.reshape(b, h * d, 1, s)
